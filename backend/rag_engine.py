"""
rag_engine.py
-------------
Initializes a local FAISS index on the churn CSV data for vector search,
and sets up the LangGraph agent equipped with the vector search tool
and the ML prediction tool, guarded by a relevance prompt.
"""

import os
import pandas as pd
from typing import Annotated, TypedDict, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
import joblib
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from model import predict, train_models
from backend.db import SessionLocal, RetentionStrategy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Initialization ---
DATA_PATH = "data/BankChurners.csv"
vector_store = None
embeddings = None
trained_pipeline_cache = None

def get_trained_pipeline():
    global trained_pipeline_cache
    if not trained_pipeline_cache:
        model_path = "backend/models/pipeline.joblib"
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            trained_pipeline_cache = joblib.load(model_path)
        else:
            logger.warning(f"Pre-trained model {model_path} not found. Training on the fly (CPU heavy!)...")
            model_results = train_models(DATA_PATH)
            trained_pipeline_cache = model_results["Logistic Regression"]["pipeline"]
    return trained_pipeline_cache

def init_faiss():
    """Reads CSV, creates text summaries, embeds them and returns FAISS index."""
    global vector_store, embeddings
    if vector_store is not None:
        return vector_store
        
    logger.info("Initializing FAISS Vector Store...")
    try:
        df = pd.read_csv(DATA_PATH)
        # Drop columns that are completely irrelevant for semantic search like ClientNUM
        if 'CLIENTNUM' in df.columns:
            df = df.drop(columns=['CLIENTNUM'])
            
        # Create a text representation for search
        def create_text_summary(row):
            return (
                f"Customer: {row.get('Age', '')} year old {row.get('Gender', '')} "
                f"from {row.get('Geography', '')}. "
                f"Tenure: {row.get('Tenure', '')} years. "
                f"Balance: {row.get('Balance', '')}. "
                f"Credit Score: {row.get('CreditScore', '')}. "
                f"Products: {row.get('NumOfProducts', '')}. "
                f"Active: {'Yes' if row.get('IsActiveMember') == 1 else 'No'}. "
                f"Churned: {'Yes' if row.get('Exited') == 1 else 'No'}."
            )
            
        df['page_content'] = df.apply(create_text_summary, axis=1)
        loader = DataFrameLoader(df, page_content_column="page_content")
        docs = loader.load()
        
        # Take a subset if the dataset is too huge to embed fully (for speed in this demo)
        # 10,000 rows x 384 dims takes a moment. Let's embed everything or at least first 2000.
        subset_docs = docs[:500] if len(docs) > 500 else docs
        
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            logger.error("HUGGINGFACEHUB_API_TOKEN not found. RAG will fail.")
            
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
        vector_store = FAISS.from_documents(subset_docs, embeddings)
        logger.info("FAISS Vector Store initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing FAISS: {e}")
    return vector_store

# --- Tools Definition ---

@tool
def search_customer_data(query: str) -> str:
    """
    Search historical banking data to find relevant customer profiles and churn patterns.
    Use this when the user asks about trends, historical customer behavior, or retention data.
    """
    if not vector_store:
        return "Vector store is not initialized."
    results = vector_store.similarity_search(query, k=5)
    context = "\n".join([r.page_content for r in results])
    return f"Relevant Historical Data found:\n{context}"

class CustomerProfileInput(BaseModel):
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float

@tool
def predict_churn_tool(
    credit_score: int, 
    geography: str, 
    gender: str, 
    age: int, 
    tenure: int, 
    balance: float, 
    num_of_products: int, 
    has_cr_card: int, 
    is_active_member: int, 
    estimated_salary: float
) -> str:
    """
    Predict the churn probability for a SPECIFIC customer profile using the classical ML model.
    Only call this if the user asks to predict risk for a specific user, or gives specific customer stats.
    Do not guess inputs; use the inputs provided by the user.
    """
    try:
        pipeline = get_trained_pipeline()
        input_data = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
        }])
        label, churn_prob = predict(pipeline, input_data)
        return (
            f"Prediction Result: {label}. "
            f"There is a {churn_prob*100:.1f}% probability of this customer churning."
        )
    except Exception as e:
        return f"Failed to predict churn: {str(e)}"

@tool
def save_retention_strategy(context: str, strategy: str) -> str:
    """
    Save a generated retention strategy to the database so it can be reference later.
    Call this when the user explicitly asks to save or log the strategy.
    """
    db = SessionLocal()
    try:
        strat = RetentionStrategy(context=context, strategy_text=strategy)
        db.add(strat)
        db.commit()
        return "Strategy successfully saved to database."
    except Exception as e:
        db.rollback()
        return f"Error saving strategy: {str(e)}"
    finally:
        db.close()

# search_customer_data is intentionally excluded from `tools` — it runs inline
# in agent_node before every LLM call, so the model never needs to call it via
# the tool-use API (which caused tool_use_failed 400 errors on Groq).
tools = [predict_churn_tool, save_retention_strategy]

# --- LangGraph Setup ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def create_agent():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # With only 2 tools (predict + save), the model rarely needs to call tools,
    # which makes malformed XML-style tool calls extremely unlikely.
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    llm_with_tools = llm.bind_tools(tools)
    llm_plain = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    SYSTEM_PROMPT = """You are the Bank Portfolio Security Analyst.
Your sole purpose is to analyze bank customer data, predict churn, and suggest retention strategies.

GUARDRAIL: If a user asks about anything unrelated to banking, customer retention,
churn predictions, or the data, decline with:
"I am a specialized banking AI. I can only assist with customer churn and retention analysis."

Relevant customer records from the database will be provided to you above.
Use this data to give specific, data-driven answers.

You have two tools available:
- Use the churn prediction tool ONLY when the user provides specific customer attributes (age, balance, etc.).
- Use the save strategy tool ONLY when the user explicitly asks to save a strategy.

Think carefully. Provide premium, structured answers.
"""

    def agent_node(state: AgentState):
        messages = state["messages"]

        # Always run FAISS search inline and inject results into the system prompt.
        # This eliminates the need for search_customer_data as a bound tool.
        rag_context = ""
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if user_messages and vector_store is not None:
            latest_query = user_messages[-1].content
            try:
                results = vector_store.similarity_search(latest_query, k=5)
                rag_context = "\n".join([r.page_content for r in results])
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        context_block = (
            f"\n\nRelevant customer data retrieved from the database:\n{rag_context}"
            if rag_context else ""
        )
        full_system_prompt = SYSTEM_PROMPT + context_block

        # Replace (or prepend) system message with the context-enriched version
        if any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content=full_system_prompt) if isinstance(m, SystemMessage) else m
                for m in messages
            ]
        else:
            messages = [SystemMessage(content=full_system_prompt)] + messages

        # Tier-1: LLM with tools (predict + save only)
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.warning(f"Tool-calling LLM failed ({type(e).__name__}): {e}")

        # Tier-2: plain LLM, no tools — RAG context still injected
        try:
            response = llm_plain.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Fallback plain LLM also failed: {e}")

        # Tier-3: static safe response — nothing can crash past this
        return {"messages": [AIMessage(content=(
            "I'm having trouble connecting right now. "
            "Please try again in a moment."
        ))]}
        
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there is no tool call, we finish
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return "end"
        # Otherwise if there is a tool call, we route to the tool node
        else:
            return "continue"

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    
    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_node)

    # Set the entrypoint
    workflow.add_edge(START, "agent")

    # Add a conditional edge
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Add edge from tools back to agent
    workflow.add_edge("action", "agent")

    return workflow.compile()
