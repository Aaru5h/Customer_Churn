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
dataset_stats = None  # Pre-computed aggregate statistics injected into the system prompt

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

def _compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Pre-compute aggregate churn statistics from the full dataset."""
    stats = {}
    total = len(df)
    churned = int(df['Exited'].sum())
    stats['total_customers'] = total
    stats['total_churned'] = churned
    stats['overall_churn_rate'] = round(churned / total * 100, 1)

    geo_stats = df.groupby('Geography')['Exited'].agg(['sum', 'count', 'mean'])
    stats['by_geography'] = {
        geo: {'churned': int(row['sum']), 'total': int(row['count']), 'rate': round(row['mean'] * 100, 1)}
        for geo, row in geo_stats.iterrows()
    }

    gender_stats = df.groupby('Gender')['Exited'].agg(['sum', 'count', 'mean'])
    stats['by_gender'] = {
        g: {'churned': int(row['sum']), 'total': int(row['count']), 'rate': round(row['mean'] * 100, 1)}
        for g, row in gender_stats.iterrows()
    }

    active_stats = df.groupby('IsActiveMember')['Exited'].agg(['mean'])
    stats['active_churn_rate'] = round(float(active_stats.loc[1, 'mean'] * 100), 1) if 1 in active_stats.index else None
    stats['inactive_churn_rate'] = round(float(active_stats.loc[0, 'mean'] * 100), 1) if 0 in active_stats.index else None

    product_stats = df.groupby('NumOfProducts')['Exited'].agg(['sum', 'count', 'mean'])
    stats['by_num_products'] = {
        int(n): {'churned': int(row['sum']), 'total': int(row['count']), 'rate': round(row['mean'] * 100, 1)}
        for n, row in product_stats.iterrows()
    }

    df_copy = df.copy()
    df_copy['age_bracket'] = pd.cut(
        df_copy['Age'], bins=[0, 30, 40, 50, 60, 120],
        labels=['18-30', '31-40', '41-50', '51-60', '60+']
    )
    age_stats = df_copy.groupby('age_bracket', observed=True)['Exited'].agg(['sum', 'count', 'mean'])
    stats['by_age_bracket'] = {
        str(b): {'churned': int(row['sum']), 'total': int(row['count']), 'rate': round(row['mean'] * 100, 1)}
        for b, row in age_stats.iterrows()
    }

    churned_df = df[df['Exited'] == 1]
    retained_df = df[df['Exited'] == 0]
    stats['churned_avg_balance'] = round(float(churned_df['Balance'].mean()), 0)
    stats['retained_avg_balance'] = round(float(retained_df['Balance'].mean()), 0)
    stats['churned_avg_credit_score'] = round(float(churned_df['CreditScore'].mean()), 1)
    stats['retained_avg_credit_score'] = round(float(retained_df['CreditScore'].mean()), 1)
    stats['churned_avg_age'] = round(float(churned_df['Age'].mean()), 1)
    stats['retained_avg_age'] = round(float(retained_df['Age'].mean()), 1)
    stats['churned_avg_tenure'] = round(float(churned_df['Tenure'].mean()), 1)
    stats['retained_avg_tenure'] = round(float(retained_df['Tenure'].mean()), 1)
    return stats


def _format_stats_for_prompt(stats: dict) -> str:
    """Format pre-computed statistics as a concise text block for the system prompt."""
    geo_lines = "\n".join(
        f"  - {g}: {v['rate']}% churn rate ({v['churned']:,} of {v['total']:,} customers)"
        for g, v in stats.get('by_geography', {}).items()
    )
    gender_lines = "\n".join(
        f"  - {g}: {v['rate']}% churn rate ({v['churned']:,} of {v['total']:,} customers)"
        for g, v in stats.get('by_gender', {}).items()
    )
    product_lines = "\n".join(
        f"  - {n} product(s): {v['rate']}% churn rate ({v['churned']:,} churned)"
        for n, v in sorted(stats.get('by_num_products', {}).items())
    )
    age_lines = "\n".join(
        f"  - Age {b}: {v['rate']}% churn rate"
        for b, v in sorted(stats.get('by_age_bracket', {}).items())
    )
    return f"""
=== VERIFIED DATASET STATISTICS (computed from all {stats['total_customers']:,} customers) ===
Overall: {stats['overall_churn_rate']}% churn rate ({stats['total_churned']:,} out of {stats['total_customers']:,} customers churned)

By Geography:
{geo_lines}

By Gender:
{gender_lines}

By Activity Status:
  - Active members: {stats.get('active_churn_rate', 'N/A')}% churn rate
  - Inactive members: {stats.get('inactive_churn_rate', 'N/A')}% churn rate

By Number of Products:
{product_lines}

By Age Bracket:
{age_lines}

Churned vs Retained — Average Profile:
  - Balance:      ${stats['churned_avg_balance']:>10,.0f} (churned)  vs  ${stats['retained_avg_balance']:>10,.0f} (retained)
  - Age:           {stats['churned_avg_age']} yrs (churned)  vs  {stats['retained_avg_age']} yrs (retained)
  - Credit Score:  {stats['churned_avg_credit_score']} (churned)  vs  {stats['retained_avg_credit_score']} (retained)
  - Tenure:        {stats['churned_avg_tenure']} yrs (churned)  vs  {stats['retained_avg_tenure']} yrs (retained)
=== END STATISTICS ===
"""


def init_faiss():
    """Reads CSV, pre-computes stats, embeds 2000 customer profiles, builds FAISS index."""
    global vector_store, embeddings, dataset_stats
    if vector_store is not None:
        return vector_store

    logger.info("Initializing FAISS Vector Store...")
    try:
        df = pd.read_csv(DATA_PATH)
        if 'CLIENTNUM' in df.columns:
            df = df.drop(columns=['CLIENTNUM'])

        # Pre-compute statistics from the FULL dataset before subsetting for embeddings
        logger.info("Pre-computing dataset aggregate statistics...")
        dataset_stats = _compute_dataset_stats(df)
        logger.info(f"Stats computed: {dataset_stats['total_customers']} customers, "
                    f"{dataset_stats['overall_churn_rate']}% overall churn rate.")

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

        # Embed up to 2000 profiles — good coverage while staying within free-tier RAM limits
        subset_docs = docs[:2000] if len(docs) > 2000 else docs
        logger.info(f"Embedding {len(subset_docs)} of {len(docs)} customer profiles...")

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

    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    llm_with_tools = llm.bind_tools(tools)
    llm_plain = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    # Build a static stats block once at agent-creation time (after init_faiss has run)
    stats_block = _format_stats_for_prompt(dataset_stats) if dataset_stats else ""

    SYSTEM_PROMPT = f"""You are NeuralVault's Bank Portfolio Security Analyst.
Your sole purpose is to analyze bank customer data, predict churn, and suggest retention strategies.

GUARDRAIL: If a user asks about anything unrelated to banking, customer retention,
churn predictions, or the data, decline with:
"I am a specialized banking AI. I can only assist with customer churn and retention analysis."

{stats_block}

INSTRUCTIONS FOR ANSWERING ANALYTICAL QUESTIONS:
- For aggregate questions (churn rates by country, gender, age, etc.) use the VERIFIED DATASET
  STATISTICS above — they cover all customers and are 100% accurate.
- Do NOT estimate, guess, or infer aggregate statistics from the small sample of retrieved profiles.
- For individual customer-level questions or examples, use the retrieved profiles provided below.
- Always cite specific numbers from the statistics when making comparisons.
- Format responses with clear bullet points, headers, and structured analysis.

IMPORTANT — Tool Usage Rules:
- You have EXACTLY two tools: `predict_churn_tool` and `save_retention_strategy`.
- You do NOT have a search tool. Do NOT attempt to call any search or retrieval function.
- Customer data is ALREADY provided to you in this prompt — just use it directly.
- Use the churn prediction tool ONLY when the user provides specific customer attributes (age, balance, etc.).
- Use the save strategy tool ONLY when the user explicitly asks to save a strategy.

Think carefully. Provide premium, structured, data-driven answers.
"""

    def agent_node(state: AgentState):
        messages = state["messages"]

        # Inline FAISS search — inject 10 most relevant profiles as supporting examples.
        # The aggregate statistics are already in the static system prompt above.
        rag_context = ""
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if user_messages and vector_store is not None:
            latest_query = user_messages[-1].content
            try:
                results = vector_store.similarity_search(latest_query, k=10)
                rag_context = "\n".join([r.page_content for r in results])
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        context_block = (
            f"\n\nRelevant individual customer profiles (examples from the database):\n{rag_context}"
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
