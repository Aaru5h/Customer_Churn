"""
main.py
-------
FastAPI entry point ("The Brain").
Serves endpoints for classical ML predictions and RAG-based AI chat.
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env if present

# Add parent directory to path so we can import model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd

from model import train_models, predict, get_feature_names
from model import train_models, predict, get_feature_names
from backend.db import init_db, get_db, ChatMessage, SessionLocal
from backend.rag_engine import init_faiss, create_agent
from langchain_core.messages import HumanMessage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
DATA_PATH = "data/BankChurners.csv"
model_results = None
agent_app = None

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI - Initializing Models and DBs...")
    global model_results, agent_app
    
    # 1. Initialize SQLite Database
    init_db()
    
    # 2. Models are loaded lazily when needed
    logger.info("Models will be loaded lazily to save RAM.")
        
    # 3. FAISS + Agent are lazy-loaded on first /chat request to save RAM
    #    (sentence-transformers model is ~200MB — too heavy for startup on free tier)
    logger.info("FAISS and Agent will be lazy-loaded on first chat request.")
        
    yield
    # Shutdown
    logger.info("Shutting down FastAPI...")

app = FastAPI(title="Churn Predictor & AI Retention Agent", lifespan=lifespan)

# Add CORS Middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Schemas ---

class PredictRequest(BaseModel):
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
    model_type: str = "Logistic Regression"

class PredictResponse(BaseModel):
    label: str
    probability: float

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Brain is operational."}

@app.get("/metrics")
def get_metrics(model_type: str = "Logistic Regression"):
    """Fetch stored performance metrics for the requested model."""
    if not model_results or model_type not in model_results:
        raise HTTPException(status_code=404, detail="Model metrics not found.")
    
    metrics = model_results[model_type]["metrics"]
    
    # Return serializable metrics
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }

@app.post("/predict", response_model=PredictResponse)
def predict_churn_endpoint(req: PredictRequest):
    """Direct inference via classical ML models."""
    if not model_results or req.model_type not in model_results:
        raise HTTPException(status_code=500, detail="Model not initialized or wrong model type.")
        
    pipeline = model_results[req.model_type]["pipeline"]
    
    # Create single-row dataframe matching expected columns
    input_df = pd.DataFrame([{
        "CreditScore": req.credit_score,
        "Geography": req.geography,
        "Gender": req.gender,
        "Age": req.age,
        "Tenure": req.tenure,
        "Balance": req.balance,
        "NumOfProducts": req.num_of_products,
        "HasCrCard": req.has_cr_card,
        "IsActiveMember": req.is_active_member,
        "EstimatedSalary": req.estimated_salary,
    }])
    
    label, prob = predict(pipeline, input_df)
    return {"label": label, "probability": prob}

@app.post("/chat", response_model=ChatResponse)
def ai_chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """Agentic chat interface powered by LangGraph, with SQLite history."""
    global agent_app
    
    # Lazy-load FAISS + Agent on first chat request to save RAM at startup
    if not agent_app:
        try:
            logger.info("Lazy-loading FAISS and LangGraph Agent...")
            init_faiss()
            agent_app = create_agent()
            logger.info("Agent initialized successfully on first request.")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}")
            raise HTTPException(status_code=500, detail=f"AI Agent failed to initialize: {str(e)}")
        
    # User message logging
    user_msg = ChatMessage(session_id=req.session_id, role="user", content=req.message)
    db.add(user_msg)
    db.commit()
    
    # Fetch conversational history limit to recent to keep context window safe
    history_records = db.query(ChatMessage).filter(ChatMessage.session_id == req.session_id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
    history_records.reverse()
    
    langchain_messages = []
    for record in history_records:
        if record.role == "user":
            langchain_messages.append(HumanMessage(content=record.content))
        else:
            langchain_messages.append({"role": "assistant", "content": record.content}) # Or AIMessage if preferred
            
    # Include the current message in the state
    state = {"messages": langchain_messages}
    
    try:
        final_state = agent_app.invoke(state)
        # The last message is the AI's response
        ai_response_content = final_state["messages"][-1].content
    except Exception as e:
        logger.error(f"Error invoking LangGraph Agent: {str(e)}")
        ai_response_content = f"Error communicating with AI Brain: {str(e)}"
        
    # AI response logging
    ai_msg = ChatMessage(session_id=req.session_id, role="ai", content=ai_response_content)
    db.add(ai_msg)
    db.commit()

    return {"response": ai_response_content}
