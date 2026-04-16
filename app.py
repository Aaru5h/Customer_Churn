"""
app.py
------
Frontend Streamlit client ("The Face").
Connects to the FastAPI backend strictly via HTTP requests. 
Features Premium Glassmorphism UI and LangGraph AI integration.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Churn Predictor & AI Retention Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure session state variables exist
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "groq_api_key_set" not in st.session_state:
    st.session_state["groq_api_key_set"] = bool(os.environ.get("GROQ_API_KEY"))

# --- Custom Premium CSS (Glassmorphism & Dark Mode) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Premium Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        color: #F8FAFC;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Metric & Glass Cards */
    [data-testid="stMetric"], .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
    }

    /* Primary Accent Button */
    .stButton > button {
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.5);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 500;
        color: rgba(255,255,255,0.5);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Chat bubbles */
    .chat-user {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-bottom: 12px;
        max-width: 85%;
        margin-left: auto;
    }
    .chat-ai {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
        max-width: 85%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 24px;">
            <div style="font-size:3rem; filter: drop-shadow(0 0 10px rgba(56,189,248,0.5));">🏦</div>
            <h2 style="background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin:4px 0; font-size:1.4rem;">
                AI Retention<br>Specialist
            </h2>
            <p style="color:rgba(248,250,252,0.6); font-size:0.85rem; margin-top:8px;">
                Agentic workflows & ML Core
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    
    # Model selector radio button
    st.markdown("### ⚙️ Engine Settings")
    selected_model = st.radio(
        label="Predictive Engine",
        options=["Logistic Regression", "Decision Tree"],
        index=0,
    )
    
    st.markdown("---")
    
    # Try to get key from env, otherwise ask user
    if not st.session_state.get("groq_api_key_set"):
        groq_api_key_input = st.text_input("Groq API Key (For AI)", type="password", help="Required to use the AI Insights Agent.")
        if groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
            st.session_state["groq_api_key_set"] = True
    
    if st.session_state.get("groq_api_key_set"):
        st.success("API Key active.", icon="🔐")
    else:
        st.warning("Enter Groq API Key to enable AI.", icon="⚠️")

# --- Verify Backend Connection ---
try:
    health = requests.get(f"{FASTAPI_URL}/")
    backend_up = health.status_code == 200
except (requests.exceptions.ConnectionError, Exception):
    backend_up = False

if not backend_up:
    st.error("🚨 Backend Server is not running. Please start FastAPI: `uvicorn backend.main:app --reload`")
    st.stop()


# --- Main Tab Layout ---
tab_predict, tab_ai, tab_performance = st.tabs(["🎯 Predict Risk", "💬 AI Agent", "📊 Core Engine Metrics"])

# ===========================================================================
# TAB 1 — Predict Churn (Uses FastAPI /predict)
# ===========================================================================
with tab_predict:
    st.markdown("## Customer Risk Assessment")
    st.markdown("<p style='color: rgba(248,250,252,0.7);'>Run live inference using the selected core predictive engine.</p>", unsafe_allow_html=True)

    with st.form(key="customer_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 👤 Demographics")
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

        with col_right:
            st.markdown("#### 💰 Financials")
            balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=500.0)
            num_of_products = st.slider("Products Held", min_value=1, max_value=4, value=2)
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active_member = st.selectbox("Active Member?", ["Yes", "No"])
            estimated_salary = st.number_input("Est. Salary ($)", min_value=0.0, value=75000.0, step=1000.0)

        predict_clicked = st.form_submit_button("🔍 Run Prediction Pipeline", use_container_width=True)

    if predict_clicked:
        payload = {
            "credit_score": int(credit_score),
            "geography": geography,
            "gender": gender,
            "age": int(age),
            "tenure": int(tenure),
            "balance": float(balance),
            "num_of_products": int(num_of_products),
            "has_cr_card": 1 if has_cr_card == "Yes" else 0,
            "is_active_member": 1 if is_active_member == "Yes" else 0,
            "estimated_salary": float(estimated_salary),
            "model_type": selected_model
        }
        
        with st.spinner("Processing via FastAPI..."):
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            churn_prob = result["probability"]
            label = result["label"]
            
            st.markdown("---")
            
            if churn_prob > 0.5:
                color, bg = "#EF4444", "rgba(239, 68, 68, 0.1)"
                title = "HIGH CHURN RISK"
            else:
                color, bg = "#10B981", "rgba(16, 185, 129, 0.1)"
                title = "LOW RISK"
                
            st.markdown(f"""
                <div style="background:{bg}; border: 1px solid {color}; border-radius: 16px; padding: 24px; text-align: center; backdrop-filter: blur(10px);">
                    <h2 style="color:{color}; margin:0;">{title}</h2>
                    <h1 style="color:{color}; font-size: 3rem; margin:10px 0;">{churn_prob*100:.1f}%</h1>
                    <p style="color:rgba(248,250,252,0.8); margin:0;">Probability of Disengagement</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Progress bar matching color
            st.progress(churn_prob)
            
        else:
            st.error(f"Error from API: {response.text}")


# ===========================================================================
# TAB 2 — AI Agent (Uses FastAPI /chat)
# ===========================================================================
with tab_ai:
    st.markdown("## AI Retention Specialist")
    st.markdown("<p style='color: rgba(248,250,252,0.7);'>Powered by LangGraph, Llama 3, and FAISS Vector Search.</p>", unsafe_allow_html=True)
    
    if not st.session_state.get("groq_api_key_set"):
        st.info("👈 Please enter your Groq API Key in the sidebar to activate the Agent.")
    else:
        # Display Chat History 
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><b>AI Agent:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                
        # Chat Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask about customer trends, request a risk analysis, or generate a retention strategy...")
            submitted = st.form_submit_button("Send to Agent")
            
        if submitted and user_input:
            # Render user msg immediately
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            st.rerun() # Refresh to show user message while loading
            
        elif len(st.session_state["chat_history"]) > 0 and st.session_state["chat_history"][-1]["role"] == "user":
            # If last message was user, fetch AI response
            user_text = st.session_state["chat_history"][-1]["content"]
            with st.spinner("Agent is analyzing context and routing tools..."):
                payload = {
                    "session_id": st.session_state["session_id"],
                    "message": user_text
                }
                res = requests.post(f"{FASTAPI_URL}/chat", json=payload)
                
                if res.status_code == 200:
                    ai_reply = res.json()["response"]
                    st.session_state["chat_history"].append({"role": "ai", "content": ai_reply})
                    st.rerun()
                else:
                    st.error(f"Agent Error: {res.text}")


# ===========================================================================
# TAB 3 — Model Performance (Uses FastAPI /metrics)
# ===========================================================================
with tab_performance:
    st.markdown(f"## {selected_model} Metrics")
    
    with st.spinner("Fetching Core Engine Metrics..."):
        res = requests.get(f"{FASTAPI_URL}/metrics?model_type={selected_model}")
        
    if res.status_code == 200:
        metrics = res.json()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h4>Interpretability</h4>
            <p style="color: rgba(248,250,252,0.7);">
            Metrics are provided directly from the backend pipeline. 
            For deeper variable analysis and feature coefficient insights, 
            please query the <b>AI Agent</b> to interpret the semantic data relations in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Could not fetch metrics. Check backend server.")
