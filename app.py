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
import matplotlib
matplotlib.use("Agg")
import os
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="NeuralVault · AI Retention Intelligence",
    page_icon="🔮",
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
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None


# ═══════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — Glassmorphism + Animated Gradients + Micro-interactions
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg-primary: #06070D;
        --bg-secondary: #0D0F1A;
        --surface-1: rgba(255, 255, 255, 0.025);
        --surface-2: rgba(255, 255, 255, 0.04);
        --surface-3: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-medium: rgba(255, 255, 255, 0.1);
        --text-primary: #F1F5F9;
        --text-secondary: rgba(241, 245, 249, 0.6);
        --text-muted: rgba(241, 245, 249, 0.35);
        --accent-blue: #3B82F6;
        --accent-indigo: #6366F1;
        --accent-violet: #8B5CF6;
        --accent-cyan: #06B6D4;
        --success: #22C55E;
        --danger: #EF4444;
        --warning: #F59E0B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* ── App Background with subtle noise ── */
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.12), transparent),
            radial-gradient(ellipse 60% 40% at 80% 60%, rgba(59, 130, 246, 0.06), transparent),
            radial-gradient(ellipse 50% 50% at 20% 80%, rgba(139, 92, 246, 0.05), transparent);
        color: var(--text-primary);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13, 15, 26, 0.95) 0%, rgba(6, 7, 13, 0.98) 100%) !important;
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: var(--surface-1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: var(--border-medium);
        transform: translateY(-1px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: var(--surface-2);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 0 24px rgba(99, 102, 241, 0.08);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700;
        font-size: 1.5rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
        filter: brightness(1.08);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface-1);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--text-muted);
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(59, 130, 246, 0.1)) !important;
        color: white !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.12);
    }

    /* ── Headers ── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    /* ── Form inputs ── */
    [data-testid="stNumberInput"] input, 
    [data-testid="stTextInput"] input,
    .stSelectbox [data-baseweb="select"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stNumberInput"] input:focus, 
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
    }

    /* ── Radio buttons ── */
    .stRadio label {
        color: var(--text-secondary) !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent-indigo) !important;
    }

    /* ── Chat bubbles ── */
    .chat-user {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(59, 130, 246, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 18px 18px 4px 18px;
        padding: 16px 20px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        color: var(--text-primary);
        font-size: 0.92rem;
        line-height: 1.6;
        animation: fadeInUp 0.3s ease;
    }
    .chat-ai {
        background: var(--surface-2);
        border: 1px solid var(--border-subtle);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        color: var(--text-primary);
        font-size: 0.92rem;
        line-height: 1.6;
        animation: fadeInUp 0.3s ease;
    }
    .chat-label {
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        display: block;
    }
    .chat-user .chat-label { color: var(--accent-indigo); }
    .chat-ai .chat-label { color: var(--accent-cyan); }

    /* ── Result cards ── */
    .result-card {
        border-radius: 20px;
        padding: 36px 32px;
        text-align: center;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        animation: fadeInScale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .result-high {
        background: linear-gradient(160deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.02));
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .result-low {
        background: linear-gradient(160deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.02));
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .result-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
    }
    .badge-high { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-low { background: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.3); }

    .result-percentage {
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 8px 0 4px;
    }
    .result-subtitle {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 400;
    }

    /* ── Metric Mini-Card ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .metric-mini {
        background: var(--surface-2);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-mini:hover {
        border-color: rgba(99, 102, 241, 0.25);
        transform: translateY(-2px);
    }
    .metric-mini .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-mini .metric-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-online {
        background: rgba(34, 197, 94, 0.1);
        color: var(--success);
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse-dot 2s infinite;
    }

    /* ── Section header ── */
    .section-header {
        margin-bottom: 6px;
    }
    .section-header h2 {
        font-size: 1.5rem;
        margin-bottom: 4px;
    }
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 0.88rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* ── Divider ── */
    .subtle-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 24px 0;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    .animate-in {
        animation: fadeInUp 0.4s ease both;
    }

    /* ── Tech stack badges ── */
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
    }
    .tech-badge {
        background: var(--surface-2);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    /* ── Hide streamlit defaults ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--surface-1) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Premium brand header
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 20px;">
            <div style="
                width: 56px; height: 56px; margin: 0 auto 12px;
                background: linear-gradient(135deg, #6366F1, #3B82F6);
                border-radius: 16px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.6rem;
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
            ">🔮</div>
            <h2 style="
                font-size: 1.3rem; margin: 0;
                background: linear-gradient(135deg, #E0E7FF, #C7D2FE, #A5B4FC);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 800;
                letter-spacing: -0.02em;
            ">NeuralVault</h2>
            <p style="color: rgba(241,245,249,0.4); font-size: 0.72rem; margin: 4px 0 0; font-weight: 500; text-transform: uppercase; letter-spacing: 0.12em;">
                AI Retention Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Engine Settings
    st.markdown(
        '<p style="color: rgba(241,245,249,0.4); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">⚙ Model Engine</p>',
        unsafe_allow_html=True,
    )
    selected_model = st.radio(
        label="Predictive Engine",
        options=["Logistic Regression", "Decision Tree"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # API Key section
    st.markdown(
        '<p style="color: rgba(241,245,249,0.4); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">🔑 Authentication</p>',
        unsafe_allow_html=True,
    )
    if not st.session_state.get("groq_api_key_set"):
        groq_api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            help="Required for the AI agent.",
            label_visibility="collapsed",
            placeholder="Enter Groq API Key...",
        )
        if groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
            st.session_state["groq_api_key_set"] = True

    if st.session_state.get("groq_api_key_set"):
        st.markdown(
            '<span class="status-pill status-online"><span class="status-dot"></span> Agent Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("API Key required for AI features", icon="🔒")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Tech stack
    st.markdown(
        """
        <p style="color: rgba(241,245,249,0.4); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">Stack</p>
        <div class="tech-stack">
            <span class="tech-badge">LangGraph</span>
            <span class="tech-badge">FAISS</span>
            <span class="tech-badge">Llama 3</span>
            <span class="tech-badge">FastAPI</span>
            <span class="tech-badge">Neon DB</span>
            <span class="tech-badge">scikit-learn</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VERIFY BACKEND  (with animated loading screen for cold-start wake-ups)
# ═══════════════════════════════════════════════════════════════════════════
if "wake_retries" not in st.session_state:
    st.session_state["wake_retries"] = 0

try:
    health = requests.get(f"{FASTAPI_URL}/", timeout=5)
    backend_up = health.status_code == 200
except Exception:
    backend_up = False

if backend_up:
    # Reset counter once the backend is alive
    st.session_state["wake_retries"] = 0
else:
    st.session_state["wake_retries"] += 1
    retries = st.session_state["wake_retries"]

    # Rotating status messages
    status_messages = [
        "Initialising neural pathways…",
        "Warming up ML inference engine…",
        "Loading predictive models into memory…",
        "Connecting to Neon database…",
        "Calibrating FAISS vector store…",
        "Bootstrapping LangGraph agent…",
        "Hydrating model weights…",
        "Spinning up FastAPI workers…",
    ]
    current_msg = status_messages[(retries - 1) % len(status_messages)]
    elapsed = retries * 5  # approx seconds

    st.markdown(
        f"""
        <style>
            /* ── Loading Orb ── */
            @keyframes orb-float {{
                0%, 100% {{ transform: translateY(0px) scale(1); }}
                50% {{ transform: translateY(-18px) scale(1.05); }}
            }}
            @keyframes orb-glow {{
                0%, 100% {{ box-shadow: 0 0 40px rgba(99,102,241,0.3), 0 0 80px rgba(99,102,241,0.1); }}
                50% {{ box-shadow: 0 0 60px rgba(99,102,241,0.5), 0 0 120px rgba(99,102,241,0.15); }}
            }}
            @keyframes ring-spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            @keyframes ring-spin-reverse {{
                from {{ transform: rotate(360deg); }}
                to {{ transform: rotate(0deg); }}
            }}
            @keyframes status-pulse {{
                0%, 100% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
            }}
            @keyframes progress-sweep {{
                0% {{ background-position: -200% 0; }}
                100% {{ background-position: 200% 0; }}
            }}
            @keyframes dot-bounce {{
                0%, 80%, 100% {{ transform: scale(0); opacity: 0.4; }}
                40% {{ transform: scale(1); opacity: 1; }}
            }}

            .wake-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 70vh;
                animation: fadeInUp 0.6s ease;
            }}

            /* Orb with rings */
            .orb-wrapper {{
                position: relative;
                width: 140px;
                height: 140px;
                margin-bottom: 40px;
            }}
            .orb-core {{
                position: absolute;
                top: 50%; left: 50%;
                width: 56px; height: 56px;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                background: linear-gradient(135deg, #6366F1, #3B82F6, #8B5CF6);
                animation: orb-float 3s ease-in-out infinite, orb-glow 3s ease-in-out infinite;
                z-index: 2;
            }}
            .orb-ring {{
                position: absolute;
                top: 50%; left: 50%;
                border-radius: 50%;
                border: 1.5px solid transparent;
            }}
            .orb-ring-1 {{
                width: 90px; height: 90px;
                margin: -45px 0 0 -45px;
                border-top-color: rgba(99,102,241,0.4);
                border-right-color: rgba(99,102,241,0.1);
                animation: ring-spin 3s linear infinite;
            }}
            .orb-ring-2 {{
                width: 120px; height: 120px;
                margin: -60px 0 0 -60px;
                border-bottom-color: rgba(59,130,246,0.3);
                border-left-color: rgba(59,130,246,0.08);
                animation: ring-spin-reverse 4s linear infinite;
            }}
            .orb-ring-3 {{
                width: 140px; height: 140px;
                margin: -70px 0 0 -70px;
                border-top-color: rgba(139,92,246,0.2);
                animation: ring-spin 6s linear infinite;
            }}

            .wake-title {{
                font-size: 1.4rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #E0E7FF, #C7D2FE, #A5B4FC);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .wake-subtitle {{
                color: rgba(241,245,249,0.45);
                font-size: 0.88rem;
                line-height: 1.6;
                max-width: 400px;
                text-align: center;
                margin-bottom: 32px;
            }}

            /* Shimmer progress bar */
            .progress-track {{
                width: 280px;
                height: 4px;
                background: rgba(255,255,255,0.06);
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 20px;
            }}
            .progress-fill {{
                height: 100%;
                border-radius: 4px;
                background: linear-gradient(90deg,
                    transparent 0%,
                    rgba(99,102,241,0.6) 30%,
                    rgba(59,130,246,0.8) 50%,
                    rgba(99,102,241,0.6) 70%,
                    transparent 100%
                );
                background-size: 200% 100%;
                animation: progress-sweep 1.8s ease-in-out infinite;
            }}

            /* Status text */
            .wake-status {{
                color: rgba(165,180,252,0.8);
                font-size: 0.78rem;
                font-weight: 500;
                letter-spacing: 0.02em;
                animation: status-pulse 2s ease infinite;
                margin-bottom: 8px;
            }}
            .wake-timer {{
                color: rgba(241,245,249,0.2);
                font-size: 0.7rem;
                font-weight: 400;
            }}

            /* Bouncing dots */
            .dots-row {{
                display: flex;
                gap: 6px;
                margin: 24px 0 0;
            }}
            .dots-row span {{
                width: 6px; height: 6px;
                background: rgba(99,102,241,0.5);
                border-radius: 50%;
                display: inline-block;
            }}
            .dots-row span:nth-child(1) {{ animation: dot-bounce 1.4s 0s infinite ease-in-out; }}
            .dots-row span:nth-child(2) {{ animation: dot-bounce 1.4s 0.16s infinite ease-in-out; }}
            .dots-row span:nth-child(3) {{ animation: dot-bounce 1.4s 0.32s infinite ease-in-out; }}
        </style>

        <div class="wake-container">
            <div class="orb-wrapper">
                <div class="orb-core"></div>
                <div class="orb-ring orb-ring-1"></div>
                <div class="orb-ring orb-ring-2"></div>
                <div class="orb-ring orb-ring-3"></div>
            </div>

            <div class="wake-title">Waking Up Neural Engine</div>
            <div class="wake-subtitle">
                Free-tier servers sleep after inactivity. The backend is booting up — 
                this usually takes 30–60 seconds. Hang tight!
            </div>

            <div class="progress-track">
                <div class="progress-fill"></div>
            </div>

            <div class="wake-status">{current_msg}</div>
            <div class="wake-timer">Attempt {retries} · ~{elapsed}s elapsed</div>

            <div class="dots-row">
                <span></span><span></span><span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Auto-retry every 5 seconds
    time.sleep(5)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
tab_predict, tab_ai, tab_performance = st.tabs([
    "🎯  Risk Assessment",
    "💬  AI Agent",
    "📊  Engine Metrics",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT CHURN
# ═══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        """
        <div class="section-header animate-in">
            <h2>Customer Risk Assessment</h2>
            <p class="section-subtitle">
                Run live inference against the <strong>%s</strong> engine to evaluate churn probability for a customer profile.
            </p>
        </div>
        """ % selected_model,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    with st.form(key="customer_form"):
        col_left, col_spacer, col_right = st.columns([5, 0.5, 5])

        with col_left:
            st.markdown(
                '<p style="color: var(--accent-indigo); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">👤 Demographics</p>',
                unsafe_allow_html=True,
            )
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

        with col_right:
            st.markdown(
                '<p style="color: var(--accent-cyan); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">💰 Financial Profile</p>',
                unsafe_allow_html=True,
            )
            balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=500.0)
            num_of_products = st.slider("Products Held", min_value=1, max_value=4, value=2)
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active_member = st.selectbox("Active Member?", ["Yes", "No"])
            estimated_salary = st.number_input("Est. Salary ($)", min_value=0.0, value=75000.0, step=1000.0)

        st.markdown("")  # spacer
        predict_clicked = st.form_submit_button("⚡ Run Prediction Pipeline", use_container_width=True)

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
            "model_type": selected_model,
        }

        with st.spinner("Neural Engine processing..."):
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            churn_prob = result["probability"]
            label = result["label"]

            is_high = churn_prob > 0.5
            color = "#EF4444" if is_high else "#22C55E"
            risk_class = "high" if is_high else "low"
            badge_text = "⚠ HIGH CHURN RISK" if is_high else "✓ LOW RISK"

            st.markdown(f"""
                <div class="result-card result-{risk_class}" style="margin: 24px 0;">
                    <span class="result-badge badge-{risk_class}">{badge_text}</span>
                    <div class="result-percentage" style="color: {color};">{churn_prob*100:.1f}%</div>
                    <p class="result-subtitle">Probability of Customer Disengagement</p>
                </div>
            """, unsafe_allow_html=True)

            # Detail breakdown
            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size: 1.6rem; margin-bottom: 4px;">🌍</div>
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;">Region</div>
                    <div style="font-weight: 700; font-size: 1.1rem; margin-top: 4px;">{geography}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size: 1.6rem; margin-bottom: 4px;">📊</div>
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;">Credit Score</div>
                    <div style="font-weight: 700; font-size: 1.1rem; margin-top: 4px;">{credit_score}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                risk_emoji = "🔴" if is_high else "🟢"
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size: 1.6rem; margin-bottom: 4px;">{risk_emoji}</div>
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;">Risk Level</div>
                    <div style="font-weight: 700; font-size: 1.1rem; margin-top: 4px; color: {color};">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        elif response.status_code == 503:
            st.warning("⏳ The ML engine is warming up. Please wait a moment and try again.", icon="⏳")
        else:
            st.error(f"Engine Error: {response.text}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — AI AGENT
# ═══════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown(
        """
        <div class="section-header animate-in">
            <h2>AI Retention Specialist</h2>
            <p class="section-subtitle">
                Conversational agent powered by <strong>LangGraph</strong>, <strong>Llama 3</strong>, and <strong>FAISS</strong> vector search.
                Ask about customer segments, churn drivers, or request retention strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    if not st.session_state.get("groq_api_key_set"):
        st.markdown(
            """
            <div style="
                text-align: center; padding: 60px 20px;
                background: var(--surface-1);
                border: 1px dashed var(--border-medium);
                border-radius: 20px;
                margin: 20px 0;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 12px; opacity: 0.4;">🔒</div>
                <h4 style="margin-bottom: 6px;">Agent Locked</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem;">
                    Enter your Groq API key in the sidebar to activate the AI agent.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Suggestion chips (only show when chat is empty)
        if len(st.session_state["chat_history"]) == 0:
            st.markdown(
                """
                <div style="
                    text-align: center; padding: 40px 20px 20px;
                    animation: fadeInUp 0.5s ease;
                ">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🧠</div>
                    <p style="color: var(--text-muted); font-size: 0.85rem;">Try asking something like...</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chip_cols = st.columns(3)
            suggestions = [
                "What common traits do churners share?",
                "Compare churn rates across Germany, France, and Spain",
                "Generate a retention strategy for inactive customers",
            ]
            for i, suggestion in enumerate(suggestions):
                with chip_cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        st.session_state["chat_history"].append({"role": "user", "content": suggestion})
                        st.rerun()

        # Display Chat History
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><span class="chat-label">You</span>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai"><span class="chat-label">Neural Agent</span>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Chat Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                placeholder="Ask about customer behaviour, retention strategy, or churn trends...",
                label_visibility="collapsed",
            )
            col_send, col_clear = st.columns([4, 1])
            with col_send:
                submitted = st.form_submit_button("Send →", use_container_width=True)
            with col_clear:
                clear = st.form_submit_button("Clear", use_container_width=True)

        if clear:
            st.session_state["chat_history"] = []
            st.rerun()

        if submitted and user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            st.rerun()

        elif (
            len(st.session_state["chat_history"]) > 0
            and st.session_state["chat_history"][-1]["role"] == "user"
        ):
            user_text = st.session_state["chat_history"][-1]["content"]
            with st.spinner("Agent routing tools and analyzing context..."):
                payload = {
                    "session_id": st.session_state["session_id"],
                    "message": user_text,
                }
                try:
                    res = requests.post(f"{FASTAPI_URL}/chat", json=payload, timeout=120)
                    if res.status_code == 200:
                        ai_reply = res.json()["response"]
                        st.session_state["chat_history"].append({"role": "ai", "content": ai_reply})
                        st.rerun()
                    else:
                        st.error(f"Agent Error: {res.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The agent might be processing a complex query.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_performance:
    st.markdown(
        f"""
        <div class="section-header animate-in">
            <h2>Engine Performance</h2>
            <p class="section-subtitle">
                Real-time evaluation metrics from the <strong>{selected_model}</strong> pipeline, served by the backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    try:
        res = requests.get(f"{FASTAPI_URL}/metrics?model_type={selected_model}", timeout=30)

        if res.status_code == 200:
            metrics = res.json()

            # Custom metric grid using HTML for premium look
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-mini">
                    <div class="metric-value">{metrics['accuracy']:.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-mini">
                    <div class="metric-value">{metrics['precision']:.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-mini">
                    <div class="metric-value">{metrics['recall']:.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-mini">
                    <div class="metric-value">{metrics['f1']:.3f}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            # Interpretability card
            st.markdown("""
            <div class="glass-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem;">🧬</span>
                    <h4 style="margin: 0; font-size: 1rem;">Interpretability & Deep Analysis</h4>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.7; margin: 0;">
                    These metrics are computed from the backend ML pipeline on the holdout test set.
                    For deeper feature-level analysis, coefficient exploration, or SHAP-style interpretability, 
                    switch to the <strong>AI Agent</strong> tab and ask it to analyze model behavior with natural language.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif res.status_code == 503:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">⏳</div>
                <h4 style="margin-bottom: 8px;">ML Engine Warming Up</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem;">
                    The predictive models are being loaded into memory.<br>
                    This happens once after a cold start. Refresh in a few seconds.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
                <h4 style="margin-bottom: 8px;">Metrics Loading</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem;">
                    The backend is preparing model metrics. Please try again shortly.
                </p>
            </div>
            """, unsafe_allow_html=True)
    except requests.exceptions.Timeout:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">⏳</div>
            <h4 style="margin-bottom: 8px;">Models Training</h4>
            <p style="color: var(--text-secondary); font-size: 0.88rem;">
                The ML models are being trained for the first time. This can take 15–30 seconds.<br>
                The page will automatically show results on next refresh.
            </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Connection error: {e}")
