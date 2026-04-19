# AI Agentic Customer Churn Predictor

A production-grade, full-stack machine learning application that predicts the probability of credit card customer churn. It features a classical ML predictive core powered by Scikit-learn, combined with a cutting-edge **LangGraph & FAISS AI Retention Specialist Agent**.

This architecture runs on a **FastAPI** backend ("The Brain") and a **Streamlit** frontend ("The Face") styled with a premium Glassmorphism design and highly secure guardrails.

---

## 🔥 Key Features

- **Agentic AI Specialist**: Chat with a LangGraph AI agent powered by Groq (Llama 3) that can retrieve historical database trends (FAISS) and run live ML predictions on explicit command.
- **Unbreakable Guardrails**: AI persona is strictly limited to banking and churn analysis. Unrelated queries are rejected.
- **Hybrid Database**: Local `FAISS` for semantic vector search and `SQLite` for persistent Agent Chat History and Strategies.
- **Full-Stack Separation**: A complete decouple of the UI from the Machine Learning logic. Fast frontend iteration, stable backend training.
- **Premium Glass UI**: A visually breathtaking "Dark Mode" aesthetic using the Outfit typeface and glassmorphism styling.

---

## 🛠️ Tech Stack

| Layer          | Technology                                           |
|----------------|------------------------------------------------------|
| **Frontend**   | Streamlit 1.32 (Requests, Custom CSS)                |
| **Backend**    | FastAPI, Uvicorn, SQLite (SQLAlchemy)                |
| **ML Core**    | Scikit-learn, Pandas, NumPy                          |
| **AI Agent**   | LangGraph, LangChain, Groq API (Llama 3)             |
| **Vector DB**  | FAISS, HuggingFace Sentence Transformers             |

---

## 📂 Dataset Setup

1. Download the **BankChurners** dataset.
2. Place the file at the following path relative to the project root:

```
CHURN_PRJ/
└── data/
    └── BankChurners.csv
```

The FastAPI backend will read this to train the classical ML models and build the FAISS Vector Index on startup.

---

## 🚀 Installation & Setup

```bash
# 1. Clone or download the project
cd CHURN_PRJ

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the System Locally

Because this is a full-stack project, you must run both the backend server and the frontend client simultaneously in separate terminal windows.

### Terminal 1: Start the FastAPI Backend (The Brain)
```bash
# Set your Groq API key if you want to use the AI Agent in the backend environment
export GROQ_API_KEY="your-groq-api-key"

uvicorn backend.main:app --reload
```
The backend runs on `http://127.0.0.1:8000`. You can access the API Swagger documentation at `http://127.0.0.1:8000/docs`.

### Terminal 2: Start the Streamlit Frontend (The Face)
```bash
streamlit run app.py
```
The UI runs automatically at `http://localhost:8501`.

---

## 🌐 Cloud Deployment (Render)

This project is configured for one-click deployment using **Render Blueprints**.

### 1. Database Setup (Neon)
- Create a free account at [Neon.tech](https://neon.tech).
- Create a new project and copy the **Connection String** (`postgresql://...`).

### 2. Render Deployment
- Push your code to a GitHub repository.
- In the Render Dashboard, click **New +** > **Blueprint**.
- Select your repository.
- Render will automatically detect the `render.yaml` file and create:
    - `churn-prediction-backend` (FastAPI)
    - `churn-prediction-frontend` (Streamlit)

### 3. Environment Variables
Ensure the following are set in your Render services:
- `GROQ_API_KEY`: Your Groq API key (Set in the Backend service).
- `DATABASE_URL`: Your Neon Postgres connection string (Set in the Backend service).

---

## 🤖 Using the AI Agent

1. Open the Streamlit App.
2. Ensure the Backend is running (if not, the UI will warn you).
3. Open the sidebar and enter your **Groq API Key**.
4. Navigate to the **💬 AI Agent** tab.
5. Ask questions like:
   - *"What are the main traits of churned customers in Germany?"*
   - *"Suggest a retention strategy for older customers with high balances."*
   - *"Analyze churn risk for a male, 42 years old, 700 credit score, in France, 4 tenure, 40000 balance."*

---

## 🛡️ Architecture & Security

- **Pydantic Validation**: All frontend requests are strictly validated before entering the ML pipeline.
- **Agent Guardrails**: The LangGraph state machine enforces the `SYSTEM_PROMPT` recursively. It physically lacks tools to execute system operations or non-banking domain reasoning.
- **Data Privacy**: Vector search runs entirely locally via FAISS and `sentence-transformers/all-MiniLM-L6-v2`. No raw tabular data is sent to external APIs (only context segments specifically requested in chat).

---

## 👥 Team Details

- **Aarush Gupta :** 2401020001
- **Utkarsh :** 2401010484
