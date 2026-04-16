# Credit Card Customer Churn Predictor

A production-grade machine learning web application that predicts the probability of a credit card customer churning, identifies the key behavioral drivers of disengagement, and presents model performance insights — all powered by classical ML (Scikit-learn) with a Streamlit interface.

---

## Features

- **Churn probability prediction** with color-coded risk banners (🔴 HIGH / 🟢 LOW)
- **Two selectable models**: Logistic Regression and Decision Tree — switchable via sidebar
- **Top-5 prediction drivers** — per-customer feature importance chart
- **Model performance dashboard**: Confusion matrix, classification report, feature importance/coefficient chart
- **Full Scikit-learn Pipeline** — consistent preprocessing at training and prediction time
- **Streamlit Community Cloud ready** — deploy in minutes

---

## Tech Stack

| Layer         | Library                          |
|---------------|----------------------------------|
| Language      | Python 3.10+                     |
| ML Framework  | Scikit-learn (Pipelines)         |
| UI            | Streamlit 1.32                   |
| Visualization | Matplotlib 3.8, Seaborn 0.13     |
| Data          | Pandas 2.2, NumPy 1.26           |

---

## Dataset Setup

1. Download the **BankChurners** dataset (Credit Card Customer Churn dataset).
2. Place the file at the following path relative to the project root:

```
CHURN_PRJ/
└── data/
    └── BankChurners.csv
```

The app will not start without this file being present.

---

## Installation

```bash
# 1. Clone or download the project
cd CHURN_PRJ

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running Locally

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Project Structure

```
CHURN_PRJ/
├── app.py              # Streamlit UI — all rendering logic
├── model.py            # ML pipelines, training, evaluation, prediction
├── preprocessing.py    # Feature lists, ColumnTransformer, data loader
├── requirements.txt    # Pinned dependencies
├── README.md           # This file
└── data/
    └── BankChurners.csv  # Dataset (not included — see Dataset Setup)
```

---

## Deploying to Streamlit Community Cloud

1. Push the entire project (minus the CSV) to a **public GitHub repository**.
2. Add `BankChurners.csv` to the `data/` folder and commit it, **or** use [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management) to host it remotely and update the `DATA_PATH` in `app.py`.
3. Log in to [share.streamlit.io](https://share.streamlit.io).
4. Click **New app** → select your repository → set **Main file path** to `app.py`.
5. Click **Deploy** — the app will be live within a few minutes.

> **Note:** Streamlit Community Cloud uses Python 3.10 by default. All pinned versions in `requirements.txt` are validated against that runtime.

---

## Models

| Model               | Configuration                          |
|---------------------|----------------------------------------|
| Logistic Regression | `max_iter=1000`, `random_state=42`     |
| Decision Tree       | `max_depth=6`, `random_state=42`       |

Both models are trained on an **80/20 stratified train-test split** using a full Scikit-learn Pipeline:

```
ColumnTransformer (impute + scale/encode) → Classifier
```

---

## Evaluation Metrics

- Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
- Confusion Matrix (seaborn annotated heatmap)
- Classification Report (per-class breakdown)
- Feature Importances (Decision Tree) / Absolute Coefficients (Logistic Regression)
