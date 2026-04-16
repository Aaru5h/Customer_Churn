"""
app.py
------
Main Streamlit application for the Bank Customer Churn Predictor.
All UI rendering lives here. ML logic is delegated to model.py and
preprocessing.py — no training or prediction code appears in this file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model import (
    get_all_top_features,
    get_feature_names,
    get_top_features,
    predict,
    train_models,
)

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-themed styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c1121f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(233, 69, 96, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(233, 69, 96, 0.5);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        color: rgba(255,255,255,0.6);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #c1121f) !important;
        color: white !important;
    }

    /* Risk banners */
    .high-risk-banner {
        background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(193,18,31,0.15));
        border: 1px solid rgba(233,69,96,0.5);
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        margin: 16px 0;
    }
    .low-risk-banner {
        background: linear-gradient(135deg, rgba(0,200,100,0.15), rgba(0,150,70,0.1));
        border: 1px solid rgba(0,200,100,0.45);
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        margin: 16px 0;
    }

    /* Input widgets */
    .stNumberInput > div, .stSelectbox > div {
        border-radius: 8px;
    }

    /* Section headers */
    h2, h3 {
        color: #f0f0f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model training — cached so retraining only happens once per session.
# st.cache_resource is used (not st.experimental_cache).
# ---------------------------------------------------------------------------

DATA_PATH = "data/BankChurners.csv"


@st.cache_resource(show_spinner="Training models on the dataset — please wait…")
def load_trained_models():
    """
    Train both Logistic Regression and Decision Tree pipelines on the dataset
    and cache the result for the session lifetime.
    Returns the full results dict from model.train_models().
    """
    return train_models(DATA_PATH)


# Load models (will be cached after the first run)
try:
    model_results = load_trained_models()
    all_feature_names = model_results["feature_names"]
except FileNotFoundError:
    st.error(
        f"❌ Dataset not found at `{DATA_PATH}`. "
        "Please place `BankChurners.csv` inside the `data/` directory and restart the app."
    )
    st.stop()
except Exception as exc:
    st.error(f"❌ An error occurred while training the models: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — branding and model selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 24px;">
            <div style="font-size:2.8rem;">🏦</div>
            <h2 style="color:#e94560; margin:4px 0; font-size:1.2rem; font-weight:700;">
                Customer Churn<br>Predictor
            </h2>
            <p style="color:rgba(255,255,255,0.55); font-size:0.82rem; margin-top:8px;">
                Identify at-risk customers using classical machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Model selector radio button
    st.markdown("### 🤖 Select Model")
    selected_model = st.radio(
        label="Model",
        options=["Logistic Regression", "Decision Tree"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color:rgba(255,255,255,0.4); font-size:0.75rem; text-align:center;">
            Built with Scikit-learn · Streamlit<br>
            Classical ML · No LLMs
        </div>
        """,
        unsafe_allow_html=True,
    )

# Retrieve the selected model's pipeline and metrics
active_pipeline = model_results[selected_model]["pipeline"]
active_metrics  = model_results[selected_model]["metrics"]
X_test          = model_results[selected_model]["X_test"]
y_test          = model_results[selected_model]["y_test"]

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_predict, tab_performance = st.tabs(["🎯  Predict Churn", "📊  Model Performance"])

# ===========================================================================
# TAB 1 — Predict Churn
# ===========================================================================

with tab_predict:
    st.markdown("## Enter Customer Details")
    st.markdown(
        "Fill in the customer profile below and click **Predict Churn** to assess their risk level."
    )

    # -----------------------------------------------------------------------
    # Input form — 2-column layout
    # -----------------------------------------------------------------------
    with st.form(key="customer_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 👤 Demographics & Account")

            geography = st.selectbox(
                "Geography",
                options=["France", "Germany", "Spain"],
            )
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
            )
            age = st.number_input(
                "Customer Age", min_value=18, max_value=100, value=38, step=1
            )
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=850, value=650, step=1
            )
            tenure = st.slider(
                "Tenure (Years with Bank)", min_value=0, max_value=10, value=5
            )

        with col_right:
            st.markdown("#### 💰 Financial & Engagement")

            balance = st.number_input(
                "Account Balance ($)", min_value=0.0, value=50000.0, step=500.0
            )
            num_of_products = st.slider(
                "Number of Products Held", min_value=1, max_value=4, value=2
            )
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                options=["Yes", "No"],
            )
            is_active_member = st.selectbox(
                "Active Member?",
                options=["Yes", "No"],
            )
            estimated_salary = st.number_input(
                "Estimated Annual Salary ($)", min_value=0.0, value=75000.0, step=1000.0
            )

        # Full-width predict button inside the form
        predict_clicked = st.form_submit_button(
            "🔍 Predict Churn", use_container_width=True
        )

    # -----------------------------------------------------------------------
    # Prediction output — displayed after form submission
    # -----------------------------------------------------------------------

    if predict_clicked:
        # Assemble the input into a single-row DataFrame matching training columns
        input_data = pd.DataFrame(
            [
                {
                    "CreditScore": int(credit_score),
                    "Geography": geography,
                    "Gender": gender,
                    "Age": int(age),
                    "Tenure": int(tenure),
                    "Balance": float(balance),
                    "NumOfProducts": int(num_of_products),
                    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                    "EstimatedSalary": float(estimated_salary),
                }
            ]
        )

        # Run prediction through the cached pipeline (includes preprocessing)
        label, churn_prob = predict(active_pipeline, input_data)

        # Color-coded risk banner
        if churn_prob > 0.5:
            st.markdown(
                f"""
                <div class="high-risk-banner">
                    <div style="font-size:2.4rem;">🔴</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#e94560; margin-top:8px;">
                        HIGH CHURN RISK
                    </div>
                    <div style="font-size:1rem; color:rgba(255,255,255,0.7); margin-top:6px;">
                        This customer shows a strong likelihood of disengaging.
                        Consider targeted retention outreach.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="low-risk-banner">
                    <div style="font-size:2.4rem;">🟢</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#00c864; margin-top:8px;">
                        LOW CHURN RISK
                    </div>
                    <div style="font-size:1rem; color:rgba(255,255,255,0.7); margin-top:6px;">
                        This customer is likely to remain engaged.
                        Standard relationship management applies.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Churn probability metric row and progress bar
        st.markdown("")
        prob_col, spacer_col = st.columns([1, 2])
        with prob_col:
            st.metric(
                label="Churn Probability",
                value=f"{churn_prob * 100:.1f}%",
            )
        st.progress(churn_prob)

        st.markdown("---")

        # -----------------------------------------------------------------------
        # Top 5 feature drivers for this specific prediction
        # -----------------------------------------------------------------------
        st.subheader("🔍 Top Factors Influencing This Prediction")
        st.markdown(
            "The chart below shows the five features that most strongly drive "
            "the model's output for this particular customer profile."
        )

        top_names, top_values = get_top_features(
            active_pipeline, selected_model, all_feature_names, n=5
        )

        fig_drivers, ax_drivers = plt.subplots(figsize=(7, 3.5))
        fig_drivers.patch.set_facecolor("#1a1a2e")
        ax_drivers.set_facecolor("#1a1a2e")

        bar_color = "#e94560" if churn_prob > 0.5 else "#00c864"

        # Reverse so the most important feature appears at the top of the chart
        ax_drivers.barh(
            top_names[::-1],
            top_values[::-1],
            color=bar_color,
            edgecolor="none",
            height=0.55,
        )

        # Annotate bars with numeric values
        for i, (val) in enumerate(top_values[::-1]):
            ax_drivers.text(
                top_values[::-1][i] + max(top_values) * 0.01,
                i,
                f"{val:.4f}",
                va="center",
                ha="left",
                color="#cccccc",
                fontsize=9,
            )

        ax_drivers.set_xlabel("Importance / |Coefficient|", color="#aaaaaa", fontsize=10)
        ax_drivers.tick_params(colors="#cccccc", labelsize=10)
        ax_drivers.spines["top"].set_visible(False)
        ax_drivers.spines["right"].set_visible(False)
        ax_drivers.spines["left"].set_color("#333355")
        ax_drivers.spines["bottom"].set_color("#333355")
        fig_drivers.tight_layout()

        st.pyplot(fig_drivers)
        plt.close(fig_drivers)

# ===========================================================================
# TAB 2 — Model Performance
# ===========================================================================

with tab_performance:
    model_icon = "🌳" if selected_model == "Decision Tree" else "📈"
    st.markdown(f"## {model_icon} {selected_model} — Performance Report")
    st.markdown(
        f"Evaluation results on the **held-out 20% test set** for the "
        f"**{selected_model}** model."
    )

    # -----------------------------------------------------------------------
    # Top-level metric cards
    # -----------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{active_metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{active_metrics['precision']:.3f}")
    m3.metric("Recall",    f"{active_metrics['recall']:.3f}")
    m4.metric("F1-Score",  f"{active_metrics['f1']:.3f}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Confusion matrix and feature importance chart side-by-side
    # -----------------------------------------------------------------------
    chart_col_left, chart_col_right = st.columns(2)

    with chart_col_left:
        st.markdown("### Confusion Matrix")

        cm = active_metrics["confusion_matrix"]
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#1a1a2e")
        ax_cm.set_facecolor("#1a1a2e")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="magma",
            linewidths=0.8,
            linecolor="#333355",
            xticklabels=["Not Churned", "Churned"],
            yticklabels=["Not Churned", "Churned"],
            ax=ax_cm,
            cbar_kws={"shrink": 0.8},
        )
        ax_cm.set_xlabel("Predicted Label", color="#cccccc", fontsize=11)
        ax_cm.set_ylabel("True Label", color="#cccccc", fontsize=11)
        ax_cm.tick_params(colors="#cccccc")
        ax_cm.set_title("Confusion Matrix", color="#f0f0f0", fontsize=12, pad=10)
        fig_cm.tight_layout()

        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with chart_col_right:
        # Chart title and axis label differ by model type
        if selected_model == "Decision Tree":
            chart_title = "Top 15 Feature Importances"
            bar_color   = "#e94560"
            x_label     = "Gini Importance"
        else:
            chart_title = "Top 15 Feature Coefficients (|value|)"
            bar_color   = "#0d6efd"
            x_label     = "|Coefficient|"

        st.markdown(f"### {chart_title}")

        top15_names, top15_values = get_all_top_features(
            active_pipeline, selected_model, all_feature_names, n=15
        )

        fig_fi, ax_fi = plt.subplots(figsize=(5, 5.5))
        fig_fi.patch.set_facecolor("#1a1a2e")
        ax_fi.set_facecolor("#1a1a2e")

        # Gradient opacity for visual interest — most important bar is darkest
        n_bars  = len(top15_names)
        alphas  = np.linspace(1.0, 0.45, n_bars)
        r, g, b = plt.matplotlib.colors.to_rgb(bar_color)
        bar_colors = [(r, g, b, a) for a in alphas[::-1]]

        ax_fi.barh(
            top15_names[::-1],
            top15_values[::-1],
            color=bar_colors,
            edgecolor="none",
            height=0.65,
        )

        ax_fi.set_xlabel(x_label, color="#aaaaaa", fontsize=10)
        ax_fi.tick_params(colors="#cccccc", labelsize=8.5)
        ax_fi.spines["top"].set_visible(False)
        ax_fi.spines["right"].set_visible(False)
        ax_fi.spines["left"].set_color("#333355")
        ax_fi.spines["bottom"].set_color("#333355")
        ax_fi.set_title(chart_title, color="#f0f0f0", fontsize=11, pad=10)
        fig_fi.tight_layout()

        st.pyplot(fig_fi)
        plt.close(fig_fi)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Classification report as a styled DataFrame
    # -----------------------------------------------------------------------
    st.markdown("### Classification Report")
    report_df = active_metrics["classification_report"]

    # Identify float columns for formatting
    float_cols = [c for c in report_df.columns if report_df[c].dtype == float]

    st.dataframe(
        report_df.style.format(
            {c: "{:.3f}" for c in float_cols}
        ).background_gradient(cmap="YlOrRd", axis=0),
        use_container_width=True,
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Plain-English business interpretation of the metrics
    # -----------------------------------------------------------------------
    st.markdown("### 💡 What Do These Results Mean?")

    precision_pct = active_metrics["precision"] * 100
    recall_pct    = active_metrics["recall"]    * 100
    f1_pct        = active_metrics["f1"]        * 100

    if selected_model == "Logistic Regression":
        model_intro = (
            "Logistic Regression estimates the probability that a customer churns "
            "by fitting a linear decision boundary across all features. It is highly "
            "interpretable — each coefficient directly reflects the magnitude and "
            "direction of a feature's influence."
        )
    else:
        model_intro = (
            "The Decision Tree splits customers into progressively smaller groups "
            "based on the feature thresholds that best separate churners from "
            "non-churners, up to a maximum depth of 6 levels. It captures non-linear "
            "relationships and is intuitive to explain to business stakeholders."
        )

    st.markdown(
        f"""
{model_intro}

**Precision of {precision_pct:.1f}%** means that when the model flags a customer as
likely to churn, it is correct roughly {precision_pct:.0f}% of the time — reducing
wasted retention spend on customers who would have stayed anyway.

**Recall of {recall_pct:.1f}%** means the model successfully identifies about
{recall_pct:.0f}% of all customers who will actually churn, giving the retention
team maximum coverage to act before those customers leave.

**F1-Score of {f1_pct:.1f}%** balances both concerns — it is the harmonic mean of
precision and recall, and is the primary metric to optimize when churn classes are
imbalanced (far more existing customers than churned ones).
    """
    )
