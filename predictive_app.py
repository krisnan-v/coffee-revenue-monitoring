# predictive_app.py

import time
import joblib
import pandas as pd
import streamlit as st
from log_utils import log_prediction

st.set_page_config(page_title="Coffee Shop Revenue Prediction App",
                   layout="centered")

st.title("Coffee Shop Revenue Prediction App with Live Monitoring")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    old_model = joblib.load("revenue_model_v1.pkl")  # trained on Size only
    new_model = joblib.load("revenue_model_v2.pkl")  # trained on Size + Coffee Type + Roast Type
    return old_model, new_model

old_model, new_model = load_models()

# ---------- Initialise session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "old_pred" not in st.session_state:
    st.session_state["old_pred"] = None
if "new_pred" not in st.session_state:
    st.session_state["new_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- INPUT SECTION ----------
st.sidebar.header("Order Parameters")

size = st.sidebar.slider("Coffee Size (kg)", min_value=0.2, max_value=2.5, step=0.1, value=1.0)
coffee_type = st.sidebar.selectbox("Coffee Type", ["Arabica", "Robusta", "Excelsa", "Liberica"])
roast_type = st.sidebar.selectbox("Roast Type", ["Light", "Medium", "Dark"])

# Canonical input dataframe
input_df = pd.DataFrame({
    "Size_clean": [size],
    "Coffee Type Name": [coffee_type],
    "Roast Type Name": [roast_type],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # v1: baseline – only uses Size
    input_v1 = input_df[["Size_clean"]]
    old_pred = old_model.predict(input_v1)[0]

    # v2: improved – uses Size + Coffee Type + Roast Type
    input_v2 = input_df[["Size_clean", "Coffee Type Name", "Roast Type Name"]]
    new_pred = new_model.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store in session_state
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = f"size={size}kg, coffee={coffee_type}, roast={roast_type}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"Old Model (v1 - Size only): **${st.session_state['old_pred']:,.2f}**")
    st.write(f"New Model (v2 - Size + Coffee + Roast): **${st.session_state['new_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)

feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )
        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )