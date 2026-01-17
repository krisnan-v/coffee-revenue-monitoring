import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Coffee Shop Model Monitoring & Feedback", layout="wide")
st.title("Coffee Shop Model Monitoring & Feedback Dashboard")

# ---------- Load Logs ----------
@st.cache_data
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

logs = load_logs()

# Handle "no logs yet"
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Please run the prediction app, submit feedback at least once, and then refresh this page."
    )
    st.stop()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

if selected_model == "All":
    filtered = logs
else:
    filtered = logs[logs["model_version"] == selected_model]

# ---------- Key Metrics ----------
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(filtered))

if filtered["feedback_score"].notna().any():
    col2.metric("Avg Feedback Score", f"{filtered['feedback_score'].mean():.2f}")
else:
    col2.metric("Avg Feedback Score", "N/A")

if filtered["latency_ms"].notna().any():
    col3.metric("Avg Latency (ms)", f"{filtered['latency_ms'].mean():.1f}")
else:
    col3.metric("Avg Latency (ms)", "N/A")

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "☕ Coffee Type & Roast Analysis", "📄 Raw Logs"])

# --- Tab 1: Model Comparison ---
with tab1:
    st.subheader("Model Version Comparison (Aggregated)")
    summary = logs.groupby("model_version").agg({
        "feedback_score": "mean",
        "latency_ms": "mean",
    })
    summary = summary.rename(columns={
        "feedback_score": "avg_feedback_score",
        "latency_ms": "avg_latency_ms",
    })
    st.dataframe(summary.style.format({
        "avg_feedback_score": "{:.2f}",
        "avg_latency_ms": "{:.1f}",
    }))

# --- Tab 2: Coffee Type & Roast Analysis ---
with tab2:
    st.subheader("Average Feedback Score by Coffee Type")
    if "input_summary" in logs.columns:
        # Extract coffee type from input_summary string
        logs["coffee_type"] = logs["input_summary"].str.extract(r"coffee=(\w+)")
        fb_coffee = logs.groupby("coffee_type")["feedback_score"].mean().reset_index()
        if not fb_coffee.empty:
            st.bar_chart(fb_coffee.set_index("coffee_type"))
        else:
            st.info("No coffee type feedback yet.")

    st.subheader("Average Feedback Score by Roast Type")
    if "input_summary" in logs.columns:
        logs["roast_type"] = logs["input_summary"].str.extract(r"roast=(\w+)")
        fb_roast = logs.groupby("roast_type")["feedback_score"].mean().reset_index()
        if not fb_roast.empty:
            st.bar_chart(fb_roast.set_index("roast_type"))
        else:
            st.info("No roast type feedback yet.")

    st.subheader("Recent Comments")
    comments = logs.copy()
    comments = comments[comments["feedback_text"].astype(str).str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)

    if comments.empty:
        st.info("No qualitative comments yet.")
    else:
        for _, row in comments.iterrows():
            st.write(f"**[{row['timestamp']}] {row['model_version']} – Score: {row['feedback_score']}**")
            st.write(row["feedback_text"])
    st.markdown("---")

# --- Tab 3: Raw Logs ---
with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)