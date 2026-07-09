import sys
from pathlib import Path
import os
import re
import datetime
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Ensure the local path is in sys.path so we can import automl_agents
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv(ROOT / ".env")
# Fallback to backend .env if it exists
backend_env = Path("C:/Users/dwive/OneDrive/Desktop/nimbus/.env")
if backend_env.exists():
    load_dotenv(backend_env)

# Page configuration
st.set_page_config(
    page_title="Nimbus AutoML Dashboard",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [data-testid="stAppViewContainer"], .stWidget, .stMarkdown {
    font-family: 'Inter', sans-serif !important;
}

/* Premium gradient titles */
.logo-container {
    padding: 1rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(139, 127, 214, 0.15);
}

.gradient-text {
    background: linear-gradient(135deg, #A78BFA, #8B7FD6, #4FD6C4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin: 0;
    line-height: 1.2;
}

.gradient-subtitle {
    font-size: 1.15rem;
    color: #A0AEC0;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Glassmorphism card container */
.glass-card {
    background: rgba(30, 30, 47, 0.45);
    border-radius: 12px;
    border: 1px solid rgba(139, 127, 214, 0.2);
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    margin-bottom: 20px;
}

/* Metric card */
.metric-box {
    text-align: center;
    padding: 18px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(139, 127, 214, 0.15);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(79, 214, 196, 0.4);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4FD6C4;
    margin-bottom: 5px;
}
.metric-label {
    font-size: 0.8rem;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Badges for selected features */
.feature-badge {
    display: inline-block;
    background: rgba(139, 127, 214, 0.12);
    color: #A78BFA;
    border: 1px solid rgba(139, 127, 214, 0.25);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    margin: 5px;
    font-weight: 600;
    transition: all 0.2s;
}
.feature-badge:hover {
    background: rgba(139, 127, 214, 0.22);
    border-color: rgba(139, 127, 214, 0.45);
    transform: scale(1.02);
}

/* Subtitle headings */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 15px;
    border-left: 4px solid #8B7FD6;
    padding-left: 10px;
}

/* Hide default streamlit elements for custom look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Try importing backend packages, show informative warning if not found
try:
    from automl_agents.graph.pipeline import graph
    from automl_agents.schemas import RunConfig
    from automl_agents.tools.model_export import load_model_bundle
    backend_available = True
except ImportError as e:
    backend_available = False
    st.error(f"❌ Backend package `automl_agents` could not be loaded. Please ensure the backend is copied into the frontend repository directory. (Error: {e})")

def generate_synthetic_churn_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic churn dataset mirroring nimbus/scripts/generate_synthetic.py"""
    rng = np.random.default_rng(seed)
    
    age = rng.integers(18, 80, size=n_rows)
    income = rng.normal(55_000, 18_000, size=n_rows).clip(15_000, 200_000)
    tenure_months = rng.integers(0, 120, size=n_rows)
    usage_gb = rng.gamma(shape=2.5, scale=12.0, size=n_rows)
    
    segment = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.45, 0.35, 0.20])
    region = rng.choice(["north", "south", "east", "west"], size=n_rows)
    
    # Mixed-type column
    score_raw = rng.normal(650, 80, size=n_rows).clip(300, 850)
    score_str = score_raw.round(0).astype(int).astype(str)
    bad_idx = rng.choice(n_rows, size=40, replace=False)
    score_str[bad_idx] = rng.choice(["N/A", "unknown", "—"], size=len(bad_idx))
    
    # Datetime-like strings
    base = pd.Timestamp("2022-01-01")
    signup_offsets = rng.integers(0, 900, size=n_rows)
    signup_date = (base + pd.to_timedelta(signup_offsets, unit="D")).strftime("%Y-%m-%d")
    
    # True latent signal for target
    logit = (
        -2.2
        + 0.018 * (income / 1000)
        + 0.012 * tenure_months
        + 0.04 * usage_gb
        + np.where(segment == "premium", 0.6, 0.0)
        + np.where(segment == "basic", -0.35, 0.0)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)
    
    # Decoy leaky column: copy of target with tiny noise (should be flagged)
    leaky_score = churn + rng.normal(0, 0.02, size=n_rows)
    
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_rows)],
            "age": age,
            "annual_income": income.round(2),
            "tenure_months": tenure_months,
            "monthly_usage_gb": usage_gb.round(2),
            "segment": segment,
            "region": region,
            "credit_score_text": score_str,
            "signup_date": signup_date,
            "leaky_churn_copy": leaky_score.round(4),
            "churn": churn,
        }
    )
    
    # Inject nulls
    null_cols = {
        "annual_income": 0.08,
        "tenure_months": 0.05,
        "credit_score_text": 0.10,
        "signup_date": 0.03,
    }
    for col, frac in null_cols.items():
        idx = rng.choice(n_rows, size=int(n_rows * frac), replace=False)
        df.loc[idx, col] = np.nan
        
    df["all_null_feature"] = np.nan
    df["legacy_flag"] = "legacy"
    
    # Outliers
    outlier_idx = rng.choice(n_rows, size=25, replace=False)
    df.loc[outlier_idx, "monthly_usage_gb"] = rng.uniform(250, 400, size=len(outlier_idx))
    
    # Duplicate rows
    dup_rows = df.sample(20, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

# Initialize Session States
if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = ""
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = False
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "run_logs" not in st.session_state:
    st.session_state.run_logs = []

# Sidebar Controls
st.sidebar.title("🛠️ Configuration")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["gemini", "groq", "ollama"],
    index=0
)

# Helper default model mapping
default_models = {
    "gemini": "gemini-3.1-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1"
}

model_name = st.sidebar.text_input(
    "Model Name",
    value=default_models.get(provider, "")
)

# Retrieve keys from environment if loaded
env_google_key = os.getenv("GOOGLE_API_KEY", "")
env_groq_key = os.getenv("GROQ_API_KEY", "")

# API Key inputs
if provider == "gemini":
    google_key = st.sidebar.text_input(
        "Gemini API Key",
        value=env_google_key,
        type="password",
        help="Google API Key. Leave blank if already set in environment."
    )
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
elif provider == "groq":
    groq_key = st.sidebar.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        help="Groq API Key. Leave blank if already set in environment."
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

max_retries = st.sidebar.slider(
    "Max Leakage Retries",
    min_value=0,
    max_value=3,
    value=2,
    help="How many times the Supervisor agent can loop back to drop suspected leakage features."
)

st.sidebar.markdown("---")
st.sidebar.caption("🌩️ **Nimbus AutoML Platform** · Premium Multi-Agent Pipeline")

# Main Header
st.markdown("""
<div class="logo-container">
    <h1 class="gradient-text">🌩️ Nimbus AutoML</h1>
    <div class="gradient-subtitle">A multi-agent machine learning pipeline that profiles, cleans, selects, trains, and documents your dataset.</div>
</div>
""", unsafe_allow_html=True)

# Quick warning if no API key is present
if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ `GOOGLE_API_KEY` is not set. Please enter it in the sidebar to run the pipeline.")
elif provider == "groq" and not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ `GROQ_API_KEY` is not set. Please enter it in the sidebar to run the pipeline.")

# File Loading & Setup
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-title">📂 Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV must contain a header row and target labels/values."
    )
    
    if uploaded_file is not None:
        if st.session_state.df_name != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df_name = uploaded_file.name
            st.session_state.pipeline_completed = False
            st.session_state.final_state = None

with col2:
    st.markdown('<div class="section-title">💡 Or Try It Now</div>', unsafe_allow_html=True)
    st.write("No dataset handy? Generate a synthetic dataset with known statistical anomalies and decoy leaky features to stress-test Nimbus.")
    
    if st.button("✨ Generate Synthetic Dataset"):
        with st.spinner("Creating synthetic churn dataset..."):
            st.session_state.df = generate_synthetic_churn_data()
            st.session_state.df_name = "synthetic_ground_truth.csv"
            st.session_state.pipeline_completed = False
            st.session_state.final_state = None
        st.success("✅ Loaded synthetic churn dataset (2,020 rows, churn target)!")

# Preview and Configuration
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    
    preview_col, config_col = st.columns([3, 2])
    
    with preview_col:
        st.markdown(f'<div class="section-title">📊 Dataset Preview: {st.session_state.df_name}</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)
        
        # Dimensions card
        rows, cols = df.shape
        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-top: 10px;">
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); padding: 8px 15px; border-radius: 6px;">
                <span style="color: #A0AEC0; font-size: 0.85rem;">Rows:</span> <strong style="color: #FFF;">{rows:,}</strong>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); padding: 8px 15px; border-radius: 6px;">
                <span style="color: #A0AEC0; font-size: 0.85rem;">Columns:</span> <strong style="color: #FFF;">{cols:,}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with config_col:
        st.markdown('<div class="section-title">🎯 Set Pipeline Targets</div>', unsafe_allow_html=True)
        
        # Target column dropdown
        default_target = "churn" if "churn" in df.columns else df.columns[-1]
        target_index = list(df.columns).index(default_target) if default_target in df.columns else 0
        target = st.selectbox("Target Column (y)", df.columns, index=target_index)
        
        st.write("")
        st.write("")
        
        # Check API key configuration before running
        run_disabled = not backend_available
        if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
            run_disabled = True
        elif provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            run_disabled = True
            
        if st.button("🚀 Run Nimbus AutoML Pipeline", use_container_width=True, type="primary", disabled=run_disabled):
            st.session_state.pipeline_completed = False
            st.session_state.final_state = None
            st.session_state.run_logs = []
            
            # Save uploaded dataframe to temporary CSV
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_csv = Path(temp_dir) / st.session_state.df_name
                df.to_csv(temp_csv, index=False)
                
                # Setup Run ID
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"run_{timestamp}"
                
                initial_state = {
                    "dataset_path": str(temp_csv.resolve()),
                    "target_column": target,
                    "eda_report": None,
                    "cleaned_data_path": None,
                    "prep_plan": None,
                    "selected_features": [],
                    "selection_rationale": "",
                    "model_results": [],
                    "best_model_id": None,
                    "model_path": None,
                    "report_path": None,
                    "stage_log": [],
                    "retry_count": {},
                    "token_usage": [],
                    "validation_errors": None,
                }
                
                context: RunConfig = {
                    "run_id": run_id,
                    "llm_provider": provider,
                    "model_name": model_name,
                    "max_retries": max_retries,
                    "token_budget": None,
                }
                
                # Progress and streaming output
                st.markdown("### ⚡ Live Pipeline Output")
                
                with st.status("Initializing agents...", expanded=True) as status_box:
                    try:
                        current_state = dict(initial_state)
                        # Stream state changes from LangGraph
                        for event in graph.stream(current_state, context=context, stream_mode="updates"):
                            for node_name, state_update in event.items():
                                # Merge state values
                                for key, val in state_update.items():
                                    if key in ["stage_log", "token_usage"]:
                                        current_state[key] = (current_state.get(key) or []) + val
                                    else:
                                        current_state[key] = val
                                        
                                # Render status and updates live based on graph execution node
                                if node_name == "profiler":
                                    status_box.update(label="Profiling and analyzing dataset...", state="running")
                                    eda = current_state.get("eda_report")
                                    if eda:
                                        st.write(f"📊 **Profiler Complete**")
                                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• Dtypes: Classification/Regression? **{eda.problem_type}**")
                                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• Shape: `{eda.n_rows}` rows × `{eda.n_cols}` columns")
                                        if eda.concerns:
                                            with st.expander("Show flagged concerns"):
                                                for c in eda.concerns:
                                                    st.write(f"- ⚠️ {c}")
                                                    
                                elif node_name in ["classification_prep", "regression_prep"]:
                                    status_box.update(label="Cleaning and preparing columns...", state="running")
                                    prep_plan = current_state.get("prep_plan")
                                    st.write(f"🧹 **Data Prep Plan Created**")
                                    if prep_plan:
                                        with st.expander("Show preprocessing plan details"):
                                            st.json(prep_plan)
                                            
                                elif node_name in ["classification_selector", "regression_selector"]:
                                    status_box.update(label="Selecting optimal features...", state="running")
                                    selected_feats = current_state.get("selected_features")
                                    rationale = current_state.get("selection_rationale")
                                    st.write(f"✂️ **Feature Selection Complete**")
                                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• Selected `{len(selected_feats)}` features")
                                    if rationale:
                                        with st.expander("Show feature selector rationale"):
                                            st.write(rationale)
                                            
                                elif node_name in ["classification_trainer", "regression_trainer"]:
                                    status_box.update(label="Evaluating model batteries & tuning winners...", state="running")
                                    best_model_id = current_state.get("best_model_id")
                                    validation_errors = current_state.get("validation_errors")
                                    model_results = current_state.get("model_results")
                                    
                                    if validation_errors:
                                        st.warning(f"⚠️ **Validation flagged issues**: {', '.join(validation_errors)}")
                                    else:
                                        st.write(f"🎯 **Model Training Complete**")
                                        if best_model_id:
                                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• Best Model: **{best_model_id}**")
                                        if model_results:
                                            with st.expander("Show full leaderboard details"):
                                                leaderboard = []
                                                for res in model_results:
                                                    leaderboard.append({
                                                        "Model": res.get("model_id"),
                                                        "Metric": res.get("metric_name"),
                                                        "CV Mean": f"{res.get('val_score_mean'):.4f}" if res.get('val_score_mean') is not None else "N/A",
                                                        "CV Std": f"{res.get('val_score_std'):.4f}" if res.get('val_score_std') is not None else "N/A",
                                                        "Tuned": "✅" if res.get("is_tuned") else "❌"
                                                    })
                                                st.table(leaderboard)
                                                
                                elif node_name == "retry_supervisor":
                                    status_box.update(label="Rerouting due to suspected leakage...", state="running")
                                    st.warning(f"🔄 **Retry Supervisor triggered**: data leakage warning detected, initiating retry loop...")
                                    
                                elif node_name == "reporter":
                                    status_box.update(label="Compiling executive markdown report...", state="running")
                                    st.write(f"📝 **Reporter Completed**")
                                    
                        status_box.update(label="AutoML Pipeline Completed Successfully!", state="complete")
                        
                        # Store the final run files into the local runs path so the user can download them
                        # (LangGraph executes them and writes them to local directories inside the workspace,
                        # let's resolve and fetch the generated report and pickle)
                        st.session_state.final_state = current_state
                        st.session_state.pipeline_completed = True
                        st.success("🎉 Pipeline complete! Explore results in the dashboard below.")
                        
                    except Exception as e:
                        status_box.update(label="Execution Failed!", state="error")
                        st.error(f"Pipeline crashed during execution: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# Post-completion dashboard
if st.session_state.pipeline_completed and st.session_state.final_state is not None:
    state = st.session_state.final_state
    
    st.markdown("---")
    st.markdown('<div class="logo-container"><h2 class="gradient-text" style="font-size: 2rem;">🏆 Execution Results</h2></div>', unsafe_allow_html=True)
    
    # Overview metrics cards
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Winning Model</div>
            <div class="metric-value" style="color: #8B7FD6;">{state.get('best_model_id', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        # Find best validation score
        best_score = "N/A"
        results = state.get("model_results", [])
        if results:
            # find candidate with best validation mean score
            # (sort by val_score_mean descending)
            valid_results = [r for r in results if r.get("val_score_mean") is not None]
            if valid_results:
                best_res = max(valid_results, key=lambda x: x.get("val_score_mean"))
                best_score = f"{best_res.get('val_score_mean'):.4f} ({best_res.get('metric_name')})"
                
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Cross-Validation Score</div>
            <div class="metric-value">{best_score}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        # Calculate tokens consumed
        total_tokens = 0
        token_usage = state.get("token_usage", [])
        for entry in token_usage:
            total_tokens += entry.get("input_tokens", 0) + entry.get("output_tokens", 0)
            
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">LLM Tokens Consumed</div>
            <div class="metric-value" style="color: #A78BFA;">{total_tokens:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Result Tabs
    tab_report, tab_features, tab_logs = st.tabs([
        "📄 Executive Report", 
        "✂️ Selected Features & Leaders", 
        "⚙️ Pipeline Logs & Stats"
    ])
    
    with tab_report:
        report_path_str = state.get("report_path")
        if report_path_str and Path(report_path_str).exists():
            report_path = Path(report_path_str)
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                
            # Render report
            st.markdown("### Executive Summary & Analysis")
            st.markdown(report_content)
            
            st.write("")
            st.download_button(
                label="📥 Download Full Markdown Report (.md)",
                data=report_content,
                file_name="nimbus_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.info("Report could not be retrieved from disk. Check if the backend executed successfully.")
            
    with tab_features:
        st.markdown('<div class="section-title">Selected Features</div>', unsafe_allow_html=True)
        features = state.get("selected_features", [])
        if features:
            for f in features:
                st.markdown(f'<span class="feature-badge">{f}</span>', unsafe_allow_html=True)
        else:
            st.write("No features were selected (or all features were dropped).")
            
        st.markdown('<div class="section-title" style="margin-top: 30px;">Model Performance Leaderboard</div>', unsafe_allow_html=True)
        model_results = state.get("model_results", [])
        if model_results:
            tbl_data = []
            for res in model_results:
                tbl_data.append({
                    "Model ID": res.get("model_id"),
                    "Optimization Metric": res.get("metric_name"),
                    "CV Train Mean": f"{res.get('train_score_mean'):.5f}" if res.get('train_score_mean') is not None else "N/A",
                    "CV Validation Mean": f"{res.get('val_score_mean'):.5f}" if res.get('val_score_mean') is not None else "N/A",
                    "CV Val Std": f"{res.get('val_score_std'):.5f}" if res.get('val_score_std') is not None else "N/A",
                    "Hyperparameters Tuned?": "Yes (Optuna)" if res.get("is_tuned") else "No (Default Parameters)"
                })
            st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
            
        # Download Pickle File
        model_path_str = state.get("model_path")
        if model_path_str and Path(model_path_str).exists():
            model_path = Path(model_path_str)
            st.markdown('<div class="section-title" style="margin-top: 30px;">Download Deployable Bundle</div>', unsafe_allow_html=True)
            st.write("Download the self-contained model bundle (`.pkl`). It contains the fitted model and the exact preprocessing rules for clean, train-inference feature parity.")
            
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model Bundle (model.pkl)",
                    data=f,
                    file_name="model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("Pickle bundle could not be found on disk.")
            
    with tab_logs:
        st.markdown('<div class="section-title">Stage Execution Logs</div>', unsafe_allow_html=True)
        stage_log = state.get("stage_log", [])
        if stage_log:
            logs_df = pd.DataFrame(stage_log)
            st.table(logs_df)
        else:
            st.write("No stage execution logs available.")
            
        st.markdown('<div class="section-title" style="margin-top: 30px;">Token Usage Breakdown</div>', unsafe_allow_html=True)
        token_usage = state.get("token_usage", [])
        if token_usage:
            usage_df = pd.DataFrame(token_usage)
            st.table(usage_df)
        else:
            st.write("No LLM token usage tracked.")
