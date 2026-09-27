import os
import io
import time
import hashlib
import importlib
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from cleaner import clean_and_profile_dataframe, detect_anomalies, impute_missing_values, encode_categorical_features, generate_data_profile
import nlp_engine as nlp_engine_module
import executor as executor_module
import narrative as narrative_module
import version_manager as version_manager_module


def refresh_changed_query_modules():
    """Reload changed parser/executor files on the next Streamlit rerun."""
    reloaded = False
    narrative_reloaded = False
    version_manager_reloaded = False
    for module in (nlp_engine_module, executor_module, narrative_module, version_manager_module):
        with open(module.__file__, "rb") as source_file:
            current_digest = hashlib.sha256(source_file.read()).hexdigest()
        if getattr(module, "SHEETLINGO_SOURCE_DIGEST", None) != current_digest:
            importlib.reload(module)
            if module in (nlp_engine_module, executor_module):
                reloaded = True
            elif module is narrative_module:
                narrative_reloaded = True
            elif module is version_manager_module:
                version_manager_reloaded = True
    if narrative_reloaded and not reloaded:
        # executor.py imports this summary function directly, so refresh that
        # reference as well when the narrative implementation changes.
        importlib.reload(executor_module)
        reloaded = True
    if reloaded and "benchmark" in sys.modules:
        importlib.reload(sys.modules["benchmark"])
    return reloaded, version_manager_reloaded


def series_values_match(actual, expected):
    """Compare an existing derived column with a newly calculated result."""
    actual = pd.Series(actual).reset_index(drop=True)
    expected = pd.Series(expected).reset_index(drop=True)
    if len(actual) != len(expected):
        return False

    actual_numeric = pd.to_numeric(actual, errors="coerce")
    expected_numeric = pd.to_numeric(expected, errors="coerce")
    actual_is_numeric = actual.isna() | actual_numeric.notna()
    expected_is_numeric = expected.isna() | expected_numeric.notna()
    if actual_is_numeric.all() and expected_is_numeric.all():
        return bool(np.allclose(
            actual_numeric.to_numpy(dtype=float),
            expected_numeric.to_numpy(dtype=float),
            rtol=1e-9,
            atol=0.005,
            equal_nan=True,
        ))

    return actual.astype("string").fillna("<missing>").equals(
        expected.astype("string").fillna("<missing>")
    )


query_modules_reloaded, version_manager_reloaded = refresh_changed_query_modules()
AdvancedNLPEngine = nlp_engine_module.AdvancedNLPEngine
execute_nlp_query = executor_module.execute_nlp_query
generate_narrative_summary = narrative_module.generate_narrative_summary
VersionManager = version_manager_module.VersionManager
from benchmark import run_benchmark_suite
from duckdb_engine import FastDuckDBEngine
from agent_planner import AgentPlanner

# Page Configuration
st.set_page_config(
    page_title="SheetLingo | Ask your data",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Core Engines
@st.cache_resource
def get_nlp_engine(source_revision):
    return AdvancedNLPEngine()

@st.cache_resource
def get_duckdb_engine():
    return FastDuckDBEngine()

@st.cache_resource
def get_agent_planner():
    return AgentPlanner()

if query_modules_reloaded:
    get_nlp_engine.clear()
nlp_engine = get_nlp_engine(nlp_engine_module.SHEETLINGO_SOURCE_DIGEST)
duckdb_engine = get_duckdb_engine()
agent_planner = get_agent_planner()

# Sidebar appearance preference
is_dark_selected = st.sidebar.toggle("Dark theme", value=True)

# Function to dynamically sync .streamlit/config.toml
def sync_streamlit_config(is_dark: bool):
    config_dir = os.path.join(".", ".streamlit")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")
    
    if is_dark:
        toml_content = """[theme]
base = "dark"
primaryColor = "#10B981"
backgroundColor = "#091912"
secondaryBackgroundColor = "#0D2318"
textColor = "#ECFDF5"
font = "sans serif"

[server]
fileWatcherType = "none"
headless = true
"""
    else:
        toml_content = """[theme]
base = "light"
primaryColor = "#059669"
backgroundColor = "#F0FDF4"
secondaryBackgroundColor = "#E6F4EA"
textColor = "#064E3B"
font = "sans serif"

[server]
fileWatcherType = "none"
headless = true
"""
    try:
        with open(config_path, "w") as f:
            f.write(toml_content)
    except Exception:
        pass

sync_streamlit_config(is_dark_selected)

# Custom HTML Dataset Table Renderer
def render_theme_dataframe(df: pd.DataFrame, height: int = 300, is_dark: bool = True):
    if df is None or df.empty:
        st.info("No records to display.")
        return

    if is_dark:
        bg_color = "#0D2318"
        text_color = "#ECFDF5"
        header_bg = "#133324"
        header_text = "#6EE7B7"
        border_color = "rgba(167, 243, 208, 0.25)"
        alt_row = "rgba(16, 185, 129, 0.06)"
    else:
        bg_color = "#FFFFFF"
        text_color = "#064E3B"
        header_bg = "#E6F4EA"
        header_text = "#059669"
        border_color = "#A7F3D0"
        alt_row = "#F0FDF4"

    table_html = f"""
    <div style="max-height: {height}px; overflow-y: auto; overflow-x: auto; border: 1px solid {border_color}; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 16px;">
        <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', system-ui, sans-serif; font-size: 14px; background-color: {bg_color}; color: {text_color}; text-align: left;">
            <thead>
                <tr style="background-color: {header_bg}; color: {header_text}; position: sticky; top: 0; z-index: 2;">
                    <th style="padding: 12px 16px; border-bottom: 2px solid {border_color}; font-weight: 700;">#</th>
                    {''.join([f'<th style="padding: 12px 16px; border-bottom: 2px solid {border_color}; font-weight: 700;">{col}</th>' for col in df.columns])}
                </tr>
            </thead>
            <tbody>
    """
    for idx, row in df.reset_index(drop=True).iterrows():
        row_bg = alt_row if idx % 2 == 1 else bg_color
        table_html += f'<tr style="background-color: {row_bg}; border-bottom: 1px solid {border_color};">'
        table_html += f'<td style="padding: 10px 16px; opacity: 0.7; font-weight: 600;">{idx}</td>'
        for val in row:
            display_val = "" if pd.isna(val) else str(val)
            table_html += f'<td style="padding: 10px 16px;">{display_val}</td>'
        table_html += '</tr>'

    table_html += """
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

# Dataset Download Action Component
def render_download_buttons(df: pd.DataFrame, filename_prefix: str = "sheetlingo_cleaned_dataset"):
    if df is None or df.empty:
        return
        
    c1, c2 = st.columns([1, 1])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    with c1:
        st.download_button(
            label="📥 Download Cleaned Dataset (.csv)",
            data=csv_bytes,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        excel_bytes = excel_buffer.getvalue()
        with c2:
            st.download_button(
                label="📊 Download Cleaned Dataset (.xlsx)",
                data=excel_bytes,
                file_name=f"{filename_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except Exception:
        pass

# Theme CSS
if is_dark_selected:
    theme_css = """
    <style>
        /* DARK PASTEL EMERALD THEME */
        .stApp, header[data-testid="stHeader"], .stAppHeader {
            background: linear-gradient(180deg, #091912 0%, #06130D 100%) !important;
            color: #ECFDF5 !important;
        }
        header[data-testid="stHeader"] * { color: #ECFDF5 !important; }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D2318 0%, #06130D 100%) !important;
            color: #ECFDF5 !important;
            border-right: 1px solid rgba(110, 231, 183, 0.15) !important;
        }
        [data-testid="stSidebar"] * { color: #ECFDF5 !important; }
        
        .main-header {
            background: linear-gradient(135deg, #133324 0%, #0A1E15 100%) !important;
            padding: 24px;
            border-radius: 16px;
            border: 1px solid rgba(110, 231, 183, 0.2) !important;
            box-shadow: 0 10px 30px rgba(6, 78, 59, 0.35) !important;
            margin-bottom: 24px;
        }
        .header-title { color: #ECFDF5 !important; }
        .header-sub { color: #A7F3D0 !important; }
        
        .metric-card {
            background: rgba(16, 185, 129, 0.08) !important;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(167, 243, 208, 0.25) !important;
            text-align: center;
        }
        .metric-value { font-size: 28px; font-weight: 700; color: #6EE7B7 !important; }
        .metric-label { font-size: 12px; color: #A7F3D0 !important; font-weight: 600; text-transform: uppercase; }
        
        .narrative-box {
            background: rgba(6, 78, 59, 0.3) !important;
            border-left: 5px solid #34D399 !important;
            padding: 18px 22px;
            border-radius: 12px;
            margin-bottom: 24px;
            color: #D1FAE5 !important;
        }
        .narrative-box h4 { color: #6EE7B7 !important; }
        .narrative-box ul { color: #D1FAE5 !important; }
        
        .badge-intent {
            background: linear-gradient(135deg, #059669, #10B981) !important;
            color: #ECFDF5 !important;
            padding: 5px 12px;
            border-radius: 8px;
            font-weight: 700;
        }
        .badge-mode {
            background: linear-gradient(90deg, #34D399, #059669) !important;
            color: #064E3B !important;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
        }
        
        div[data-baseweb="input"] input, div[data-baseweb="select"] input {
            background-color: #0F291E !important;
            color: #ECFDF5 !important;
            border: 1px solid rgba(167, 243, 208, 0.3) !important;
        }
        div[data-baseweb="input"], div[data-baseweb="select"] { background-color: #0F291E !important; }
        
        button[data-baseweb="tab"] { color: #A7F3D0 !important; font-weight: 600 !important; }
        button[aria-selected="true"] { color: #6EE7B7 !important; border-bottom-color: #6EE7B7 !important; }
        
        .stButton > button, div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
            color: #ECFDF5 !important;
            border: 1px solid rgba(167, 243, 208, 0.3) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        
        code, pre, div[data-testid="stCodeBlock"], div[data-testid="stCode"], span.stCode {
            background-color: #0F291E !important;
            color: #6EE7B7 !important;
            border: 1px solid rgba(167, 243, 208, 0.25) !important;
            border-radius: 8px !important;
        }
        code *, span.stCode * { color: #6EE7B7 !important; background-color: transparent !important; }
        
        div[data-testid="stExpander"], div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary {
            background-color: #0D2318 !important;
            color: #ECFDF5 !important;
            border: 1px solid rgba(167, 243, 208, 0.25) !important;
            border-radius: 10px !important;
        }
        div[data-testid="stExpander"] summary * { color: #6EE7B7 !important; font-weight: 700 !important; }
        div[data-testid="stAlert"] {
            background-color: rgba(6, 78, 59, 0.4) !important;
            color: #ECFDF5 !important;
            border: 1px solid rgba(167, 243, 208, 0.3) !important;
        }
        .stWidgetLabel, label { color: #A7F3D0 !important; font-weight: 600 !important; }
    </style>
    """
    plotly_template = "plotly_dark"
else:
    theme_css = """
    <style>
        /* LIGHT PASTEL SAGE THEME */
        .stApp, header[data-testid="stHeader"], .stAppHeader {
            background: linear-gradient(180deg, #F0FDF4 0%, #E6F4EA 100%) !important;
            color: #064E3B !important;
        }
        header[data-testid="stHeader"] * { color: #064E3B !important; }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #E6F4EA 0%, #D1FAE5 100%) !important;
            color: #064E3B !important;
            border-right: 1px solid #A7F3D0 !important;
        }
        [data-testid="stSidebar"] * { color: #064E3B !important; }
        
        .main-header {
            background: linear-gradient(135deg, #FFFFFF 0%, #E6F4EA 100%) !important;
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #A7F3D0 !important;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.12) !important;
            margin-bottom: 24px;
        }
        .header-title { color: #064E3B !important; }
        .header-sub { color: #047857 !important; }
        
        .metric-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #A7F3D0 !important;
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.08) !important;
            text-align: center;
        }
        .metric-value { font-size: 28px; font-weight: 700; color: #059669 !important; }
        .metric-label { font-size: 12px; color: #047857 !important; font-weight: 600; text-transform: uppercase; }
        
        .narrative-box {
            background: #FFFFFF !important;
            border-left: 5px solid #10B981 !important;
            padding: 18px 22px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.06) !important;
            color: #065F46 !important;
        }
        .narrative-box h4 { color: #059669 !important; }
        .narrative-box ul { color: #064E3B !important; }
        
        .badge-intent {
            background: linear-gradient(135deg, #10B981, #059669) !important;
            color: #FFFFFF !important;
            padding: 5px 12px;
            border-radius: 8px;
            font-weight: 700;
        }
        .badge-mode {
            background: linear-gradient(90deg, #059669, #047857) !important;
            color: #FFFFFF !important;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
        }
        
        div[data-baseweb="input"] input, div[data-baseweb="select"] input {
            background-color: #FFFFFF !important;
            color: #064E3B !important;
            border: 1px solid #A7F3D0 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="input"], div[data-baseweb="select"] { background-color: #FFFFFF !important; }
        
        button[data-baseweb="tab"] { color: #047857 !important; font-weight: 600 !important; }
        button[aria-selected="true"] { color: #059669 !important; border-bottom-color: #059669 !important; }
        
        .stButton > button, div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #059669 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
            color: #FFFFFF !important;
        }
        
        code, pre, div[data-testid="stCodeBlock"], div[data-testid="stCode"], span.stCode {
            background-color: #E6F4EA !important;
            color: #064E3B !important;
            border: 1px solid #A7F3D0 !important;
            border-radius: 8px !important;
        }
        code *, span.stCode * { color: #064E3B !important; background-color: transparent !important; }
        
        div[data-testid="stExpander"], div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary {
            background-color: #FFFFFF !important;
            color: #064E3B !important;
            border: 1px solid #A7F3D0 !important;
            border-radius: 10px !important;
        }
        div[data-testid="stExpander"] summary * { color: #059669 !important; font-weight: 700 !important; }
        div[data-testid="stAlert"] {
            background-color: #E6F4EA !important;
            color: #064E3B !important;
            border: 1px solid #A7F3D0 !important;
        }
        .stWidgetLabel, label { color: #064E3B !important; font-weight: 600 !important; }
    </style>
    """
    plotly_template = "plotly_white"

st.markdown(theme_css, unsafe_allow_html=True)

# Keep the primary setup focused on choosing data. Less common modes are
# available under Advanced settings for users who need them.
st.sidebar.markdown("## Choose your data")
dataset_choice = st.sidebar.radio(
    "Dataset",
    ["Use Sample Dataset", "Upload Custom File (.csv / .xlsx)"],
    format_func=lambda option: "Use a sample dataset" if option == "Use Sample Dataset" else "Upload a CSV or Excel file",
    label_visibility="collapsed"
)

with st.sidebar.expander("Advanced settings", expanded=False):
    mode = st.radio(
        "Workspace",
        ["💼 Business Analytics Workspace", "🔬 Data Science Preprocessing Workspace"],
        format_func=lambda option: "Business tools" if "Business Analytics" in option else "Data science tools"
    )
    execution_mode = st.radio(
        "How commands are applied",
        ["🔄 Standalone Mode (Full Dataset)", "🔗 Chained Multi-Turn Mode"],
        format_func=lambda option: "Start each request from the original data" if "Standalone Mode" in option else "Continue from the last change",
        help="Choose whether each request starts from the cleaned upload or continues from the last committed change."
    )

raw_df = None
uploaded_filename = "Dataset"

if dataset_choice == "Use Sample Dataset":
    sample_option = st.sidebar.selectbox(
        "Choose a sample",
        ["Employee salaries (demo with data issues)", "Retail sales and inventory"]
    )
    sample_path = os.path.join("sample_data", "employee_salaries.csv" if "Employee salaries" in sample_option else "retail_sales.csv")
    uploaded_filename = "employee_salaries.csv" if "Employee salaries" in sample_option else "retail_sales.csv"
    if os.path.exists(sample_path):
        raw_df = pd.read_csv(sample_path)
else:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        uploaded_filename = uploaded_file.name
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

# Dataset Ingestion & VersionManager Initialization
if raw_df is not None:
    cleaned_base = clean_and_profile_dataframe(raw_df)
    if "version_manager" not in st.session_state or st.session_state.get("current_filename") != uploaded_filename:
        st.session_state.version_manager = VersionManager(cleaned_base)
        st.session_state.current_filename = uploaded_filename
        st.session_state.pop("latest_exec", None)
        st.session_state.pop("nl_query_input", None)
    elif version_manager_reloaded:
        st.session_state.version_manager.__class__ = VersionManager

    vm: VersionManager = st.session_state.version_manager

    with st.sidebar.expander("Change history", expanded=False):
        for idx, commit in enumerate(vm.history):
            is_active = (idx == vm.current_index)
            state_label = "Current" if is_active else "Saved"
            st.caption(f"{state_label} · {commit.commit_id} · {commit.row_count} rows · {commit.col_count} columns")
            st.caption(commit.message)
            if not is_active and st.button(f"Restore {commit.commit_id}", key=f"rollback_{idx}", use_container_width=True):
                vm.rollback_to(idx)
                st.rerun()

        if st.button("Restore original data", use_container_width=True):
            vm.rollback_to(0)
            st.rerun()

# Header
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="header-title" style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">🌿 SheetLingo</h1>
            <p class="header-sub" style="margin: 6px 0 0 0; font-size: 15px;">Ask a question or describe a change to your spreadsheet.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if raw_df is not None:
    vm: VersionManager = st.session_state.version_manager
    active_df = cleaned_base.copy() if "Standalone Mode" in execution_mode else vm.get_active_df()
    profile = generate_data_profile(active_df)
    narrative = generate_narrative_summary(active_df)

    if dataset_choice == "Use Sample Dataset" and "demo with data issues" in sample_option:
        st.caption("Demo data includes intentional missing and unusual values. You can upload your own spreadsheet from the sidebar.")

    # Make the plain-language action the first thing users see.
    st.markdown("### What would you like to do?")
    st.caption("Type a request in everyday language. SheetLingo will show you the result or ask if it needs clarification.")

    with st.form(key="command_form", clear_on_submit=False):
        user_query = st.text_input(
            "Describe what you want to do",
            placeholder="For example: show active employees older than 40",
            key="nl_query_input"
        )
        submit_btn = st.form_submit_button("Run request", type="primary", use_container_width=True)

    st.caption("Examples: “Average Salary by Department” · “Median of all numeric columns” · “Add a row with Name: Alex Lee, Department: Sales, Age: 28, Salary: 70000” · “Add a new column named Avg Salary based on Department”.")

    # Handle Query Execution when form is submitted
    if submit_btn and not user_query.strip():
        st.warning("Enter a request or choose one of the examples above.")
    elif submit_btn and user_query:
        start_t = time.time()
        
        # 1. Multi-Step Decomposition
        steps = agent_planner.decompose_query(user_query)

        # Execute processing steps
        step_df = active_df.copy()
        step_results = []
        exec_res = None

        for step_info in steps:
            sub_query = step_info["query"]
            intent = nlp_engine.predict_intent(sub_query)
            entities = nlp_engine.extract_entities(sub_query, step_df.columns.tolist(), dataframe=step_df)
            existing_column = entities.get("new_column_name") if intent == "ADD_COL" else None
            if existing_column in step_df.columns and not entities.get("operation_clarification"):
                # Re-running a compound request should not fail just because an
                # earlier step already created the same column. Reuse it only
                # when its values match the requested calculation exactly.
                without_existing = step_df.drop(columns=[existing_column])
                recalculated = execute_nlp_query(without_existing, intent, entities)
                if (
                    not recalculated.get("needs_clarification")
                    and existing_column in recalculated["transformed_df"].columns
                    and series_values_match(
                        step_df[existing_column],
                        recalculated["transformed_df"][existing_column],
                    )
                ):
                    step_res = dict(recalculated)
                    step_res["transformed_df"] = step_df.copy()
                    step_res["status_message"] = (
                        f"The '{existing_column}' column is already present with the requested values; kept it as-is."
                    )
                    step_res["code_snippet"] = (
                        f"# Verified existing column {existing_column!r}; no overwrite was needed."
                    )
                else:
                    step_res = execute_nlp_query(step_df, intent, entities)
            else:
                step_res = execute_nlp_query(step_df, intent, entities)
            step_results.append({
                "step": step_info["step"],
                "query": sub_query,
                "intent": intent,
                "entities": entities,
                "res": step_res,
            })
            exec_res = step_res

            # Stop the chain if any operation needs clarification. Do not
            # queue earlier partial results as an Apply change draft.
            if step_res.get("needs_clarification"):
                clarification = step_res.get("clarification_message") or step_res["status_message"]
                exec_res = dict(step_res)
                exec_res["transformed_df"] = active_df.copy()
                exec_res["status_message"] = (
                    f"Step {step_info['step']} needs clarification. No changes were queued. {clarification}"
                )
                exec_res["clarification_message"] = clarification
                break

            step_df = step_res["transformed_df"]

        latency = round((time.time() - start_t) * 1000, 2)

        # Retain each successful operation in the final result so the user can
        # see that the dataset is the cumulative output of the whole request.
        if exec_res is not None and not exec_res.get("needs_clarification"):
            exec_res = dict(exec_res)
            exec_res["transformed_df"] = step_df
            if len(step_results) > 1:
                messages = [item["res"]["status_message"].strip().rstrip(".") for item in step_results]
                if messages:
                    messages[0] = messages[0][:1].upper() + messages[0][1:]
                    messages[1:] = [message[:1].lower() + message[1:] for message in messages[1:]]
                exec_res["status_message"] = (
                    f"Completed all {len(step_results)} steps: " + "; then ".join(messages) + "."
                )
                exec_res["code_snippet"] = "\n\n".join(
                    f"# Step {item['step']}: {item['query']}\n{item['res']['code_snippet']}"
                    for item in step_results
                )
        
        # Save the cumulative result and per-step details for the tab view.
        st.session_state["latest_exec"] = {
            "res": exec_res,
            "entities": step_results[-1]["entities"] if step_results else {},
            "steps": steps,
            "step_results": step_results,
            "latency": latency,
            "query": user_query
        }

        # If operation mutates data, push to uncommitted draft for Safety Preview Diff
        if exec_res["intent"] in ["UPDATE_VALUE", "DROP_NULLS", "DROP_COL", "RENAME_COL", "IMPUTE_KNN", "ADD_COL", "ADD_ROW"] and not exec_res.get("needs_clarification"):
            draft_intent = exec_res["intent"]
            if len(step_results) > 1 and len({item["intent"] for item in step_results}) > 1:
                draft_intent = "MULTI_STEP"
            vm.create_draft(
                modified_df=step_df,
                message=exec_res["status_message"],
                intent=draft_intent,
                code_trace=exec_res["code_snippet"]
            )
            st.rerun()

    # Keep overview numbers secondary to the natural-language task.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{profile['total_rows']}</div><div class="metric-label">Rows</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{profile['total_columns']}</div><div class="metric-label">Columns</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{profile['total_missing_cells']}</div><div class="metric-label">Missing values</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{profile['data_hygiene_score']}%</div><div class="metric-label">Completeness</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Safety Mutation Draft Preview Card (Uncommitted Draft)
    if vm.draft_df is not None:
        st.warning("Review these proposed changes")
        st.info(f"Requested change: {vm.pending_message}")
        latest_draft_request = st.session_state.get("latest_exec", {})
        if (
            latest_draft_request.get("query") == st.session_state.get("nl_query_input", "")
            and len(latest_draft_request.get("steps", [])) > 1
        ):
            st.caption(
                f"All {len(latest_draft_request['steps'])} steps are included in this preview. "
                "Apply change saves them together."
            )
        d_df, d_sum = vm.create_draft(vm.draft_df, vm.pending_message, vm.pending_intent, vm.pending_trace)
        
        diff1, diff2, diff3, diff4, diff5 = st.columns(5)
        with diff1:
            st.metric("Existing cells updated", d_sum["modified_cells"])
        with diff2:
            st.metric("Rows Added", d_sum["rows_added"])
        with diff3:
            st.metric("Rows Removed", d_sum["rows_removed"])
        with diff4:
            st.metric("Rows after change", d_sum["total_rows_after"])
        with diff5:
            active_columns = set(vm.get_active_df().columns)
            st.metric("Columns added", len(set(d_df.columns) - active_columns))

        st.markdown("#### Preview the updated data")
        render_theme_dataframe(d_df, height=240, is_dark=is_dark_selected)
        
        btn_c1, btn_c2 = st.columns([1, 1])
        compound_draft = (
            latest_draft_request.get("query") == st.session_state.get("nl_query_input", "")
            and len(latest_draft_request.get("steps", [])) > 1
        )
        with btn_c1:
            if st.button("Apply all changes" if compound_draft else "Apply change", type="primary", use_container_width=True):
                vm.confirm_draft()
                st.success("All requested changes applied and saved to history." if compound_draft else "Change applied and saved to history.")
                st.rerun()
        with btn_c2:
            if st.button("Discard change", use_container_width=True):
                vm.discard_draft()
                st.info("No changes were applied.")
                st.rerun()
        st.markdown("---")

    # Modular Navigation Tabs
    tab_query, tab_data, tab_duckdb, tab_history, tab_bench = st.tabs([
        "Results",
        "Your data",
        "SQL (Advanced)",
        "Change history",
        "More tools"
    ])

    with tab_query:
        if (
            "latest_exec" in st.session_state
            and st.session_state["latest_exec"].get("query") == st.session_state.get("nl_query_input", "")
        ):
            latest = st.session_state["latest_exec"]
            exec_res = latest["res"]
            entities = latest["entities"]
            steps = latest["steps"]
            step_results = latest.get("step_results", [])
            latency = latest["latency"]

            if len(steps) > 1:
                st.info(f"I split your request into {len(steps)} steps:")
                for s in steps:
                    st.write(f"• **Step {s['step']}:** `{s['query']}`")

            if exec_res.get("needs_clarification"):
                st.warning(exec_res["status_message"])
            else:
                st.success(exec_res["status_message"])

            with st.expander("How SheetLingo interpreted this request", expanded=False):
                if len(step_results) > 1:
                    for item in step_results:
                        st.markdown(f"**Step {item['step']} · {item['intent'].replace('_', ' ').title()}**")
                        st.write(item["res"]["status_message"])
                        st.caption(f"Columns: {item['entities'].get('columns', [])}")
                        if item["res"].get("code_snippet"):
                            st.code(item["res"]["code_snippet"], language="python")
                    st.caption(f"Total processing time: {latency} ms")
                else:
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.write(f"**Operation:** {exec_res['intent'].replace('_', ' ').title()}")
                        st.write(f"**Processing time:** {latency} ms")
                    with tc2:
                        st.write(f"**Columns:** {entities['columns']}")
                        st.write(f"**Values:** {entities['numbers'] or entities['literal_values']}")
                    with tc3:
                        st.write("**Technical details:**")
                        st.code(exec_res['code_snippet'], language='python')

            if not exec_res.get("needs_clarification") and "Business Analytics" in mode:
                if exec_res["result_metric"]:
                    m = exec_res["result_metric"]
                    if m.get("type") == "correlation":
                        st.markdown("#### Correlation between numeric columns")
                        render_theme_dataframe(pd.DataFrame(m["matrix"]), height=220, is_dark=is_dark_selected)
                    elif m.get("type") == "table":
                        st.markdown(f"#### {m['metric'].title()} across numeric columns")
                        render_theme_dataframe(exec_res["transformed_df"], height=120, is_dark=is_dark_selected)
                    else:
                        st.metric(
                            label=f"{m['metric']} of {m['target_column']}",
                            value=f"{m['value']:,}" if isinstance(m['value'], (int, float)) else m['value']
                        )

                if exec_res["chart_spec"]:
                    spec = exec_res["chart_spec"]
                    st.markdown("#### Chart")
                    fig = px.bar(
                        exec_res["transformed_df"],
                        x=spec["x"],
                        y=spec["y"],
                        title=spec["title"],
                        template=plotly_template,
                        color_discrete_sequence=["#34D399", "#10B981", "#059669", "#A7F3D0", "#6EE7B7"]
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Resulting data")
                render_theme_dataframe(exec_res["transformed_df"], height=300, is_dark=is_dark_selected)
                render_download_buttons(exec_res["transformed_df"], filename_prefix=f"sheetlingo_transformed_{uploaded_filename.split('.')[0]}")

            elif not exec_res.get("needs_clarification"):  # Data Science Workspace
                if exec_res["intent"] == "ANOMALY":
                    anomalies = exec_res["transformed_df"]
                    st.markdown(f"#### Potential outliers found: {len(anomalies)}")
                    if not anomalies.empty:
                        render_theme_dataframe(anomalies, height=220, is_dark=is_dark_selected)
                
                st.markdown("#### Resulting data")
                render_theme_dataframe(exec_res["transformed_df"], height=300, is_dark=is_dark_selected)
                render_download_buttons(exec_res["transformed_df"], filename_prefix=f"sheetlingo_transformed_{uploaded_filename.split('.')[0]}")

                st.markdown("---")
                enc_col1, enc_col2 = st.columns([2, 2])
                with enc_col1:
                    enc_method = st.selectbox("Prepare category columns for machine learning", ["onehot", "label"], format_func=lambda option: "One-hot encoding" if option == "onehot" else "Label encoding")
                    if st.button("Prepare data"):
                        encoded_df = encode_categorical_features(exec_res["transformed_df"], method=enc_method)
                        vm.create_draft(
                            modified_df=encoded_df,
                            message=f"Applied {enc_method} categorical encoding.",
                            intent="ENCODE",
                            code_trace=f"encoded_df = encode_categorical_features(df, method='{enc_method}')"
                        )
                        st.rerun()
        else:
            st.info("Run the request above to see its result here.")

    with tab_data:
        # Automated Insights Narrative Card
        bullet_items = "".join([f"<li style='margin-bottom: 6px;'>{item}</li>" for item in narrative['key_insights']])
        st.markdown(f"""
        <div class="narrative-box">
            <h4 style="margin: 0 0 12px 0;">Dataset insights</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 15px; line-height: 1.6;">
                {bullet_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Current data")
        render_theme_dataframe(active_df, height=340, is_dark=is_dark_selected)
        render_download_buttons(active_df, filename_prefix=f"sheetlingo_cleaned_{uploaded_filename.split('.')[0]}")

    with tab_duckdb:
        st.markdown("#### Query your data with SQL")
        st.caption("For users who already know SQL, the current table is available as `dataset`.")
        
        default_sql = "SELECT Department, AVG(Salary) AS avg_salary, COUNT(*) AS emp_count FROM dataset GROUP BY Department ORDER BY avg_salary DESC"
        sql_input = st.text_area("SQL query", value=default_sql, height=100)
        
        if st.button("Run SQL query"):
            start_sql_t = time.time()
            try:
                duckdb_res = duckdb_engine.execute_sql_query(active_df, sql_input)
                sql_latency = round((time.time() - start_sql_t) * 1000, 2)
                st.success(f"Query complete · {len(duckdb_res)} rows returned · {sql_latency} ms")
                render_theme_dataframe(duckdb_res, height=260, is_dark=is_dark_selected)
                render_download_buttons(duckdb_res, filename_prefix="sheetlingo_duckdb_result")
            except Exception as sql_err:
                st.error(f"DuckDB SQL Execution Error: {str(sql_err)}")

    with tab_history:
        st.markdown("#### Saved versions")
        for commit in vm.history:
            st.markdown(f"**{commit.commit_id}** · {commit.timestamp} · {commit.row_count} rows")
            st.caption(commit.message)
            with st.expander("Technical details"):
                st.code(commit.code_trace, language='python')

    with tab_bench:
        st.caption("Additional tools for advanced users.")
        if st.button("Run performance benchmark"):
            with st.spinner("Running benchmark suite..."):
                bench_res = run_benchmark_suite()
                st.json(bench_res)

else:
    st.info("Choose a sample dataset or upload a CSV or Excel file from the sidebar to get started.")
