import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.azure_agent import run_azure_forensic_agent
from src.analyzer import analyze_dataset
from src.ai_assistant import generate_forensic_plan
from src.cleaner import clean_dataset
from src.rag_engine import PolicyRAGEngine
from src.guardrails import validate_plan_grounding
from src.s3_uploader import archive_cleaned_dataset

st.set_page_config(page_title="DataSentry", page_icon="🛡️", layout="wide")

# --- CRIMSON & COBALT NEO-NOIR CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Deep obsidian background with dual Electric Blue & Crimson Noir spotlights */
    .stApp {
        background:
            radial-gradient(circle at 12% 15%, rgba(0, 163, 255, 0.16) 0%, transparent 42%),
            radial-gradient(circle at 88% 18%, rgba(255, 23, 68, 0.18) 0%, transparent 45%),
            radial-gradient(circle at 50% 95%, rgba(255, 23, 68, 0.08) 0%, transparent 50%),
            linear-gradient(180deg, #05070D 0%, #030408 100%);
        color: #E2E8F0;
    }

    /* Subtle cyber scanline texture */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            rgba(255, 255, 255, 0.015) 0px,
            rgba(255, 255, 255, 0.015) 1px,
            transparent 1px,
            transparent 3px
        );
        pointer-events: none;
        z-index: 0;
    }

    header[data-testid="stHeader"] {
        background: rgba(5, 7, 13, 0.75);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 23, 68, 0.25);
    }

    /* Dual-Neon Noir Banner (Blue Left -> Crimson Right) */
    .noir-banner {
        background: linear-gradient(125deg, rgba(6, 18, 38, 0.92) 0%, rgba(12, 10, 20, 0.96) 50%, rgba(36, 8, 18, 0.92) 100%);
        border-top: 1px solid rgba(0, 180, 255, 0.45);
        border-bottom: 1px solid rgba(255, 23, 68, 0.45);
        border-left: 5px solid #00B4FF;
        border-right: 5px solid #FF1744;
        box-shadow:
            -12px 0 28px rgba(0, 180, 255, 0.18),
            12px 0 28px rgba(255, 23, 68, 0.20),
            inset 0 0 30px rgba(0, 0, 0, 0.65);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.6rem;
    }
    .noir-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: 2.2px;
        background: linear-gradient(90deg, #00D4FF 0%, #FFFFFF 48%, #FF1744 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 24px rgba(0, 180, 255, 0.25);
        margin: 0;
    }
    .noir-subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 0.45rem;
        letter-spacing: 0.8px;
    }
    .noir-badge-blue {
        color: #00D4FF;
        font-weight: 600;
    }
    .noir-badge-red {
        color: #FF2A5F;
        font-weight: 600;
    }

    /* Glassmorphism Metric Cards with Blue/Red Rim Lighting */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(10, 16, 30, 0.85), rgba(22, 10, 18, 0.85));
        border: 1px solid rgba(0, 180, 255, 0.32);
        border-bottom: 2px solid #FF1744;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow:
            0 6px 24px rgba(0, 0, 0, 0.65),
            inset 0 0 14px rgba(0, 180, 255, 0.06);
        transition: all 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #FF1744;
        border-bottom-color: #00D4FF;
        box-shadow:
            0 0 20px rgba(255, 23, 68, 0.35),
            0 0 10px rgba(0, 212, 255, 0.25);
        transform: translateY(-3px);
    }
    div[data-testid="stMetric"] label {
        font-family: 'JetBrains Mono', monospace !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        font-size: 0.74rem !important;
        letter-spacing: 1.2px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00D4FF 0%, #FF2A5F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* File Uploader Dropzone */
    section[data-testid="stFileUploader"] {
        background: rgba(9, 13, 24, 0.78);
        border: 1px dashed rgba(0, 180, 255, 0.45);
        border-radius: 12px;
        padding: 0.6rem;
        box-shadow: inset 0 0 20px rgba(255, 23, 68, 0.08);
    }

    /* Primary & Download Buttons: Blue-to-Red Neon Split */
    .stButton > button, .stDownloadButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        background: linear-gradient(90deg, rgba(0, 119, 255, 0.22) 0%, rgba(255, 23, 68, 0.24) 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid #00D4FF !important;
        border-right: 2px solid #FF1744 !important;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.8px;
        box-shadow:
            -4px 0 14px rgba(0, 180, 255, 0.25),
            4px 0 14px rgba(255, 23, 68, 0.25);
        transition: all 0.25s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #0099FF 0%, #FF1744 100%) !important;
        color: #FFFFFF !important;
        border-color: #FF1744 !important;
        box-shadow:
            0 0 24px rgba(255, 23, 68, 0.65),
            0 0 16px rgba(0, 180, 255, 0.55);
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #64748B !important;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00D4FF !important;
        border-bottom: 2px solid #FF1744 !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.45);
    }

    /* Subheaders */
    h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
        color: #F1F5F9 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER BANNER ---
st.markdown("""
<div class="noir-banner">
    <div class="noir-title">🛡️ DATASENTRY // FORENSIC DATA ENGINE</div>
    <div class="noir-subtitle">
        <span class="noir-badge-blue">[BLUE-TEAM TELEMETRY]</span> Autonomous Anomaly Detection &amp; Qdrant RAG
        &nbsp;•&nbsp;
        <span class="noir-badge-red">[RED-ALERT GUARDRAILS]</span> Deterministic Verification &amp; Remediation
    </div>
</div>
""", unsafe_allow_html=True)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📂 DROP TARGET CSV DATASET FOR FORENSIC INSPECTION", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    analysis = analyze_dataset(df)

    # Reset AI state when a new file is uploaded
    if st.session_state.get("active_file") != uploaded_file.name:
        st.session_state.active_file = uploaded_file.name
        st.session_state.ai_plan = None
        st.session_state.ai_engine = ""
        st.session_state.ai_guardrail = None

    # --- TOP KPI METRICS ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Integrity Score", f"{analysis['quality_score']}%")
    c2.metric("Total Records", f"{analysis.get('rows', analysis.get('total_rows', len(df))):,}")
    c3.metric("Dimensions", f"{analysis.get('cols', analysis.get('total_columns', df.shape[1]))} Cols")
    c4.metric("Null Cells", analysis["total_missing"])
    c5.metric("Duplicates", analysis["duplicate_rows"])

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS FOR CLEAN WORKFLOW ---
    tab1, tab2, tab3 = st.tabs([
        "📡 DIAGNOSTICS & OUTLIER RADAR",
        "🤖 AI FORENSIC BRIEFING",
        "🧹 REMEDIATION & EXPORT"
    ])

    # === TAB 1: VISUAL DIAGNOSTICS ===
    with tab1:
        col_gauge, col_missing = st.columns([1, 1.4])

        with col_gauge:
            st.subheader("System Integrity Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analysis["quality_score"],
                number={"suffix": "%", "font": {"color": "#00D4FF", "size": 44}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                    "bar": {"color": "#00B4FF"},
                    "bgcolor": "#080C16",
                    "borderwidth": 1,
                    "bordercolor": "#FF1744",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255, 23, 68, 0.38)"},
                        {"range": [50, 80], "color": "rgba(139, 92, 246, 0.25)"},
                        {"range": [80, 100], "color": "rgba(0, 180, 255, 0.28)"}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E2E8F0"},
                height=290,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_missing:
            st.subheader("Missing Value Distribution")
            missing_df = pd.DataFrame(
                list(analysis["missing_by_col"].items()),
                columns=["Column", "Missing Count"]
            )
            fig_missing = px.bar(
                missing_df,
                x="Column",
                y="Missing Count",
                color="Missing Count",
                color_continuous_scale=["#0055FF", "#00D4FF", "#FF1744"]
            )
            fig_missing.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8, 12, 24, 0.65)",
                font={"color": "#E2E8F0"},
                height=290,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_missing, use_container_width=True)

        # Outlier Box Plot Radar
        st.markdown("---")
        st.subheader("🎯 Outlier Radar (Crimson Anomaly Scan)")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            selected_col = st.selectbox("Select numeric feature to inspect anomalies:", num_cols)
            fig_box = px.box(
                df,
                y=selected_col,
                points="all",
                color_discrete_sequence=["#FF1744"]
            )
            fig_box.update_traces(
                marker=dict(color="#00D4FF", line=dict(color="#FF1744", width=1.2)),
                line=dict(color="#FF1744", width=2)
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8, 12, 24, 0.65)",
                font={"color": "#E2E8F0"},
                height=320
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Raw Dataset Snapshot")
        st.dataframe(df.head(12), use_container_width=True)

    # === TAB 2: AI BRIEFING (QDRANT RAG + PYDANTIC PLAN + GUARDRAILS) ===
    with tab2:
        st.subheader("Generative AI Diagnostics & Governance Policy Audit")
        st.caption("Privacy-First Architecture: Only aggregated statistical telemetry is transmitted—never raw rows.")

        if st.button("⚡ RUN AI DIAGNOSTIC SCAN"):
            with st.spinner("Retrieving Qdrant governance rules & running Azure AI Agent Service..."):
                rag = PolicyRAGEngine()
                policies = rag.retrieve_relevant_policies(analysis)

                try:
                    plan, engine_src = run_azure_forensic_agent(analysis, policies)
                except Exception as azure_err:
                    st.warning(f"Azure AI Agent unavailable ({azure_err}); falling back to Gemini...")
                    plan, engine_src = generate_forensic_plan(analysis, policies)

                guardrail = validate_plan_grounding(plan, analysis)

                st.session_state.ai_plan = plan
                st.session_state.ai_engine = engine_src
                st.session_state.ai_guardrail = guardrail

        plan = st.session_state.ai_plan
        if plan:
            st.caption(f"Inference Engine: **{st.session_state.ai_engine}**")

            guardrail = st.session_state.ai_guardrail
            if guardrail["passed"]:
                st.success(f"🛡️ Guardrail Verified — Cited Policies: {', '.join(guardrail['cited_policy_ids'])}")
            else:
                st.error(f"🚨 Guardrail Flagged Issues: {guardrail['issues']}")

            st.markdown("### Executive Forensic Summary")
            st.markdown(plan.executive_summary)

            st.markdown("### Triggered Governance Policies (Qdrant RAG)")
            for policy in plan.cited_policies:
                st.markdown(f"- {policy}")

            st.info(f"**Technical Rationale:** {plan.rationale}")

            markdown_report = (
                f"# DataSentry Forensic Audit Report\n\n"
                f"**Engine:** {st.session_state.ai_engine}\n"
                f"**Guardrail Status:** {'PASSED' if guardrail['passed'] else 'FLAGGED'}\n\n"
                f"## Executive Summary\n{plan.executive_summary}\n\n"
                f"## Triggered Governance Policies\n"
                + "\n".join(f"- {p}" for p in plan.cited_policies)
                + f"\n\n## Recommended Remediation Plan\n"
                f"- Purge Duplicates: `{plan.drop_duplicates}`\n"
                f"- Standardize Text: `{plan.standardize_text}`\n"
                f"- Imputation Strategy: `{plan.imputation_method}`\n\n"
                f"## Technical Rationale\n{plan.rationale}\n"
            )
            st.download_button(
                label="📄 Download Forensic Audit Report (.md)",
                data=markdown_report.encode("utf-8"),
                file_name="datasentry_audit_report.md",
                mime="text/markdown"
            )

    # === TAB 3: HUMAN-IN-THE-LOOP REMEDIATION & EXPORT ===
    with tab3:
        st.subheader("Human-in-the-Loop (HITL) Remediation & Audit Export")

        plan = st.session_state.ai_plan
        imputation_options = [
            "None",
            "Column Median",
            "Linear Interpolation (Between Rows)",
            "Linear Regression Prediction"
        ]

        default_dups = plan.drop_duplicates if plan else True
        default_text = plan.standardize_text if plan else True
        default_imp_idx = (
            imputation_options.index(plan.imputation_method)
            if plan and plan.imputation_method in imputation_options
            else 0
        )

        if plan:
            st.caption("✨ Controls pre-configured from AI Forensic Plan — review and adjust before execution.")

        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            drop_dups = st.checkbox("🗑️ Purge Duplicate Rows", value=default_dups)
            standardize_text = st.checkbox(
                "🔤 Standardize Text Casing & Whitespace (e.g., india → India)",
                value=default_text
            )
            drop_sparse = st.checkbox(
                "✂️ Drop High-Sparsity Columns (>60% Missing, e.g., Price)",
                value=True
            )
        with ctrl_col2:
            missing_method = st.radio(
                "🧠 Missing Numerical Imputation Strategy:",
                imputation_options,
                index=default_imp_idx
            )

        clean_df, cleaning_logs = clean_dataset(
            df,
            drop_dups=drop_dups,
            standardize_text=standardize_text,
            drop_sparse_cols=drop_sparse,
            missing_method=missing_method
        )
        post_analysis = analyze_dataset(clean_df)

        if cleaning_logs:
            delta = round(post_analysis["quality_score"] - analysis["quality_score"], 1)
            st.success(f"Post-Cleaning Integrity Score: **{post_analysis['quality_score']}%** (+{delta}% improvement)")
            for log in cleaning_logs:
                st.write(f"✦ {log}")

        st.write("### Cleaned Output Preview")
        st.dataframe(clean_df.head(15), use_container_width=True)

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = clean_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 EXPORT CLEANED DATASET (.CSV)",
                data=csv_data,
                file_name=f"datasentry_cleaned_{uploaded_file.name}",
                mime="text/csv"
            )
        with exp_col2:
            if st.button("☁️ ARCHIVE TO CLOUD / AUDIT VAULT"):
                archive_info = archive_cleaned_dataset(clean_df, uploaded_file.name)
                st.success(f"Archived to **{archive_info['destination']}**: `{archive_info['location']}`")