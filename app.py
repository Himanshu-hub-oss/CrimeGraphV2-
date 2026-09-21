"""
CrimeGraph AI — Police Intelligence Command Center
AI-Powered Criminal Network Analysis System
Built for Smart India Hackathon (SIH)
"""
import os
import sys
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processor import DataProcessor
from src.graph_engine import CriminalGraphEngine
from src.blockchain_engine import EvidenceBlockchain
from src.ml_models import MLPredictorSuite
from src.advanced_features import (ROLES, load_cases, create_case, update_case_status, load_evidence, register_uploaded_evidence, generate_demo_financial_data, build_alerts, investigation_report)

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="CrimeGraph AI — Criminal Network Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────── CUSTOM CSS ───────────────────────
st.markdown("""
<style>
    /* Dark Cyber Theme */
    .stApp { background-color: #0B0F19; }
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%);
        border-right: 1px solid #1E3A5F;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00D4FF !important;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid #0F3460;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.08);
    }
    .kpi-card h3 { color: #8892B0; font-size: 0.78rem; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card h1 { color: #00D4FF; font-size: 2rem; margin: 0; font-weight: 700; }
    .kpi-card p { color: #64FFDA; font-size: 0.72rem; margin-top: 4px; }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #0F3460, transparent);
        padding: 10px 18px;
        border-left: 4px solid #00D4FF;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }
    .section-header h2 { color: #E6F1FF; font-size: 1.2rem; margin: 0; }
    
    /* Status Badges */
    .badge-critical { background: #FF0055; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-high { background: #FF7700; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium { background: #FFCC00; color: #1A1A2E; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-low { background: #00DD88; color: #1A1A2E; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    
    /* Blockchain block display */
    .block-card {
        background: #111827;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .block-card .hash { color: #00D4FF; word-break: break-all; }
    .block-card .label { color: #8892B0; font-weight: 600; }
    .block-card .value { color: #E6F1FF; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── OFFICER AUTHENTICATION ───────────────────────
DEMO_USERS = {
    "admin": ("crime123", "ADMIN", "HQ-ADMIN-001", "National Crime Intelligence HQ"),
    "officer": ("crime123", "INVESTIGATING_OFFICER", "IO-2048", "PS Central"),
    "cyber": ("crime123", "CYBER_CELL", "CYBER-449", "Cyber Crime Unit"),
    "forensic": ("crime123", "FORENSIC_EXPERT", "FORENSIC-904", "State FSL"),
    "magistrate": ("crime123", "MAGISTRATE_JUDGE", "JUDGE-101", "District Court"),
}

def login_gate():
    if st.session_state.get("authenticated"): return
    st.markdown("#  CrimeGraph AI — Officer Login")
    st.info("Demo accounts are provided for the SIH prototype. Replace these credentials with your organization's SSO/identity provider in production.")
    with st.form("login_form"):
        u=st.text_input("Username")
        pw=st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", use_container_width=True):
            rec=DEMO_USERS.get(u.lower())
            if rec and pw==rec[0]:
                st.session_state.update(authenticated=True, username=u.lower(), role=rec[1], officer_badge=rec[2], station=rec[3])
                st.rerun()
            else: st.error("Invalid demo credentials.")
    st.markdown("**Demo users:** `admin`, `officer`, `cyber`, `forensic`, `magistrate` — password: `crime123` 'THESE ARE ONLY FOR YOUR LOGIN'")
    st.stop()

login_gate()

# ─────────────────────── CACHED LOADERS ───────────────────────
def get_data_signature():
    """Return a cache key that changes whenever an input dataset changes."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    data_files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".csv")
    ]
    return tuple(sorted((path, os.path.getmtime(path), os.path.getsize(path)) for path in data_files))


@st.cache_resource(show_spinner="Loading Criminal Intelligence Data...")
def load_data(data_signature):
    return DataProcessor()

@st.cache_resource(show_spinner="Building Criminal Network Graph...")
def load_graph(_dp, data_signature):
    return CriminalGraphEngine(_dp)

@st.cache_resource(show_spinner="Initializing Blockchain Evidence Ledger...")
def load_blockchain():
    return EvidenceBlockchain()

@st.cache_resource(show_spinner="Loading AI Prediction Models...")
def load_ml():
    return MLPredictorSuite()


def render_kpi(label, value, subtitle=""):
    st.markdown(f"""
    <div class="kpi-card">
        <h3>{label}</h3>
        <h1>{value}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-header"><h2>{title}</h2></div>', unsafe_allow_html=True)


def risk_badge(level):
    badge_class = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low"
    }.get(level, "badge-medium")
    return f'<span class="{badge_class}">{level}</span>'


# ─────────────────────── LOAD RESOURCES ───────────────────────
data_signature = get_data_signature()
dp = load_data(data_signature)
graph_engine = load_graph(dp, data_signature)
blockchain = load_blockchain()
ml_suite = load_ml()
kpis = dp.get_kpis()


# ─────────────────────── SIDEBAR ───────────────────────
with st.sidebar:
    st.markdown("## CrimeGraph AI")
    st.markdown("##### Police Intelligence Command Center")
    st.markdown("---")

    tabs = [
        "Executive Overview",
        "2D Network Explorer",
        "3D Network Topology",
        "Kingpin Analyzer",
        "AI Risk Predictor",
        "AI Link Predictor",
        "NLP Threat Monitor",
        "Crime Hotspot Map",
        "Blockchain Evidence Vault",
        "Advanced Alert Center",
        "Suspect Dossier",
        "Case Management",
        "Evidence Upload",
        "Financial Flow Graph",
        "Live Network Alerts",
        "AI Investigation Report",
        "Advanced Command Dashboard"
    ]
    selected_label = st.radio("Navigation", tabs, label_visibility="collapsed", key="main_navigation")
    # Use a stable page key instead of relying on list indexes. This prevents
    # Streamlit reruns/state from ever rendering the previous feature under a
    # newly selected navigation item.
    page_map = {label: f"page_{i}" for i, label in enumerate(tabs)}
    selected_tab = page_map[selected_label]

    st.markdown("---")
    st.success(f"{st.session_state.get('username','officer')} · {st.session_state.get('role','')}")
    if st.button("Logout", use_container_width=True):
        for k in ["authenticated","username","role","officer_badge","station"]: st.session_state.pop(k, None)
        st.rerun()
    st.markdown("---")
    st.markdown(f"**Suspects:** {kpis['total_suspects']} &nbsp;|&nbsp; **FIRs:** {kpis['total_firs']}")
    st.markdown(f"**Gangs:** {kpis['active_gangs']} &nbsp;|&nbsp; **Edges:** {kpis['total_network_edges']}")
    st.caption("v4.0 Advanced — SIH 2026")


# ═══════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
if selected_tab == "page_0":
    st.markdown("# Executive Intelligence Overview")
    st.markdown("Real-time aggregate intelligence dashboard for senior law enforcement officers.")
    st.markdown("---")

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_kpi("Total Suspects", kpis['total_suspects'], "In Database")
    with c2: render_kpi("Active Gangs", kpis['active_gangs'], "Identified")
    with c3: render_kpi("Critical Risk", kpis['critical_risk'], "Suspects")
    with c4: render_kpi("High Risk", kpis['high_risk'], "Suspects")
    with c5: render_kpi("Total FIRs", kpis['total_firs'], "Registered")
    with c6: render_kpi("Threat Posts", kpis['flagged_threat_posts'], "Flagged Online")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Risk Level Distribution")
        risk_counts = dp.suspects_df['risk_level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"Critical": "#FF0055", "High": "#FF7700", "Medium": "#FFCC00", "Low": "#00DD88"},
            hole=0.45
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6F1FF"), legend=dict(font=dict(color="#8892B0")),
            margin=dict(l=20, r=20, t=30, b=20), height=340
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        section_header("Top Crime Types (FIR)")
        crime_counts = dp.fir_df['crime_type'].value_counts().head(8)
        fig_crime = px.bar(
            x=crime_counts.values, y=crime_counts.index,
            orientation='h',
            color=crime_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_crime.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6F1FF"), xaxis_title="FIR Count", yaxis_title="",
            coloraxis_showscale=False, margin=dict(l=20, r=20, t=30, b=20), height=340,
            xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F")
        )
        st.plotly_chart(fig_crime, use_container_width=True)

    # Gang Summary Table
    section_header("Gang Intelligence Summary")
    gang_df = dp.get_gang_summary()
    st.dataframe(
        gang_df.style.background_gradient(subset=['critical_suspects', 'high_suspects'], cmap='YlOrRd'),
        use_container_width=True, height=350
    )

    # CDR Activity Timeline
    section_header("CDR Call Activity Timeline")
    cdr_ts = dp.cdr_df.copy()
    cdr_ts['date'] = pd.to_datetime(cdr_ts['timestamp']).dt.date
    daily_calls = cdr_ts.groupby('date').size().reset_index(name='call_count')
    fig_cdr = px.area(daily_calls, x='date', y='call_count', color_discrete_sequence=['#00D4FF'])
    fig_cdr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1FF"), xaxis_title="Date", yaxis_title="Intercepted Calls",
        xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F"),
        margin=dict(l=20, r=20, t=30, b=20), height=280
    )
    st.plotly_chart(fig_cdr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: 2D NETWORK EXPLORER
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_1":
    st.markdown("# 2D Interactive Criminal Network Explorer")
    st.markdown("Physics-based Force-Directed graph with hover intelligence on every node and edge.")
    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        gang_list = ["All"] + sorted(dp.suspects_df['gang_affiliation'].unique().tolist())
        filter_gang = st.selectbox("Filter by Gang", gang_list, key="2d_gang")
    with col_f2:
        risk_list = ["All", "Critical", "High", "Medium", "Low"]
        filter_risk = st.selectbox("Filter by Risk", risk_list, key="2d_risk")
    with col_f3:
        sus_list = ["None"] + sorted(dp.suspects_df['suspect_id'].tolist())
        highlight = st.selectbox("Highlight Suspect", sus_list, key="2d_highlight")

    highlight_id = highlight if highlight != "None" else None
    html = graph_engine.generate_pyvis_html(
        filter_gang=filter_gang,
        filter_risk=filter_risk,
        highlight_id=highlight_id,
        height="620px"
    )
    components.html(html, height=650, scrolling=True)

    st.info("**Hover** on any node for detailed suspect intelligence. **Scroll** to zoom. **Drag** to explore clusters.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: 3D NETWORK TOPOLOGY
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_2":
    st.markdown("#3D Criminal Syndicate Topology")
    st.markdown("Immersive 3D visualization with Kingpin Score heatmap coloring and influence spheres.")
    st.markdown("---")

    col_3d_1, col_3d_2 = st.columns([1, 1])
    with col_3d_1:
        gang_3d = st.selectbox("Filter Gang", ["All"] + sorted(dp.suspects_df['gang_affiliation'].unique().tolist()), key="3d_gang")
    with col_3d_2:
        top_n_3d = st.slider("Max Nodes", 30, 150, 80, key="3d_top")

    fig_3d = graph_engine.generate_plotly_3d_graph(filter_gang=gang_3d, top_n=top_n_3d)
    st.plotly_chart(fig_3d, use_container_width=True, height=700)


# ═══════════════════════════════════════════════════════════════
# TAB 4: KINGPIN ANALYZER
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_3":
    st.markdown("# Kingpin Influence Analyzer")
    st.markdown("Identifies the most powerful network influencers using centrality metrics and composite scoring.")
    st.markdown("---")

    top_n_king = st.slider("Number of Top Kingpins", 5, 30, 15, key="king_n")
    kingpins_df = graph_engine.get_top_kingpins(top_n=top_n_king)

    # Kingpin Leaderboard
    section_header(f"Top {top_n_king} Network Kingpins")
    
    display_cols = ['suspect_id', 'name', 'gang_affiliation', 'role', 'primary_crime', 'risk_level',
                    'kingpin_score', 'betweenness_centrality', 'eigenvector_centrality', 'degree', 'city']
    st.dataframe(
        kingpins_df[display_cols].reset_index(drop=True).style.background_gradient(
            subset=['kingpin_score', 'betweenness_centrality'], cmap='hot'
        ),
        use_container_width=True, height=400
    )

    # Bar Chart Comparison
    section_header("Kingpin Score vs Betweenness Centrality")
    fig_king = make_subplots(specs=[[{"secondary_y": True}]])
    fig_king.add_trace(
        go.Bar(x=kingpins_df['name'], y=kingpins_df['kingpin_score'],
               name='Kingpin Score', marker_color='#FF0055', opacity=0.85),
        secondary_y=False
    )
    fig_king.add_trace(
        go.Scatter(x=kingpins_df['name'], y=kingpins_df['betweenness_centrality'],
                   name='Betweenness', mode='lines+markers', marker=dict(color='#00D4FF', size=8)),
        secondary_y=True
    )
    fig_king.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1FF"), xaxis=dict(tickangle=-45),
        yaxis=dict(title="Kingpin Score", gridcolor="#1E3A5F"),
        yaxis2=dict(title="Betweenness Centrality", gridcolor="#1E3A5F"),
        legend=dict(font=dict(color="#8892B0")),
        margin=dict(l=20, r=20, t=30, b=80), height=380
    )
    st.plotly_chart(fig_king, use_container_width=True)

    # Shortest Path Finder
    st.markdown("---")
    section_header("Criminal Path & Intermediary Tracer")
    col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
    all_suspects = sorted(dp.suspects_df['suspect_id'].tolist())
    with col_p1:
        src = st.selectbox("Source Suspect", all_suspects, index=0, key="path_src")
    with col_p2:
        tgt = st.selectbox("Target Suspect", all_suspects, index=min(5, len(all_suspects)-1), key="path_tgt")
    with col_p3:
        st.markdown("<br>", unsafe_allow_html=True)
        find_path = st.button("🔍 Trace Path", use_container_width=True)

    if find_path:
        result = graph_engine.find_shortest_criminal_path(src, tgt)
        if result['found']:
            st.success(f"**Path Found!** {result['degrees_of_separation']} degree(s) of separation, {result['intermediary_count']} intermediaries")
            for hop in result['hops']:
                st.markdown(f"**{hop['from_name']}** ({hop['from_id']}) → **{hop['to_name']}** ({hop['to_id']}) — _Type: {hop['connection_types']}_ | Weight: {hop['weight']}")
        else:
            st.error(result['message'])


# ═══════════════════════════════════════════════════════════════
# TAB 5: AI RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_4":
    st.markdown("# ⚡ AI Suspect Risk Predictor & XAI Explainer")
    st.markdown("Random Forest Risk Classification with Explainable AI feature attribution.")
    st.markdown("---")

    suspect_id_risk = st.selectbox(
        "Select Suspect to Analyze",
        sorted(dp.enriched_df['suspect_id'].tolist()),
        key="risk_suspect"
    )

    if st.button(" Predict Risk Level", use_container_width=True, key="risk_btn"):
        row = dp.enriched_df[dp.enriched_df['suspect_id'] == suspect_id_risk]
        if not row.empty:
            features = row.iloc[0].to_dict()
            result = ml_suite.predict_suspect_risk(features)

            if 'error' not in result:
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    render_kpi("Predicted Risk", result['predicted_risk_level'], "AI Prediction")
                with col_r2:
                    render_kpi("Confidence", f"{result['confidence']*100:.1f}%", "Model Certainty")
                with col_r3:
                    # Suspect's actual label
                    actual = dp.suspects_df[dp.suspects_df['suspect_id'] == suspect_id_risk]['risk_level'].values
                    actual_label = actual[0] if len(actual) > 0 else "N/A"
                    render_kpi("Actual Label", actual_label, "Ground Truth")

                st.markdown("---")

                # Probability Distribution
                section_header("Risk Probability Distribution")
                prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=["Risk Level", "Probability"])
                fig_prob = px.bar(prob_df, x="Risk Level", y="Probability",
                                  color="Risk Level",
                                  color_discrete_map={"Critical": "#FF0055", "High": "#FF7700", "Medium": "#FFCC00", "Low": "#00DD88"})
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E6F1FF"), yaxis=dict(gridcolor="#1E3A5F"),
                    showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=300
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # XAI Feature Attribution
                section_header("Explainable AI — Top Feature Drivers")
                xai_df = pd.DataFrame(result['xai_top_features'])
                fig_xai = px.bar(xai_df, x='impact_score', y='feature', orientation='h',
                                  color='importance_weight', color_continuous_scale='Plasma',
                                  hover_data=['value', 'importance_weight'])
                fig_xai.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E6F1FF"), xaxis_title="Impact Score",
                    yaxis_title="", coloraxis_colorbar_title="Weight",
                    xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F"),
                    margin=dict(l=20, r=20, t=30, b=20), height=300
                )
                st.plotly_chart(fig_xai, use_container_width=True)
            else:
                st.error(result['error'])
        else:
            st.warning("Enriched features not available for this suspect.")


# ═══════════════════════════════════════════════════════════════
# TAB 6: AI LINK PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_5":
    st.markdown("# AI Criminal Link Predictor")
    st.markdown("Graph topology-based prediction of hidden criminal collusion between suspect pairs.")
    st.markdown("---")

    all_sus = sorted(dp.suspects_df['suspect_id'].tolist())
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        sus_a = st.selectbox("Suspect A", all_sus, index=0, key="link_a")
    with col_l2:
        sus_b = st.selectbox("Suspect B", all_sus, index=min(10, len(all_sus)-1), key="link_b")

    if st.button("🔗 Predict Criminal Link", use_container_width=True, key="link_btn"):
        result = ml_suite.predict_link_probability(graph_engine.G, sus_a, sus_b, dp.suspects_df)

        if 'error' not in result:
            c1, c2, c3, c4 = st.columns(4)
            with c1: render_kpi("Link Probability", f"{result['link_probability']*100:.1f}%", "Collusion Score")
            with c2: render_kpi("Common Accomplices", result['common_accomplices_count'], "Shared Contacts")
            with c3: render_kpi("Jaccard Similarity", f"{result['jaccard_similarity']:.3f}", "Network Overlap")
            with c4: render_kpi("Direct Edge?", "Yes" if result['direct_edge_exists'] else "No", "Known Connection")

            st.markdown(f"### Threat Assessment: {result['threat_status']}")
            st.markdown(f"- **Same Gang:** {'Yes' if result['same_gang'] else 'No'}")
            st.markdown(f"- **Same City:** {'Yes' if result['same_city'] else 'No'}")
            st.markdown(f"- **Adamic-Adar Score:** {result['adamic_adar_score']:.4f}")

            # Mutual Accomplices
            mutual = graph_engine.get_mutual_accomplices(sus_a, sus_b)
            if mutual:
                section_header(f"Mutual Accomplices ({len(mutual)})")
                mut_df = pd.DataFrame(mutual)
                st.dataframe(mut_df, use_container_width=True)
        else:
            st.error(result['error'])


# ═══════════════════════════════════════════════════════════════
# TAB 7: NLP THREAT MONITOR
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_6":
    st.markdown("#  NLP Social Media Threat Intelligence")
    st.markdown("AI-powered analysis of intercepted social media communications for threat detection.")
    st.markdown("---")

    # Threat Distribution
    col_n1, col_n2 = st.columns(2)

    with col_n1:
        section_header("Sentiment Flag Distribution")
        sent_counts = dp.posts_df['ai_sentiment_flag'].value_counts()
        fig_sent = px.pie(values=sent_counts.values, names=sent_counts.index,
                          color=sent_counts.index,
                          color_discrete_map={"Neutral": "#00DD88", "Suspicious": "#FFCC00",
                                              "Threatening": "#FF0055", "Coded/Ambiguous": "#AB63FA"}, hole=0.4)
        fig_sent.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6F1FF"),
            legend=dict(font=dict(color="#8892B0")), margin=dict(l=20, r=20, t=30, b=20), height=320
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_n2:
        section_header("Posts by Platform")
        plat_counts = dp.posts_df['platform'].value_counts()
        fig_plat = px.bar(x=plat_counts.index, y=plat_counts.values, color=plat_counts.values,
                          color_continuous_scale='Viridis')
        fig_plat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6F1FF"), xaxis_title="Platform", yaxis_title="Posts",
            coloraxis_showscale=False, xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F"),
            margin=dict(l=20, r=20, t=30, b=20), height=320
        )
        st.plotly_chart(fig_plat, use_container_width=True)

    # Flagged Posts Table
    section_header("Flagged Threat Posts")
    # Ensure engagement_score exists before sorting threat posts.
    # Some datasets do not contain this column, so derive it from common
    # engagement fields when available; otherwise use 0 as a safe fallback.
    if 'engagement_score' not in dp.posts_df.columns:
        engagement_candidates = ['likes', 'comments', 'shares', 'views', 'retweets', 'replies']
        available_engagement = [
            col for col in engagement_candidates if col in dp.posts_df.columns
        ]

        if available_engagement:
            dp.posts_df['engagement_score'] = (
                dp.posts_df[available_engagement]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .sum(axis=1)
            )
        else:
            dp.posts_df['engagement_score'] = 0

    threat_posts = dp.posts_df[
        dp.posts_df['ai_sentiment_flag'].isin(
            ['Threatening', 'Suspicious', 'Coded/Ambiguous']
        )
    ].sort_values(
        by='engagement_score', ascending=False
    ).head(30)
    display_threat_cols = [
        c for c in ['suspect_id', 'platform', 'post_text', 'ai_sentiment_flag',
                    'engagement_score', 'post_date', 'timestamp']
        if c in threat_posts.columns
    ]
    st.dataframe(threat_posts[display_threat_cols],
                 use_container_width=True, height=350)

    # Custom Text Analyzer
    st.markdown("---")
    section_header("Real-time Text Threat Analyzer")
    custom_text = st.text_area("Paste intercepted communication text:", height=100,
                               placeholder="e.g., New shipment arriving from border tonight, lay low near highway naka")

    if st.button(" Analyze Threat Level", use_container_width=True, key="nlp_btn") and custom_text.strip():
        nlp_result = ml_suite.analyze_social_post_threat(custom_text)
        if 'error' not in nlp_result:
            c1, c2, c3 = st.columns(3)
            with c1: render_kpi("Threat Category", nlp_result['threat_category'], "AI Classification")
            with c2: render_kpi("Confidence", f"{nlp_result['confidence']*100:.1f}%", "Model Certainty")
            with c3: render_kpi("Alert Level", nlp_result['alert_level'], "Threat Status")

            if nlp_result['detected_keywords']:
                st.warning(f"**Trigger Keywords Detected:** {', '.join(nlp_result['detected_keywords'])}")

            prob_df = pd.DataFrame(list(nlp_result['probabilities'].items()), columns=["Category", "Probability"])
            fig_nlp = px.bar(prob_df, x="Category", y="Probability", color="Category",
                             color_discrete_map={"Neutral": "#00DD88", "Suspicious": "#FFCC00",
                                                 "Threatening": "#FF0055", "Coded/Ambiguous": "#AB63FA"})
            fig_nlp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F1FF"), showlegend=False,
                yaxis=dict(gridcolor="#1E3A5F"), margin=dict(l=20, r=20, t=30, b=20), height=280
            )
            st.plotly_chart(fig_nlp, use_container_width=True)
        else:
            st.error(nlp_result['error'])


# ═══════════════════════════════════════════════════════════════
# TAB 8: CRIME HOTSPOT MAP
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_7":
    st.markdown("#  Crime Hotspot Intelligence Map")
    st.markdown("Geographic crime density analysis with multi-layered severity scoring.")
    st.markdown("---")

    hotspot_df = dp.get_city_crime_hotspots()

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1: render_kpi("Cities Covered", len(hotspot_df), "Geographic Zones")
    with c2: render_kpi("Top Hotspot", hotspot_df.iloc[0]['city'] if len(hotspot_df) > 0 else "N/A", "Highest Severity")
    with c3: render_kpi("Max Severity", f"{hotspot_df['severity_score'].max():.0f}" if len(hotspot_df) > 0 else "0", "Composite Score")

    st.markdown("---")

    # Scatter Mapbox
    section_header("Geographic Crime Density Scatter Map")
    fig_map = px.scatter_mapbox(
        hotspot_df,
        lat="lat", lon="lon",
        size="severity_score",
        color="severity_score",
        color_continuous_scale="YlOrRd",
        hover_name="city",
        hover_data={"fir_count": True, "suspect_count": True, "cdr_activity_count": True, "severity_score": ":.1f"},
        size_max=40,
        zoom=4,
        mapbox_style="carto-darkmatter",
        height=550
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1FF"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Detailed Table
    section_header("City Crime Severity Breakdown")
    st.dataframe(
        hotspot_df[['city', 'fir_count', 'suspect_count', 'cdr_activity_count', 'severity_score']].style.background_gradient(
            subset=['severity_score'], cmap='YlOrRd'
        ),
        use_container_width=True, height=350
    )


# ═══════════════════════════════════════════════════════════════
# TAB 9: BLOCKCHAIN EVIDENCE VAULT
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_8":
    st.markdown("# 🔒 Blockchain Evidence Chain of Custody")
    st.markdown("Proof-of-Authority SHA-256 blockchain with Merkle root hashing, smart contract access control, and tamper detection.")
    st.markdown("---")

    # Chain Integrity Check
    integrity = blockchain.verify_chain_integrity()
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("Total Blocks", integrity['total_blocks'], "On-Chain")
    with c2: render_kpi("Verified", integrity['verified_blocks'], "Blocks OK")
    with c3: render_kpi("Corrupted", len(integrity['corrupted_blocks']), "Blocks Flagged")
    with c4: render_kpi("Status", "SECURE" if integrity['is_valid'] else "TAMPERED", integrity['status_message'][:30])

    if integrity['is_valid']:
        st.success(integrity['status_message'])
    else:
        st.error(integrity['status_message'])
        for cb in integrity['corrupted_blocks']:
            st.error(f"Block #{cb['block_index']} ({cb['evidence_id']}): {cb['reason']}")

    st.markdown("---")

    # Chain Explorer
    section_header("Evidence Blockchain Explorer")
    for block in blockchain.chain:
        bd = block.to_dict()
        st.markdown(f"""
        <div class="block-card">
            <span class="label">Block #{bd['index']}</span> — <span class="value">{bd['evidence_type']}</span><br>
            <span class="label">Evidence ID:</span> <span class="value">{bd['evidence_id']}</span> &nbsp;|&nbsp;
            <span class="label">Case:</span> <span class="value">{bd['case_id']}</span><br>
            <span class="label">Officer:</span> <span class="value">{bd['officer_badge']}</span> &nbsp;|&nbsp;
            <span class="label">Station:</span> <span class="value">{bd['police_station']}</span><br>
            <span class="label">Role:</span> <span class="value">{bd['authorized_role']}</span><br>
            <span class="label">Evidence Hash:</span> <span class="hash">{bd['evidence_hash'][:40]}...</span><br>
            <span class="label">Merkle Root:</span> <span class="hash">{bd['merkle_root'][:40]}...</span><br>
            <span class="label">Block Hash:</span> <span class="hash">{bd['block_hash'][:40]}...</span><br>
            <span class="label">Previous Hash:</span> <span class="hash">{bd['previous_hash'][:40]}...</span><br>
            <span class="label">Payload:</span> <span class="value">{bd['payload_snippet']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Tampering Simulation
    st.markdown("---")
    section_header("Tampering Attack Simulator (Demo)")
    tamper_idx = st.number_input("Block Index to Tamper", min_value=1,
                                 max_value=max(1, len(blockchain.chain)-1), value=2, step=1, key="tamper_idx")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button(" Simulate Tampering Attack", use_container_width=True, key="tamper_btn"):
            result = blockchain.simulate_tampering_attack(int(tamper_idx))
            if result['success']:
                st.warning(f"Tampering injected: {result['message']}")
                st.session_state['tamper_done'] = True
            else:
                st.error(result['message'])
    with col_t2:
        if st.button(" Verify Chain Integrity", use_container_width=True, key="verify_btn"):
            new_check = blockchain.verify_chain_integrity()
            if new_check['is_valid']:
                st.success(new_check['status_message'])
            else:
                st.error(new_check['status_message'])
                for cb in new_check['corrupted_blocks']:
                    st.error(f"Block #{cb['block_index']} ({cb['evidence_id']}): {cb['reason']}")


# ═══════════════════════════════════════════════════════════════
# TAB 10: ADVANCED ALERT CENTER
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_9":
    st.markdown("# 🚨 Advanced Alert Center")
    st.caption("Prioritized cross-source alerts for investigative review. Alerts are analytical leads, not proof of guilt.")
    fin_alert = generate_demo_financial_data()
    c1, c2 = st.columns([1, 1])
    with c1:
        alert_min = st.slider("Minimum Priority", 0, 100, 50, key="advanced_alert_min")
    with c2:
        if st.button(" Refresh Alert Intelligence", use_container_width=True, key="advanced_alert_refresh"):
            st.session_state["advanced_alert_last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = build_alerts(dp, graph_engine, fin_alert)
    if not alerts.empty:
        alerts = alerts[alerts["priority"] >= alert_min].copy()
    a,b,c,d = st.columns(4)
    a.metric("Active Alerts", len(alerts))
    b.metric("Critical", int((alerts.priority >= 90).sum()) if not alerts.empty else 0)
    c.metric("High", int(((alerts.priority >= 70) & (alerts.priority < 90)).sum()) if not alerts.empty else 0)
    d.metric("Last Refresh", st.session_state.get("advanced_alert_last_refresh", "Now"))
    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True, height=500)
        st.download_button("⬇ Export Alert CSV", alerts.to_csv(index=False), "crimegraph_advanced_alerts.csv", "text/csv", use_container_width=True)
    else:
        st.success("No alerts match the selected priority threshold.")


# ═══════════════════════════════════════════════════════════════
# TAB 11: SUSPECT DOSSIER
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_10":
    st.markdown("#  Suspect Intelligence Dossier Generator")
    st.markdown("Comprehensive legal/intelligence dossier compiled from all data sources.")
    st.markdown("---")

    suspect_id_dossier = st.selectbox(
        "Select Suspect",
        sorted(dp.suspects_df['suspect_id'].tolist()),
        key="dossier_sus"
    )

    # ── Suspect Photo / Identity Reference (shown immediately) ──
    photo_dir = os.path.join(PROJECT_ROOT, "data", "suspect_photos")
    os.makedirs(photo_dir, exist_ok=True)
    existing_photo = None
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
        candidate = os.path.join(photo_dir, f"{suspect_id_dossier}{ext}")
        if os.path.isfile(candidate):
            existing_photo = candidate
            break

    pc_photo, pc_upload = st.columns([1, 2])
    with pc_photo:
        if existing_photo:
            st.image(existing_photo, caption=f"{suspect_id_dossier} — Reference Photo", use_container_width=True)
        else:
            st.info(f" No reference photo found for {suspect_id_dossier}. Upload one on the right.")
    with pc_upload:
        st.markdown("###  Photo Reference")
        st.caption("Authorized reference image only. No facial recognition or automatic identity matching.")
        uploaded = st.file_uploader("Upload suspect reference photo", type=["jpg","jpeg","png","webp"], key=f"photo_{suspect_id_dossier}")
        if uploaded is not None:
            ext = os.path.splitext(uploaded.name)[1].lower() or ".jpg"
            saved = os.path.join(photo_dir, f"{suspect_id_dossier}{ext}")
            data = uploaded.getvalue()
            with open(saved, "wb") as fh:
                fh.write(data)
            sha256 = hashlib.sha256(data).hexdigest()
            st.success(f"Photo saved as {os.path.basename(saved)}")
            st.code(f"SHA-256 Evidence Hash: {sha256}", language="text")
            st.image(data, caption="Uploaded reference image", width=260)

    st.markdown("---")

    if st.button(" Generate Full Dossier", use_container_width=True, key="dossier_btn"):
        dossier = dp.get_suspect_dossier(suspect_id_dossier)
        if dossier:
            # Photo is already rendered above; dossier content starts here.
            pc_photo, pc_upload = st.columns([1, 2])
            # Profile Header
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1: render_kpi("Name", dossier.get('name', 'N/A'), "Full Name")
            with col_d2:
                risk_level = dossier.get('risk_level', 'Unknown')
                render_kpi("Risk Level", risk_level, "Threat Assessment")
            with col_d3: render_kpi("Gang", dossier.get('gang_affiliation', 'N/A'), "Syndicate")
            with col_d4: render_kpi("Role", dossier.get('role_in_network', 'N/A'), "Network Position")

            st.markdown("---")

            # Personal Info
            section_header("Personal Profile")
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1: st.metric("Age", dossier.get('age', 'N/A'))
            with pc2: st.metric("Gender", dossier.get('gender', 'N/A'))
            with pc3: st.metric("City", dossier.get('city', 'N/A'))
            with pc4: st.metric("Prior Cases", dossier.get('prior_cases_count', 0))

            # FIR Records
            section_header(f"Linked FIR Cases ({len(dossier.get('linked_firs', []))})")
            if dossier.get('linked_firs'):
                fir_df = pd.DataFrame(dossier['linked_firs'])
                display_fir_cols = [c for c in ['fir_id', 'crime_type', 'ipc_section', 'city', 'date_filed', 'status'] if c in fir_df.columns]
                st.dataframe(fir_df[display_fir_cols], use_container_width=True, height=200)
            else:
                st.info("No FIR records linked to this suspect.")

            st.markdown(f"**Co-Accused Count:** {dossier.get('co_accused_count', 0)} &nbsp;|&nbsp; **Co-Accused IDs:** {', '.join(dossier.get('co_accused_ids', [])[:10])}")

            # CDR Summary
            section_header(f"CDR Call Intelligence ({dossier.get('cdr_total_calls', 0)} Total Calls)")
            cc1, cc2 = st.columns(2)
            with cc1: st.metric("Total Calls", dossier.get('cdr_total_calls', 0))
            with cc2: st.metric("Unique Contacts", dossier.get('unique_contacts_count', 0))

            if dossier.get('recent_calls'):
                recent_df = pd.DataFrame(dossier['recent_calls'])
                display_cdr_cols = [c for c in ['caller_id', 'callee_id', 'timestamp', 'duration_seconds', 'call_type', 'cell_tower_city'] if c in recent_df.columns]
                st.dataframe(recent_df[display_cdr_cols].head(10), use_container_width=True, height=200)

            # Social Media
            section_header(f"Social Media Footprint ({len(dossier.get('social_posts', []))} Posts)")
            st.metric("Threat Posts", dossier.get('threat_posts_count', 0))
            if dossier.get('social_posts'):
                posts_df = pd.DataFrame(dossier['social_posts'])
                display_post_cols = [c for c in ['platform', 'post_text', 'ai_sentiment_flag', 'engagement_score', 'post_date'] if c in posts_df.columns]
                st.dataframe(posts_df[display_post_cols].head(15), use_container_width=True, height=250)

            # Network position (from graph engine)
            section_header("Network Position Intelligence")
            node_data = graph_engine.G.nodes.get(suspect_id_dossier, {})
            if node_data:
                nc1, nc2, nc3, nc4, nc5 = st.columns(5)
                with nc1: st.metric("Degree", node_data.get('degree', 0))
                with nc2: st.metric("Kingpin Score", f"{node_data.get('kingpin_score', 0):.1f}")
                with nc3: st.metric("Betweenness", f"{node_data.get('betweenness', 0):.5f}")
                with nc4: st.metric("Community ID", node_data.get('community_id', 'N/A'))
                with nc5: st.metric("Clustering", f"{node_data.get('clustering_coeff', 0):.4f}")
        else:
            st.error("Suspect not found in database.")


# ═══════════════════════════════════════════════════════════════
# TAB 12: CASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_11":
    st.markdown("#  Case Management")
    st.caption("Create, track and review investigation cases with role-aware controls.")
    cases=load_cases()
    can_edit = st.session_state.get('role') in ['ADMIN','INVESTIGATING_OFFICER','CYBER_CELL','FORENSIC_EXPERT']
    if can_edit:
        with st.expander("➕ Create New Case", expanded=True):
            c1,c2=st.columns(2)
            cid=c1.text_input("Case ID", placeholder="CASE-2026-001")
            title=c2.text_input("Case Title")
            c3,c4=st.columns(2)
            crime=c3.text_input("Crime Type")
            priority=c4.selectbox("Priority",['Critical','High','Medium','Low'])
            desc=st.text_area("Investigation Description")
            if st.button("Create Case", use_container_width=True):
                if cid and title:
                    ok,msg=create_case(cid,title,crime,priority,st.session_state['officer_badge'],desc)
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()
                else: st.warning("Case ID and title are required.")
    if cases:
        cdf=pd.DataFrame(cases)
        st.dataframe(cdf, use_container_width=True, height=360)
        case_id=st.selectbox("Update Case Status", cdf.case_id.tolist())
        status=st.selectbox("New Status",['Open','Under Investigation','Evidence Review','Court','Closed'])
        if st.button("Update Status"):
            if update_case_status(case_id,status): st.success("Case status updated."); st.rerun()
    else: st.info("No cases yet. Create the first case above.")

# ═══════════════════════════════════════════════════════════════
# TAB 13: EVIDENCE UPLOAD
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_12":
    st.markdown("#  Digital Evidence Upload & Chain of Custody")
    st.caption("Upload authorized evidence, hash it with SHA-256, and register its metadata in the local evidence registry and blockchain ledger.")
    cases=load_cases(); case_ids=[c['case_id'] for c in cases] or ['UNASSIGNED-DEMO']
    allowed=st.session_state.get('role') in ['ADMIN','INVESTIGATING_OFFICER','CYBER_CELL','FORENSIC_EXPERT']
    if not allowed:
        st.warning("Your role is read-only for evidence upload.")
    else:
        c1,c2=st.columns(2)
        case_id=c1.selectbox("Case",case_ids)
        etype=c2.selectbox("Evidence Type",['FIR Record','CDR Call Log','Social Media Threat Post','Financial Flow Ledger','Digital Device Clone','Forensic Report','Other'])
        desc=st.text_area("Evidence Description")
        up=st.file_uploader("Upload evidence file", type=['pdf','csv','txt','json','jpg','jpeg','png','webp','wav','mp3','mp4','zip'])
        if st.button(" Register Evidence", use_container_width=True):
            if up:
                rec=register_uploaded_evidence(up,case_id,st.session_state['officer_badge'],st.session_state['role'],etype,desc)
                # Register a compact metadata payload on the existing blockchain.
                blockchain.register_evidence(rec['evidence_id'],etype,case_id,{'filename':rec['filename'],'sha256':rec['sha256'],'size_bytes':rec['size_bytes']},st.session_state['officer_badge'],st.session_state['station'], 'ROLE_'+st.session_state['role'])
                st.success(f"Evidence registered: {rec['evidence_id']}")
                st.code(rec['sha256'], language='text')
            else: st.warning("Choose a file first.")
    ev=load_evidence()
    if ev:
        st.markdown("### Evidence Registry")
        st.dataframe(pd.DataFrame(ev),use_container_width=True,height=320)

# ═══════════════════════════════════════════════════════════════
# TAB 14: FINANCIAL FLOW GRAPH
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_13":
    st.markdown("#  Financial Transaction Intelligence")
    st.caption("Graph financial flows, identify high-value transfers, and inspect transaction chains. Default data is synthetic demo data.")
    fin=generate_demo_financial_data()
    up=st.file_uploader("Optional: replace demo data with authorized financial CSV", type=['csv'], key='fin_upload')
    if up:
        try:
            fin=pd.read_csv(up)
            required={'transaction_id','source_suspect_id','target_suspect_id','amount_inr'}
            if not required.issubset(fin.columns): st.error(f"CSV must contain: {sorted(required)}")
        except Exception as e: st.error(str(e))
    c1,c2,c3=st.columns(3)
    c1.metric("Transactions",len(fin)); c2.metric("Total Flow",f"₹{fin.amount_inr.sum():,.0f}"); c3.metric("High Value",int((fin.amount_inr>=3500000).sum()))
    st.dataframe(fin.sort_values('amount_inr',ascending=False).head(100),use_container_width=True,height=300)
    # Build a directed network from top flows
    import networkx as nx
    top=fin.nlargest(80,'amount_inr'); G=nx.DiGraph()
    for _,r in top.iterrows(): G.add_edge(r.source_suspect_id,r.target_suspect_id,amount=float(r.amount_inr))
    pos=nx.spring_layout(G,seed=42,k=0.8)
    edge_x=[];edge_y=[]
    for a,b in G.edges(): edge_x += [pos[a][0],pos[b][0],None]; edge_y += [pos[a][1],pos[b][1],None]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=1,color='#35506F'),hoverinfo='none'))
    nodes=list(G.nodes()); fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode='markers+text',text=nodes,textposition='top center',marker=dict(size=12,color=[G.degree(n) for n in nodes],colorscale='Turbo',showscale=True,colorbar=dict(title='Connections')),hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(template='plotly_dark',height=650,showlegend=False,title='Top Financial Flow Network')
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 15: LIVE NETWORK ALERTS
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_14":
    st.markdown("#  Live Network Alert Center")
    st.caption("Refresh-driven live monitoring layer. In production, connect this to a streaming/event bus (Kafka, Azure Event Hubs, etc.).")
    fin=generate_demo_financial_data()
    if st.button(" Refresh Intelligence Feed", use_container_width=True): st.session_state['alert_refresh']=datetime.now().strftime('%H:%M:%S')
    alerts=build_alerts(dp,graph_engine,fin)
    a,b,c=st.columns(3); a.metric('Active Alerts',len(alerts)); b.metric('Critical/High',int((alerts.priority>=80).sum()) if not alerts.empty else 0); c.metric('Last Refresh',st.session_state.get('alert_refresh','Now'))
    if not alerts.empty:
        st.dataframe(alerts,use_container_width=True,height=500)
        st.download_button('⬇ Export Alerts CSV',alerts.to_csv(index=False),file_name='crimegraph_live_alerts.csv',mime='text/csv')
    else: st.success('No active alerts.')

# ═══════════════════════════════════════════════════════════════
# TAB 16: AI INVESTIGATION REPORT
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_15":
    st.markdown("#  AI Investigation Report Generator")
    st.caption("Creates a structured investigative aid from the selected suspect, graph metrics and financial flows. Human review is required.")
    fin=generate_demo_financial_data(); sid=st.selectbox('Suspect',sorted(dp.suspects_df.suspect_id.tolist()),key='report_sid'); cases=load_cases(); cid=st.selectbox('Case', [c['case_id'] for c in cases] or ['UNASSIGNED-DEMO'],key='report_case')
    if st.button(' Generate Report',use_container_width=True):
        rep=investigation_report(dp,graph_engine,sid,fin,cid); st.session_state['last_report']=rep
    rep=st.session_state.get('last_report')
    if rep:
        st.json(rep)
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from io import BytesIO
            buf=BytesIO()
            doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
            styles=getSampleStyleSheet()
            story=[Paragraph('CrimeGraph AI — Investigation Report',styles['Title']),Spacer(1,12)]
            rows=[['Field','Value']]+[[str(k),str(v)] for k,v in rep.items()]
            t=Table(rows,colWidths=[170,350])
            t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP')]))
            story.append(t)
            doc.build(story)
            buf.seek(0)
            st.download_button('⬇ Download PDF Report',buf.getvalue(),file_name=f'{sid}_investigation_report.pdf',mime='application/pdf',use_container_width=True)
        except ModuleNotFoundError:
            st.error('PDF dependency missing: reportlab is not installed in this Python environment.')
            st.code('python -m pip install reportlab', language='powershell')
            st.info('After installation, restart Streamlit and open AI Investigation Report again.')
            st.download_button('⬇ Download Report (JSON)', pd.Series(rep).to_json(indent=2), file_name=f'{sid}_investigation_report.json', mime='application/json', use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 17: ADVANCED DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif selected_tab == "page_16":
    st.markdown("#  Advanced Command Dashboard")
    st.caption("Cross-source command view for cases, evidence, financial intelligence, network risk and alert volume.")
    fin=generate_demo_financial_data(); cases=load_cases(); ev=load_evidence(); alerts=build_alerts(dp,graph_engine,fin)
    cols=st.columns(6)
    vals=[len(cases),len(ev),len(fin),len(alerts),int((alerts.priority>=80).sum()) if not alerts.empty else 0,int((fin.amount_inr>=3500000).sum())]
    labels=['Cases','Evidence','Transactions','Alerts','Priority Alerts','Financial Reviews']
    for col,label,val in zip(cols,labels,vals):
        with col: render_kpi(label,val,'Live local workspace')
    st.markdown('---')
    r1,r2=st.columns(2)
    with r1:
        section_header('Case Status')
        if cases:
            q=pd.DataFrame(cases).status.value_counts().reset_index(); q.columns=['status','count']; st.plotly_chart(px.bar(q,x='status',y='count',template='plotly_dark'),use_container_width=True)
        else: st.info('Create cases to populate this view.')
    with r2:
        section_header('Alert Types')
        if not alerts.empty:
            q=alerts.type.value_counts().reset_index(); q.columns=['type','count']; st.plotly_chart(px.bar(q,x='type',y='count',template='plotly_dark'),use_container_width=True)
        else: st.info('No active alerts.')
    section_header('Top Financial Flows')
    st.dataframe(fin.nlargest(10,'amount_inr'),use_container_width=True)
    st.info(' AI/network indicators are investigative leads, not proof of guilt. Use lawful evidence, authorization, audit logs and human review before operational decisions.')
