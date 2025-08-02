# views/dossier_view.py
# --------------------------------------------------------------
# FINAL VERSION: Mathematically correct linear normalization + robust Plotly charts
# --------------------------------------------------------------

from __future__ import annotations
import json
import math
import re
from typing import Dict, List
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ai_service import ai_service

# Fixed category order to ensure consistent radar chart alignment
CATEGORIES = ["TAM", "Capital Efficiency", "ROI Potential", "Scale Speed"]


# --------------------------------------------------------------
# ---------- Helper functions & KPI calculus (FINAL, CORRECTED VERSION) ---------------
# --------------------------------------------------------------

def _money_to_float(s: str) -> float:
    """
    Robustly convert currency strings to float in millions.
    Supports: $50B, $3M, $1.2B, $500k, $250K, etc.
    """
    try:
        if not isinstance(s, str):
            return 0.0

        # Clean the string but preserve decimal points
        s = s.replace("$", "").replace(",", "").strip().lower()
        if not s:
            return 0.0

        # Extract suffix (b, m, k) and numeric part
        suffix = ""
        if s.endswith(('b', 'm', 'k')):
            suffix = s[-1]
            numeric_str = s[:-1]
        else:
            numeric_str = s

        # Convert to float (handles decimals like 1.2)
        try:
            num = float(numeric_str)
        except (ValueError, TypeError):
            return 0.0

        # Convert to millions based on suffix
        if suffix == 'b':
            return num * 1000  # Billions to millions
        elif suffix == 'm':
            return num  # Already in millions
        elif suffix == 'k':
            return num / 1000  # Thousands to millions
        else:
            # No suffix - assume raw number (could be millions)
            return num

    except (ValueError, TypeError):
        return 0.0


def _calc_metrics(bp: Dict) -> Dict[str, float]:
    """Calculate the raw metrics for a single startup blueprint."""
    tbl = bp.get("data_table", {})
    invest = _money_to_float(tbl.get("Investment", "0"))
    value = _money_to_float(tbl.get("Pre-Market Value", "0"))
    tam = _money_to_float(tbl.get("Total Accessable Market", "0"))

    # Scale Speed: How quickly can this startup scale (normalized to similar range as ROI)
    # Formula: Market opportunity score + Capital velocity score
    if invest > 0 and tam > 0:
        # Market opportunity: TAM size relative to investment (capped and scaled)
        # Cap at 10 for reasonable range
        market_opportunity = min(tam / invest / 1000, 10)

        # Capital velocity: How fast they convert capital to value (ROI adjusted for market size)
        capital_velocity = (value / invest) * math.log10(tam /
                                                         1000) if tam > 1000 else (value / invest)

        # Combine with reasonable weighting (0-10 range similar to ROI Potential)
        scale_speed = (market_opportunity * 0.3 + capital_velocity * 0.7)
    else:
        scale_speed = 0

    metrics = {
        "TAM": tam,
        # Scale up to avoid near-zero values
        "Capital Efficiency": (10 / invest) if invest > 0 else 0,
        "ROI Potential": value / invest if invest > 0 else 0,
        "Scale Speed": scale_speed
    }
    return metrics


@st.cache_data
def _get_global_log_bounds(blueprints: List[Dict]) -> dict:
    """
    Calculate the global min and max for each metric AFTER log-transformation.
    This creates a fair and consistent scale for all charts.
    """
    all_metrics = [_calc_metrics(bp) for bp in blueprints]
    df = pd.DataFrame(all_metrics)

    # Apply log-transformation with epsilon to avoid log(0) error
    epsilon = 1e-6
    log_df = df.map(lambda x: math.log10(x if x > epsilon else epsilon))

    bounds = {
        'min': log_df.min().to_dict(),
        'max': log_df.max().to_dict()
    }
    return bounds


def normalise_linear(metrics: Dict[str, float], global_bounds: dict) -> Dict[str, float]:
    """
    Correctly normalizes a set of metrics to a 0-100 scale using the
    linear min-max scaling formula on log-transformed data.
    """
    normalised = {}
    # Epsilon to avoid log(0) mathematical error - much more robust
    epsilon = 1e-6
    log_metrics = {k: math.log10(v if v > epsilon else epsilon)
                   for k, v in metrics.items()}

    for cat in CATEGORIES:
        log_v = log_metrics.get(cat, 0)
        log_min = global_bounds['min'].get(cat, 0)
        log_max = global_bounds['max'].get(cat, 1)  # Avoid division by zero

        denominator = log_max - log_min
        if denominator == 0:
            norm = 100.0 if log_v >= log_max else 0.0
        else:
            # The correct linear scaling formula: (value - min) / (max - min)
            norm = (log_v - log_min) / denominator * 100

        # Ensure the value is always within the 0-100 range
        normalised[cat] = round(max(0, min(norm, 100)), 2)
    return normalised


def create_plotly_radar(metrics_data: List[Dict], names: List[str], colors: List[str] = None) -> go.Figure:
    """Creates a robust, correctly styled, and correctly aligned Plotly radar chart."""
    if colors is None:
        colors = ["#667eea", "#f5576c"]

    fig = go.Figure()

    for i, (data, name) in enumerate(zip(metrics_data, names)):
        values = [data.get(cat, 0) for cat in CATEGORIES]

        base_color = colors[i % len(colors)]
        hex_color = base_color.lstrip('#')
        r, g, b = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        fill_color = f"rgba({r}, {g}, {b}, 0.3)"

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the shape
            theta=CATEGORIES + [CATEGORIES[0]],  # Close the shape
            fill='toself',
            name=name,
            line_color=base_color,
            fillcolor=fill_color,
            hoverinfo="r+name",
            hovertemplate='%{r:.0f}<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor="#E5E7EB",
                linecolor="#9CA3AF"
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=CATEGORIES,
                ticktext=[f"<b>{cat}</b>" for cat in CATEGORIES],
                gridcolor="#E5E7EB",
                linecolor="#9CA3AF",
                tickfont=dict(size=14, color="#374151")
            ),
            bgcolor="#FFFFFF"
        ),
        showlegend=(len(names) > 1),
        height=450,
        margin=dict(l=100, r=100, t=120, b=90),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            font=dict(color="#374151", size=10),
            orientation="h",
            yanchor="bottom", y=1.15,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E7EB",
            borderwidth=1
        )
    )
    return fig


# --------------------------------------------------------------
# ---------- Business Overview Helper Functions ----------
# --------------------------------------------------------------

def extract_business_data(blueprint: Dict) -> Dict[str, Dict[str, str]]:
    """
    Extract and organize all business data from startup blueprint.
    Only returns fields that have meaningful content (not 'Not specified').
    """
    canvas = blueprint.get('canvas', {})
    data_table = blueprint.get('data_table', {})

    # Financial & Market Data
    financial_data = {}
    if data_table.get('Total Accessable Market') and data_table.get('Total Accessable Market') != 'Not specified':
        financial_data['Total Addressable Market'] = data_table['Total Accessable Market']
    if data_table.get('Target Market') and data_table.get('Target Market') != 'Not specified':
        financial_data['Target Market'] = data_table['Target Market']
    if data_table.get('Investment') and data_table.get('Investment') != 'Not specified':
        financial_data['Investment Required'] = data_table['Investment']
    if data_table.get('Pre-Market Value') and data_table.get('Pre-Market Value') != 'Not specified':
        financial_data['Pre-Market Valuation'] = data_table['Pre-Market Value']

    # Add industry and customer segments
    if blueprint.get('industry_focus'):
        financial_data['Industry Focus'] = blueprint['industry_focus']
    if canvas.get('Customer Segments') and canvas.get('Customer Segments') != 'Not specified':
        financial_data['Customer Segments'] = canvas['Customer Segments']
    if canvas.get('Key Metrics') and canvas.get('Key Metrics') != 'Not specified':
        financial_data['Key Metrics'] = canvas['Key Metrics']

    # Business Model Core
    business_model = {}
    if canvas.get('Problem') and canvas.get('Problem') != 'Not specified':
        business_model['Problem'] = canvas['Problem']
    if canvas.get('Unique Value Proposition') and canvas.get('Unique Value Proposition') != 'Not specified':
        business_model['Value Proposition'] = canvas['Unique Value Proposition']
    if canvas.get('Proposed Solution') and canvas.get('Proposed Solution') != 'Not specified':
        business_model['Proposed Solution'] = canvas['Proposed Solution']
    if canvas.get('Channels') and canvas.get('Channels') != 'Not specified':
        business_model['Channels'] = canvas['Channels']
    if canvas.get('Revenue Stream') and canvas.get('Revenue Stream') != 'Not specified':
        business_model['Revenue Stream'] = canvas['Revenue Stream']
    if canvas.get('Cost Structure') and canvas.get('Cost Structure') != 'Not specified':
        business_model['Cost Structure'] = canvas['Cost Structure']

    # Strategy & Execution
    strategy_execution = {}
    if canvas.get('Unfair Advantage') and canvas.get('Unfair Advantage') != 'Not specified':
        strategy_execution['Unfair Advantage'] = canvas['Unfair Advantage']

    # Handle nested Risk Management structure
    risks = canvas.get('Key Risks & Mitigation')
    if risks and risks != 'Not specified':
        if isinstance(risks, dict):
            # Format nested risks properly
            risk_text = ""
            for risk_category, mitigation in risks.items():
                risk_text += f"**{risk_category}:** {mitigation}\n\n"
            strategy_execution['Risk Management'] = risk_text.strip()
        else:
            # Handle string format
            strategy_execution['Risk Management'] = risks

    return {
        'financial': financial_data,
        'business_model': business_model,
        'strategy': strategy_execution
    }


def render_business_overview(blueprint: Dict):
    """
    Render comprehensive business overview in 3-column layout.
    """
    business_data = extract_business_data(blueprint)

    # Gleichmäßige Spalten für bessere Balance
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        # Column 1: Market & Financial Overview
        if business_data['financial']:
            with st.container(border=True):
                st.markdown("#### Market & Financial")
                for label, value in business_data['financial'].items():
                    st.markdown(f"**{label}:**  \n{value}")
                    st.markdown("")  # Spacing

    with col2:
        # Column 2: Business Model Core
        if business_data['business_model']:
            with st.container(border=True):
                st.markdown("#### Business Model")
                for label, value in business_data['business_model'].items():
                    st.markdown(f"**{label}:**  \n{value}")
                    st.markdown("")  # Spacing

    with col3:
        # Column 3: Strategy & Execution
        if business_data['strategy']:
            with st.container(border=True):
                st.markdown("#### Strategy & Execution")
                for label, value in business_data['strategy'].items():
                    st.markdown(f"**{label}:**  \n{value}")
                    st.markdown("")  # Spacing


def _initialize_chat_for_startup(blueprint: Dict):
    """Initialize chat state for a specific startup."""
    if (
        "dossier_startup_id" not in st.session_state
        or st.session_state.dossier_startup_id != blueprint["id"]
    ):
        st.session_state.dossier_startup_id = blueprint["id"]
        system_prompt = f"""
🚀 ELITE AI SPARRING PARTNER - SPECIALIZED FOR THIS STARTUP

You are NOT a generic AI assistant. You are a WORLD-CLASS startup expert who has become obsessed with this ONE specific idea. You combine the expertise of:

• 🎯 **Serial Entrepreneur** (built 3 unicorns, 2 exits)
• 💰 **Venture Capitalist** (led Series A-C rounds, portfolio worth $2B+)  
• 🧠 **Strategy Consultant** (McKinsey partner, 15 years experience)
• ⚡ **Execution Coach** (helped 50+ startups scale from 0 to $100M)

YOUR MISSION: Help this founder turn their idea into a billion-dollar reality.

PERSONALITY TRAITS:
- Passionate, almost obsessed with this specific idea
- Direct but supportive - you give tough love when needed
- Solution-oriented - every critique comes with actionable next steps
- Market-savvy - you see opportunities others miss
- Risk-aware but growth-focused

YOUR KNOWLEDGE BASE FOR THIS STARTUP:
{json.dumps(blueprint, indent=2)}

CONVERSATION STYLE:
- Use strategic frameworks (TAM/SAM/SOM, Unit Economics, North Star Metrics)
- Reference real companies and case studies when relevant
- Ask penetrating questions that expose blind spots
- Provide specific, actionable advice with timelines
- Challenge assumptions while building confidence
- Focus on execution, not just theory

REMEMBER: You're not just answering questions - you're co-building this company. Every response should feel like advice from your most successful mentor who genuinely believes in this idea and wants to see it succeed.

Ready to dive deep and build something extraordinary together?
"""
        st.session_state.chat_history = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": f"**Let's start building.** What's the biggest challenge or opportunity you want to tackle first? Whether it's:\n"
                           f"• 🎯 **Go-to-market strategy** and customer acquisition\n"
                           f"• 💰 **Fundraising roadmap** and investor positioning\n"
                           f"• ⚡ **Product development** and MVP validation\n"
                           f"• 📈 **Scale strategy** and operational excellence\n\n"
                           f"I'm ready to dive deep and give you the strategic guidance that moves the needle. What's on your mind?"
            },
        ]


def _display_chat_messages():
    """Display chat messages using native Streamlit chat components."""
    if len(st.session_state.chat_history) > 1:  # More than just system message
        for m in st.session_state.chat_history:
            if m["role"] != "system":
                if m["role"] == "user":
                    st.chat_message("user").write(m['content'])
                else:
                    st.chat_message("assistant").write(m['content'])
    else:
        st.markdown("*💭 No messages yet. Start a conversation below!*")


def _handle_chat_input():
    """Handle user chat input with modern chat_input."""
    # Use native Streamlit chat_input for better UX
    prompt = st.chat_input(
        "Ask your elite sparring partner about strategy, execution, fundraising...",
        key=f"chat_input_{st.session_state.get('dossier_startup_id', 'default')}"
    )

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})

        # Get AI response with error handling
        try:
            with st.spinner("🤖 AI is analyzing your question..."):
                reply = ai_service.ask_sparring_partner(
                    st.session_state.chat_history)

            # Add AI response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply})

        except Exception as e:
            error_msg = f"❌ AI service temporarily unavailable: {str(e)}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg})
            st.error("AI service error. Please try again.")

        # Rerun to update the display
        st.rerun()


# --------------------------------------------------------------
# ------------------- Main Dossier View --------------------
# --------------------------------------------------------------
def dossier_view(blueprint: Dict, all_blueprints: List[Dict]):
    """Renders the full interactive dossier."""

    # ----------------------------------------------------------
    # 🔥  Override: Fixe Kartenhöhe im Dossier komplett aufheben
    #    (wird NACH dem globalen styles.css injiziert und gewinnt
    #     deshalb trotz identischer Spezifität + !important)
    # ----------------------------------------------------------
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] {
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- optionale Klassierung (für spätere Styles) ----------
    st.markdown('<div class="dossier-view">', unsafe_allow_html=True)

    st.markdown(f"### {blueprint['name']}")
    st.write(blueprint["description"])

    # 1 · COMPANY INFORMATION
    st.markdown("#### 📋 Company Details")
    render_business_overview(blueprint)

    # 2. STARTUP-DNA ANALYSIS
    st.markdown("#### 🧬 Startup-DNA")
    global_bounds = _get_global_log_bounds(all_blueprints)
    m_self_raw = _calc_metrics(blueprint)
    m_self_norm = normalise_linear(m_self_raw, global_bounds)

    single_radar = create_plotly_radar([m_self_norm], [blueprint["name"]])
    st.plotly_chart(single_radar, use_container_width=True)

    # 3. HEAD-TO-HEAD ANALYSIS
    st.markdown("#### ⚔️ Head-to-Head-Analyse")
    options = {
        bp["name"]: bp for bp in all_blueprints if bp["id"] != blueprint["id"]}
    compare_name = st.selectbox("Compare with:", ["Select a startup..."] + list(options.keys()),
                                key=f"compare_selector_{blueprint['id']}")

    if compare_name and compare_name != "Select a startup...":
        other_bp = options[compare_name]
        m_other_raw = _calc_metrics(other_bp)
        m_other_norm = normalise_linear(m_other_raw, global_bounds)

        comparison_radar = create_plotly_radar(
            [m_self_norm, m_other_norm],
            [blueprint["name"], other_bp["name"]]
        )
        st.plotly_chart(comparison_radar, use_container_width=True)

        # Compact comparison table
        tbl = pd.DataFrame([
            {
                " ": "TAM ($M)",
                blueprint["name"]: round(_money_to_float(blueprint["data_table"]["Total Accessable Market"]), 1),
                other_bp["name"]: round(_money_to_float(other_bp["data_table"]["Total Accessable Market"]), 1),
            },
            {
                " ": "Investment ($M)",
                blueprint["name"]: round(_money_to_float(blueprint["data_table"]["Investment"]), 1),
                other_bp["name"]: round(_money_to_float(other_bp["data_table"]["Investment"]), 1),
            },
        ]).set_index(" ")
        st.table(tbl)

        # 4. AI SPARRING PARTNER (at the bottom)
    st.markdown("---")
    st.markdown("#### 🧠 AI-Sparring-Partner")

    # Create a dedicated container for the chat interface
    with st.container():
        # Initialize chat for this startup
        _initialize_chat_for_startup(blueprint)

        # Display chat messages in a styled container
        with st.container():
            st.markdown("---")
            _display_chat_messages()
            st.markdown("---")

        # Handle user input in a separate container
        with st.container():
            _handle_chat_input()

    # ---------- wrapper-Ende ----------
    st.markdown("</div>", unsafe_allow_html=True)
