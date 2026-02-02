"""
Commercial Scenario & Gap-Analysis Dashboard
Notifica - Management Tool

Allows management to input commercial projects, adjust global pricing scenarios,
and visualize the monthly revenue gap against a target.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNUAL_TARGET = 500_000  # EUR - Adjust this to your actual annual target

# Page configuration
st.set_page_config(
    page_title="Commercieel Scenario Dashboard | Notifica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Target metric - green */
    div[data-testid="column"]:nth-child(1) [data-testid="metric-container"] {
        border-left: 4px solid #16a34a;
    }

    /* Forecast metric - blue */
    div[data-testid="column"]:nth-child(2) [data-testid="metric-container"] {
        border-left: 4px solid #2563eb;
    }

    /* Gap metric - orange/red based on value */
    div[data-testid="column"]:nth-child(3) [data-testid="metric-container"] {
        border-left: 4px solid #f59e0b;
    }

    /* Data editor styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #16136F 0%, #3636A2 100%);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #ffffff 100%);
    }

    [data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
        color: #374151;
    }

    /* Info boxes */
    .info-box {
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .info-box p {
        margin: 0;
        color: #0369a1;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value: float, include_symbol: bool = True) -> str:
    """Format value as Dutch currency (€ 1.000,00)"""
    if pd.isna(value) or value is None:
        return "€ 0,00" if include_symbol else "0,00"

    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {formatted}" if include_symbol else formatted


def format_currency_k(value: float) -> str:
    """Format value as currency in thousands (€ 100K)"""
    if pd.isna(value) or value is None:
        return "€ 0K"
    return f"€ {value/1000:,.0f}K".replace(",", ".")


def get_default_projects() -> pd.DataFrame:
    """Return default project data for demonstration"""
    today = date.today()
    return pd.DataFrame([
        {
            "Project": "Klant A - Dashboard Implementatie",
            "Startdatum": today + relativedelta(months=1),
            "Duur (mnd)": 3,
            "Tarief (€/uur)": 125.0,
            "Uren/maand": 40,
            "Kans (%)": 80
        },
        {
            "Project": "Klant B - Data Migratie",
            "Startdatum": today + relativedelta(months=2),
            "Duur (mnd)": 2,
            "Tarief (€/uur)": 150.0,
            "Uren/maand": 60,
            "Kans (%)": 60
        },
        {
            "Project": "Klant C - Power BI Training",
            "Startdatum": today,
            "Duur (mnd)": 1,
            "Tarief (€/uur)": 175.0,
            "Uren/maand": 16,
            "Kans (%)": 95
        },
        {
            "Project": "Klant D - Jaar Abonnement",
            "Startdatum": today + relativedelta(months=1),
            "Duur (mnd)": 12,
            "Tarief (€/uur)": 100.0,
            "Uren/maand": 20,
            "Kans (%)": 70
        },
    ])


def expand_projects_to_timeseries(
    projects_df: pd.DataFrame,
    scenario_multiplier: float = 1.0,
    forecast_months: int = 12
) -> pd.DataFrame:
    """
    Expand project data into a monthly time-series.

    For each project, allocate revenue to each month within its duration.
    Calculate weighted revenue based on probability and scenario multiplier.

    Args:
        projects_df: DataFrame with project data
        scenario_multiplier: Global price adjustment factor (e.g., 1.1 for +10%)
        forecast_months: Number of months to forecast

    Returns:
        DataFrame with monthly revenue allocation
    """
    if projects_df.empty:
        return pd.DataFrame(columns=["Maand", "Project", "Omzet_Basis", "Omzet_Gewogen"])

    # Generate month range for forecast period
    today = date.today()
    start_month = date(today.year, today.month, 1)
    end_month = start_month + relativedelta(months=forecast_months)

    records = []

    for _, project in projects_df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(project.get("Startdatum")) or pd.isna(project.get("Duur (mnd)")):
            continue

        project_name = project.get("Project", "Onbekend Project")
        start_date = project["Startdatum"]
        duration = int(project.get("Duur (mnd)", 0))
        rate = float(project.get("Tarief (€/uur)", 0))
        hours = float(project.get("Uren/maand", 0))
        probability = float(project.get("Kans (%)", 0)) / 100

        # Convert start_date to date if it's a datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()

        # Calculate monthly revenue
        adjusted_rate = rate * scenario_multiplier
        monthly_revenue_base = rate * hours
        monthly_revenue_weighted = adjusted_rate * hours * probability

        # Allocate to each month within project duration
        project_start = date(start_date.year, start_date.month, 1)

        for month_offset in range(duration):
            revenue_month = project_start + relativedelta(months=month_offset)

            # Only include months within forecast period
            if start_month <= revenue_month < end_month:
                records.append({
                    "Maand": revenue_month,
                    "Project": project_name,
                    "Omzet_Basis": monthly_revenue_base,
                    "Omzet_Gewogen": monthly_revenue_weighted
                })

    if not records:
        return pd.DataFrame(columns=["Maand", "Project", "Omzet_Basis", "Omzet_Gewogen"])

    return pd.DataFrame(records)


def calculate_monthly_summary(
    timeseries_df: pd.DataFrame,
    monthly_target: float,
    forecast_months: int = 12
) -> pd.DataFrame:
    """
    Aggregate monthly revenue and calculate gap vs target.

    Args:
        timeseries_df: Expanded time-series DataFrame
        monthly_target: Monthly revenue target
        forecast_months: Number of months to include

    Returns:
        DataFrame with monthly summaries
    """
    # Generate all months in forecast period
    today = date.today()
    start_month = date(today.year, today.month, 1)
    all_months = [start_month + relativedelta(months=i) for i in range(forecast_months)]

    # Create base DataFrame with all months
    monthly_df = pd.DataFrame({"Maand": all_months})
    monthly_df["Target"] = monthly_target

    if timeseries_df.empty:
        monthly_df["Forecast_Basis"] = 0
        monthly_df["Forecast_Gewogen"] = 0
    else:
        # Aggregate by month
        agg = timeseries_df.groupby("Maand").agg({
            "Omzet_Basis": "sum",
            "Omzet_Gewogen": "sum"
        }).reset_index()
        agg.columns = ["Maand", "Forecast_Basis", "Forecast_Gewogen"]

        # Merge with all months
        monthly_df = monthly_df.merge(agg, on="Maand", how="left").fillna(0)

    # Calculate gap
    monthly_df["Gap"] = monthly_df["Target"] - monthly_df["Forecast_Gewogen"]
    monthly_df["Gap_Pct"] = (monthly_df["Gap"] / monthly_df["Target"] * 100).round(1)

    # Format month label
    monthly_df["Maand_Label"] = monthly_df["Maand"].apply(
        lambda x: x.strftime("%b '%y").capitalize()
    )

    return monthly_df


def create_gap_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Create an interactive Plotly chart showing forecast vs target with gap analysis."""

    fig = go.Figure()

    # Forecast bars (weighted)
    fig.add_trace(go.Bar(
        name="Forecast (gewogen)",
        x=monthly_df["Maand_Label"],
        y=monthly_df["Forecast_Gewogen"],
        marker_color="#2563eb",
        hovertemplate="<b>%{x}</b><br>Forecast: €%{y:,.0f}<extra></extra>"
    ))

    # Gap bars (stacked on top of forecast to reach target)
    gap_values = monthly_df["Gap"].clip(lower=0)  # Only show positive gaps
    fig.add_trace(go.Bar(
        name="Gap tot target",
        x=monthly_df["Maand_Label"],
        y=gap_values,
        marker_color="#fbbf24",
        marker_pattern_shape="/",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Gap: €%{y:,.0f}<extra></extra>"
    ))

    # Target line
    fig.add_trace(go.Scatter(
        name="Target",
        x=monthly_df["Maand_Label"],
        y=monthly_df["Target"],
        mode="lines+markers",
        line=dict(color="#dc2626", width=3, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Target: €%{y:,.0f}<extra></extra>"
    ))

    # Layout
    fig.update_layout(
        barmode="stack",
        title=dict(
            text="<b>Maandelijkse Forecast vs Target</b>",
            font=dict(size=18, color="#1f2937")
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Omzet (€)",
            tickformat="€,.0f",
            gridcolor="#e5e7eb"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=60),
        height=450,
        hovermode="x unified"
    )

    return fig


def create_project_breakdown_chart(timeseries_df: pd.DataFrame) -> go.Figure:
    """Create a stacked bar chart showing revenue by project per month."""

    if timeseries_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Geen projectdata beschikbaar",
            height=300
        )
        return fig

    # Pivot for stacked bars
    pivot = timeseries_df.pivot_table(
        index="Maand",
        columns="Project",
        values="Omzet_Gewogen",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    pivot["Maand_Label"] = pivot["Maand"].apply(
        lambda x: x.strftime("%b '%y").capitalize()
    )

    fig = go.Figure()

    # Color palette
    colors = ["#2563eb", "#7c3aed", "#059669", "#dc2626", "#ea580c", "#0891b2", "#4f46e5", "#be185d"]

    projects = [col for col in pivot.columns if col not in ["Maand", "Maand_Label"]]

    for i, project in enumerate(projects):
        fig.add_trace(go.Bar(
            name=project[:30] + "..." if len(project) > 30 else project,
            x=pivot["Maand_Label"],
            y=pivot[project],
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>{project}</b><br>€%{{y:,.0f}}<extra></extra>"
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="<b>Omzet per Project</b>",
            font=dict(size=16, color="#1f2937")
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="Omzet (€)",
            tickformat="€,.0f",
            gridcolor="#e5e7eb"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=60),
        height=350
    )

    return fig


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "projects_df" not in st.session_state:
    st.session_state.projects_df = get_default_projects()

if "scenario_multiplier" not in st.session_state:
    st.session_state.scenario_multiplier = 1.0


# =============================================================================
# SIDEBAR - SCENARIO CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### 🎚️ Scenario Instellingen")
    st.markdown("---")

    # Price scenario slider
    price_impact = st.slider(
        "**Tarief Scenario Impact**",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        format="%d%%",
        help="Pas alle tarieven aan met een percentage. Dit wijzigt de berekening, niet de ingevoerde data."
    )
    scenario_multiplier = 1 + (price_impact / 100)

    # Show multiplier effect
    if price_impact != 0:
        color = "#16a34a" if price_impact > 0 else "#dc2626"
        st.markdown(f"""
        <div style="background: {color}15; border-left: 3px solid {color}; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
            <span style="color: {color}; font-weight: 600;">
                Tarieven × {scenario_multiplier:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast period
    forecast_months = st.selectbox(
        "**Forecast Periode**",
        options=[6, 12, 18, 24],
        index=1,
        format_func=lambda x: f"{x} maanden"
    )

    # Annual target input
    st.markdown("---")
    st.markdown("### 🎯 Target Configuratie")

    annual_target = st.number_input(
        "**Jaarlijks Target (€)**",
        min_value=0,
        max_value=10_000_000,
        value=ANNUAL_TARGET,
        step=10_000,
        format="%d"
    )

    monthly_target = annual_target / 12
    st.caption(f"Maandelijks target: {format_currency(monthly_target)}")

    st.markdown("---")

    # Reset button
    if st.button("🔄 Reset naar Voorbeelddata", use_container_width=True):
        st.session_state.projects_df = get_default_projects()
        st.rerun()


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #16136F 0%, #3636A2 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;">
    <h1 style="margin: 0; font-size: 1.8rem;">📊 Commercieel Scenario Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
        Beheer projecten, pas scenario's aan en analyseer de omzet gap
    </p>
</div>
""", unsafe_allow_html=True)

# Calculate current data
timeseries_df = expand_projects_to_timeseries(
    st.session_state.projects_df,
    scenario_multiplier=scenario_multiplier,
    forecast_months=forecast_months
)

monthly_summary = calculate_monthly_summary(
    timeseries_df,
    monthly_target=monthly_target,
    forecast_months=forecast_months
)

# Calculate KPIs
total_target = monthly_summary["Target"].sum()
total_forecast = monthly_summary["Forecast_Gewogen"].sum()
total_gap = total_target - total_forecast
gap_percentage = (total_gap / total_target * 100) if total_target > 0 else 0

# =============================================================================
# KPI METRICS ROW
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Target (periode)",
        value=format_currency_k(total_target),
        help=f"Totaal target voor {forecast_months} maanden"
    )

with col2:
    st.metric(
        label="📈 Forecast (gewogen)",
        value=format_currency_k(total_forecast),
        delta=f"{(total_forecast/total_target*100):.0f}% van target" if total_target > 0 else None,
        delta_color="normal"
    )

with col3:
    gap_delta = f"-{gap_percentage:.0f}%" if total_gap > 0 else f"+{abs(gap_percentage):.0f}%"
    st.metric(
        label="⚠️ Gap",
        value=format_currency_k(total_gap),
        delta=gap_delta,
        delta_color="inverse"
    )

with col4:
    # Coverage ratio
    coverage = (total_forecast / total_target * 100) if total_target > 0 else 0
    coverage_color = "#16a34a" if coverage >= 80 else "#f59e0b" if coverage >= 50 else "#dc2626"
    st.metric(
        label="📊 Dekking",
        value=f"{coverage:.0f}%",
        help="Percentage van target gedekt door huidige forecast"
    )

st.markdown("---")

# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

# Gap analysis chart
st.plotly_chart(
    create_gap_chart(monthly_summary),
    use_container_width=True,
    config={"displayModeBar": False}
)

# =============================================================================
# PROJECT INPUT SECTION (THE SANDBOX)
# =============================================================================

st.markdown("---")
st.markdown("""
<div class="section-header">
    📝 Project Pipeline (Sandbox)
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>✏️ <strong>Tip:</strong> Klik in een cel om te bewerken. Gebruik de <strong>+</strong> knop onderaan om nieuwe projecten toe te voegen.
    Wijzigingen worden direct verwerkt in de forecast.</p>
</div>
""", unsafe_allow_html=True)

# Column configuration for the data editor
column_config = {
    "Project": st.column_config.TextColumn(
        "Project",
        help="Naam van het project of de klant",
        width="large",
        required=True
    ),
    "Startdatum": st.column_config.DateColumn(
        "Startdatum",
        help="Verwachte startdatum van het project",
        format="DD-MM-YYYY",
        required=True
    ),
    "Duur (mnd)": st.column_config.NumberColumn(
        "Duur (mnd)",
        help="Projectduur in maanden",
        min_value=1,
        max_value=36,
        step=1,
        required=True
    ),
    "Tarief (€/uur)": st.column_config.NumberColumn(
        "Tarief (€/uur)",
        help="Uurtarief in euro's",
        min_value=0,
        max_value=500,
        step=5,
        format="€ %.0f",
        required=True
    ),
    "Uren/maand": st.column_config.NumberColumn(
        "Uren/maand",
        help="Verwacht aantal uren per maand",
        min_value=0,
        max_value=200,
        step=4,
        required=True
    ),
    "Kans (%)": st.column_config.NumberColumn(
        "Kans (%)",
        help="Waarschijnlijkheid dat dit project doorgaat (0-100%)",
        min_value=0,
        max_value=100,
        step=5,
        format="%d%%",
        required=True
    )
}

# Data editor
edited_df = st.data_editor(
    st.session_state.projects_df,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="project_editor"
)

# Update session state when data changes
if not edited_df.equals(st.session_state.projects_df):
    st.session_state.projects_df = edited_df
    st.rerun()

# =============================================================================
# PROJECT BREAKDOWN CHART
# =============================================================================

st.markdown("---")

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.plotly_chart(
        create_project_breakdown_chart(timeseries_df),
        use_container_width=True,
        config={"displayModeBar": False}
    )

with col_table:
    st.markdown("#### 📋 Maandelijks Overzicht")

    # Display summary table
    display_df = monthly_summary[["Maand_Label", "Forecast_Gewogen", "Target", "Gap"]].copy()
    display_df.columns = ["Maand", "Forecast", "Target", "Gap"]

    # Format currency columns
    for col in ["Forecast", "Target", "Gap"]:
        display_df[col] = display_df[col].apply(lambda x: format_currency(x))

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=350
    )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem 0;">
    <p>Commercieel Scenario Dashboard | Notifica</p>
    <p style="font-size: 0.75rem;">
        Tariefaanpassing: <strong>{:+.0f}%</strong> |
        Forecast periode: <strong>{} maanden</strong> |
        Actieve projecten: <strong>{}</strong>
    </p>
</div>
""".format(price_impact, forecast_months, len(st.session_state.projects_df)), unsafe_allow_html=True)
