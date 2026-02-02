"""
Notifica Consultancy Forecast - Cloud Version v3
Interactive Scenario Sandbox with Gap Analysis

Tabs:
1. Overzicht - Scenario Sandbox (st.data_editor + live gap analysis)
2. Projecten - Project management by category
3. Data beheer - Upload/Export
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import io
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

# Import Google Sheets module
from src.gsheets import (
    is_gsheets_available,
    load_projects_from_gsheets,
    save_projects_to_gsheets,
    load_klanten_from_gsheets,
    save_klanten_to_gsheets,
    load_regie_from_gsheets,
    save_regie_to_gsheets,
    load_omzet_2025_from_gsheets,
    save_omzet_2025_to_gsheets
)

# Page config
st.set_page_config(
    page_title="Consultancy Forecast",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* KPI cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Target - green border */
    div[data-testid="column"]:nth-child(1) [data-testid="metric-container"] {
        border-left: 4px solid #16a34a;
    }
    /* Forecast - blue border */
    div[data-testid="column"]:nth-child(2) [data-testid="metric-container"] {
        border-left: 4px solid #2563eb;
    }
    /* Gap - orange border */
    div[data-testid="column"]:nth-child(3) [data-testid="metric-container"] {
        border-left: 4px solid #f59e0b;
    }
    /* Scenario impact - purple border */
    div[data-testid="column"]:nth-child(4) [data-testid="metric-container"] {
        border-left: 4px solid #7c3aed;
    }

    /* Category headers */
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .category-header h3 { margin: 0; color: white; }

    /* Sidebar styling */
    [data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
        color: #374151;
    }

    /* Data editor */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
PARTNERS = ["Tobias", "Arthur", "Dolf", "Mark"]
DAILY_RATE = 1150
TARGET_PER_PARTNER = 11 * DAILY_RATE  # 11 dagen x €1150 = €12.650/maand
MONTHLY_TARGET = TARGET_PER_PARTNER * 4  # 4 partners = €50.600/maand
MONTH_COLS = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun"]

CATEGORIES = [
    ("🏢", "Groei in Corporates"),
    ("🚚", "Migratie"),
    ("🤖", "AI & Innovatie"),
    ("📊", "Data Analyse"),
    ("⏱️", "Regie"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value: float) -> str:
    """Format as Dutch currency"""
    if pd.isna(value) or value is None:
        return "€ 0"
    return f"€ {value:,.0f}".replace(",", ".")


def format_currency_k(value: float) -> str:
    """Format as currency in thousands"""
    if pd.isna(value) or value is None:
        return "€ 0k"
    return f"€ {value/1000:,.0f}k".replace(",", ".")


def load_projects():
    """Laad projecten uit Google Sheets"""
    if is_gsheets_available():
        projects = load_projects_from_gsheets()
        if projects:
            return projects
    return []


def save_projects(projects):
    """Sla projecten op naar Google Sheets"""
    if is_gsheets_available():
        return save_projects_to_gsheets(projects)
    return False


def load_regie():
    """Laad regie data uit Google Sheets"""
    if is_gsheets_available():
        return load_regie_from_gsheets()
    return []


def load_klanten():
    """Laad klanten uit Google Sheets"""
    if is_gsheets_available():
        return load_klanten_from_gsheets()
    return []


def get_all_klanten(projects):
    """Haal alle klanten"""
    klanten_gsheets = load_klanten()
    klant_namen = set([k.get("Naam", "") for k in klanten_gsheets if k.get("Naam")])
    for p in projects:
        klant = p.get("Klant", "")
        if klant and klant not in ["Target", "Diverse"]:
            klant_namen.add(klant)
    return sorted(list(klant_namen))


def get_default_sandbox_projects() -> pd.DataFrame:
    """Default projects for the scenario sandbox"""
    today = date.today()
    return pd.DataFrame([
        {
            "Project": "Klant A - Dashboard Implementatie",
            "Startmaand": date(today.year, today.month, 1) + relativedelta(months=1),
            "Duur": 3,
            "Uurtarief": 125,
            "Uren_mnd": 40,
            "Kans": 80,
            "Status": "Pipeline"
        },
        {
            "Project": "Klant B - Data Migratie",
            "Startmaand": date(today.year, today.month, 1) + relativedelta(months=2),
            "Duur": 2,
            "Uurtarief": 150,
            "Uren_mnd": 60,
            "Kans": 60,
            "Status": "Pipeline"
        },
        {
            "Project": "Klant C - Power BI Training",
            "Startmaand": date(today.year, today.month, 1),
            "Duur": 1,
            "Uurtarief": 175,
            "Uren_mnd": 16,
            "Kans": 100,
            "Status": "Getekend"
        },
        {
            "Project": "Klant D - Consultancy Abonnement",
            "Startmaand": date(today.year, today.month, 1) + relativedelta(months=1),
            "Duur": 6,
            "Uurtarief": 125,
            "Uren_mnd": 20,
            "Kans": 90,
            "Status": "Getekend"
        },
    ])


def expand_projects_to_timeseries(
    projects_df: pd.DataFrame,
    price_multiplier: float = 1.0,
    capacity_pct: float = 100.0,
    forecast_months: int = 6
) -> pd.DataFrame:
    """
    Expand project data into monthly time-series with scenario adjustments.

    Args:
        projects_df: DataFrame with project data
        price_multiplier: Global price adjustment (e.g., 1.1 for +10%)
        capacity_pct: Capacity percentage (e.g., 80 means 80% beschikbaarheid)
        forecast_months: Number of months to forecast

    Returns:
        DataFrame with monthly revenue allocation
    """
    if projects_df.empty:
        return pd.DataFrame(columns=["Maand", "Project", "Status", "Omzet_Basis", "Omzet_Gewogen"])

    today = date.today()
    start_month = date(today.year, today.month, 1)
    end_month = start_month + relativedelta(months=forecast_months)

    records = []

    for _, project in projects_df.iterrows():
        # Skip rows with missing data
        if pd.isna(project.get("Startmaand")) or pd.isna(project.get("Duur")):
            continue

        project_name = project.get("Project", "Onbekend")
        start_date = project["Startmaand"]
        duration = int(project.get("Duur", 0))
        rate = float(project.get("Uurtarief", 0))
        hours = float(project.get("Uren_mnd", 0))
        probability = float(project.get("Kans", 0)) / 100
        status = project.get("Status", "Pipeline")

        # Convert to date if needed
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()

        # Calculate monthly revenue with scenario adjustments
        adjusted_rate = rate * price_multiplier
        capacity_factor = capacity_pct / 100

        # Basis omzet (zonder scenario)
        monthly_revenue_base = rate * hours

        # Gewogen omzet (met scenario en kans)
        monthly_revenue_weighted = adjusted_rate * hours * probability * capacity_factor

        # Allocate to each month
        project_start = date(start_date.year, start_date.month, 1)

        for month_offset in range(duration):
            revenue_month = project_start + relativedelta(months=month_offset)

            if start_month <= revenue_month < end_month:
                records.append({
                    "Maand": revenue_month,
                    "Project": project_name,
                    "Status": status,
                    "Omzet_Basis": monthly_revenue_base,
                    "Omzet_Gewogen": monthly_revenue_weighted
                })

    if not records:
        return pd.DataFrame(columns=["Maand", "Project", "Status", "Omzet_Basis", "Omzet_Gewogen"])

    return pd.DataFrame(records)


def calculate_monthly_summary(
    timeseries_df: pd.DataFrame,
    monthly_target: float,
    forecast_months: int = 6
) -> pd.DataFrame:
    """Aggregate monthly revenue and calculate gap vs target."""
    today = date.today()
    start_month = date(today.year, today.month, 1)
    all_months = [start_month + relativedelta(months=i) for i in range(forecast_months)]

    monthly_df = pd.DataFrame({"Maand": all_months})
    monthly_df["Target"] = monthly_target

    if timeseries_df.empty:
        monthly_df["Forecast_Hard"] = 0
        monthly_df["Forecast_Zacht"] = 0
        monthly_df["Forecast_Totaal"] = 0
    else:
        # Split by status
        hard = timeseries_df[timeseries_df["Status"] == "Getekend"].groupby("Maand")["Omzet_Gewogen"].sum()
        zacht = timeseries_df[timeseries_df["Status"] == "Pipeline"].groupby("Maand")["Omzet_Gewogen"].sum()

        monthly_df = monthly_df.merge(
            hard.reset_index().rename(columns={"Omzet_Gewogen": "Forecast_Hard"}),
            on="Maand", how="left"
        ).fillna(0)

        monthly_df = monthly_df.merge(
            zacht.reset_index().rename(columns={"Omzet_Gewogen": "Forecast_Zacht"}),
            on="Maand", how="left"
        ).fillna(0)

        monthly_df["Forecast_Totaal"] = monthly_df["Forecast_Hard"] + monthly_df["Forecast_Zacht"]

    monthly_df["Gap"] = monthly_df["Target"] - monthly_df["Forecast_Totaal"]
    monthly_df["Maand_Label"] = monthly_df["Maand"].apply(lambda x: x.strftime("%b '%y").capitalize())

    return monthly_df


def create_gap_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart with target line."""
    fig = go.Figure()

    # Hard/Getekend bars
    fig.add_trace(go.Bar(
        name="Getekend",
        x=monthly_df["Maand_Label"],
        y=monthly_df["Forecast_Hard"],
        marker_color="#22c55e",
        hovertemplate="<b>%{x}</b><br>Getekend: €%{y:,.0f}<extra></extra>"
    ))

    # Zacht/Pipeline bars (stacked)
    fig.add_trace(go.Bar(
        name="Pipeline",
        x=monthly_df["Maand_Label"],
        y=monthly_df["Forecast_Zacht"],
        marker_color="#3b82f6",
        hovertemplate="<b>%{x}</b><br>Pipeline: €%{y:,.0f}<extra></extra>"
    ))

    # Gap bars (stacked on top)
    gap_values = monthly_df["Gap"].clip(lower=0)
    fig.add_trace(go.Bar(
        name="Gap",
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

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="<b>Maandelijkse Forecast vs Target</b>",
            font=dict(size=18, color="#1f2937")
        ),
        xaxis=dict(title="", tickangle=0, tickfont=dict(size=12)),
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
            x=0.5
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=40),
        height=400,
        hovermode="x unified"
    )

    return fig


def parse_regie_excel(uploaded_file):
    """Parse regie-uren uit geupload Excel bestand"""
    current_year = date.today().year
    regie_data = []

    try:
        xlsx = pd.ExcelFile(uploaded_file)

        for partner, sheet_name in [("Arthur", "Uren Arthur"), ("Tobias", "Uren Tobias"), ("Mark", "Uren Mark")]:
            if sheet_name not in xlsx.sheet_names:
                continue

            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            if len(df.columns) < 8:
                continue

            datum_col = df.columns[0]
            klant_col = df.columns[4]
            omzet_col = df.columns[7]

            df['_datum'] = pd.to_datetime(df[datum_col], errors='coerce')
            df['_klant'] = df[klant_col].astype(str).str.strip()

            mask = (
                df['_datum'].notna() &
                (df['_datum'].dt.year == current_year) &
                (df['_klant'] != '') &
                (df['_klant'] != 'nan')
            )
            df_current = df[mask].copy()

            if len(df_current) == 0:
                continue

            df_current['_maand'] = df_current['_datum'].dt.month
            df_current['_omzet'] = pd.to_numeric(df_current[omzet_col], errors='coerce').fillna(0)

            monthly = df_current.groupby('_maand')['_omzet'].sum()
            month_map = {1: "Jan", 2: "Feb", 3: "Mrt", 4: "Apr", 5: "Mei", 6: "Jun"}

            regie_entry = {
                "Partner": partner,
                "Klant": "Diverse",
                "Opdracht": f"{partner} regie",
                "Type": "Regie",
                "Categorie": "Regie",
                "Jan": 0, "Feb": 0, "Mrt": 0, "Apr": 0, "Mei": 0, "Jun": 0,
                "Totaal": 0
            }

            for month, total in monthly.items():
                if pd.notna(total) and total > 0 and int(month) in month_map:
                    regie_entry[month_map[int(month)]] = float(total)

            regie_entry["Totaal"] = sum([regie_entry[m] for m in MONTH_COLS])

            if regie_entry["Totaal"] > 0:
                regie_data.append(regie_entry)

        return regie_data
    except Exception as e:
        st.error(f"Excel parsing fout: {e}")
        return []


def parse_pipedrive_excel(uploaded_file):
    """Parse klanten uit Pipedrive Excel export"""
    try:
        df = pd.read_excel(uploaded_file)
        if 'Label' in df.columns:
            klanten_df = df[df['Label'] == 'Klant'].copy()
        else:
            klanten_df = df.copy()

        result = []
        for _, row in klanten_df.iterrows():
            klant = {
                "Naam": str(row.get('Naam', row.get('Name', ''))).strip(),
                "Branche": str(row.get('Branche', row.get('Industry', '-'))),
            }
            if klant["Naam"]:
                result.append(klant)

        return sorted(result, key=lambda x: x["Naam"])
    except Exception as e:
        st.error(f"Pipedrive parsing fout: {e}")
        return []


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "sandbox_projects" not in st.session_state:
    st.session_state.sandbox_projects = get_default_sandbox_projects()

if "price_scenario" not in st.session_state:
    st.session_state.price_scenario = 0

if "capacity_scenario" not in st.session_state:
    st.session_state.capacity_scenario = 100


# =============================================================================
# SIDEBAR - SCENARIO CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### 🎚️ Scenario Instellingen")
    st.markdown("---")

    # Price scenario slider
    price_impact = st.slider(
        "**Prijs Scenario**",
        min_value=-20,
        max_value=20,
        value=st.session_state.price_scenario,
        step=5,
        format="%d%%",
        help="Pas alle tarieven aan met een percentage. Dit wijzigt alleen de berekening, niet de brondata."
    )
    st.session_state.price_scenario = price_impact
    price_multiplier = 1 + (price_impact / 100)

    if price_impact != 0:
        color = "#16a34a" if price_impact > 0 else "#dc2626"
        st.markdown(f"""
        <div style="background: {color}15; border-left: 3px solid {color}; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0;">
            <span style="color: {color}; font-weight: 600; font-size: 0.9rem;">
                Tarieven × {price_multiplier:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Capacity scenario slider
    capacity_pct = st.slider(
        "**Beschikbaarheid**",
        min_value=50,
        max_value=100,
        value=st.session_state.capacity_scenario,
        step=10,
        format="%d%%",
        help="Capaciteitspercentage (bijv. 80% = alleen 80% van de uren wordt gerealiseerd)"
    )
    st.session_state.capacity_scenario = capacity_pct

    if capacity_pct < 100:
        st.markdown(f"""
        <div style="background: #f59e0b15; border-left: 3px solid #f59e0b; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0;">
            <span style="color: #b45309; font-weight: 600; font-size: 0.9rem;">
                Capaciteit: {capacity_pct}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Target info
    st.markdown("### 🎯 Targets")
    st.markdown(f"""
    - Dagtarief: €{DAILY_RATE:,}
    - 50% = 11 dagen/mnd
    - Per partner: €{TARGET_PER_PARTNER:,}/mnd
    - Team (4p): €{MONTHLY_TARGET:,}/mnd
    """)

    st.markdown("---")

    # Connection status
    if is_gsheets_available():
        st.success("✅ Google Sheets verbonden")
    else:
        st.warning("⚠️ Lokale modus")

    st.markdown("---")

    # Reset button
    if st.button("🔄 Reset scenario's", use_container_width=True):
        st.session_state.price_scenario = 0
        st.session_state.capacity_scenario = 100
        st.rerun()

    if st.button("🗑️ Reset sandbox data", use_container_width=True):
        st.session_state.sandbox_projects = get_default_sandbox_projects()
        st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.title("📊 Consultancy Forecast")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Overzicht (Sandbox)", "📦 Projecten", "⚙️ Data beheer"])


# =============================================================================
# TAB 1: OVERZICHT - SCENARIO SANDBOX
# =============================================================================
with tab1:
    st.subheader("Scenario Sandbox")
    st.caption("Pas projecten aan en zie direct de impact op de forecast. Gebruik de sliders in de sidebar voor globale scenario's.")

    # Calculate current scenario impact
    timeseries_df = expand_projects_to_timeseries(
        st.session_state.sandbox_projects,
        price_multiplier=price_multiplier,
        capacity_pct=capacity_pct,
        forecast_months=6
    )

    monthly_summary = calculate_monthly_summary(
        timeseries_df,
        monthly_target=MONTHLY_TARGET,
        forecast_months=6
    )

    # Also calculate baseline (no scenario adjustments) for comparison
    timeseries_baseline = expand_projects_to_timeseries(
        st.session_state.sandbox_projects,
        price_multiplier=1.0,
        capacity_pct=100.0,
        forecast_months=6
    )
    baseline_total = timeseries_baseline["Omzet_Gewogen"].sum() if not timeseries_baseline.empty else 0

    # KPIs
    total_target = monthly_summary["Target"].sum()
    total_forecast = monthly_summary["Forecast_Totaal"].sum()
    total_hard = monthly_summary["Forecast_Hard"].sum()
    total_gap = total_target - total_forecast
    scenario_impact = total_forecast - baseline_total

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎯 Target (6 mnd)",
            format_currency_k(total_target),
            help="6 maanden × 4 partners × €12.650"
        )

    with col2:
        pct = (total_forecast / total_target * 100) if total_target > 0 else 0
        st.metric(
            "📈 Forecast (gewogen)",
            format_currency_k(total_forecast),
            delta=f"{pct:.0f}% van target",
            delta_color="normal"
        )

    with col3:
        gap_delta = f"-{total_gap/total_target*100:.0f}%" if total_gap > 0 else f"+{abs(total_gap)/total_target*100:.0f}%"
        st.metric(
            "⚠️ Gap",
            format_currency_k(total_gap),
            delta=gap_delta,
            delta_color="inverse"
        )

    with col4:
        if scenario_impact != 0:
            impact_sign = "+" if scenario_impact > 0 else ""
            st.metric(
                "🎚️ Scenario Impact",
                f"{impact_sign}{format_currency_k(scenario_impact)}",
                delta=f"prijs {price_impact:+d}%, cap {capacity_pct}%",
                delta_color="normal" if scenario_impact >= 0 else "inverse"
            )
        else:
            st.metric(
                "🎚️ Scenario Impact",
                "€ 0",
                delta="geen aanpassingen",
                delta_color="off"
            )

    st.markdown("---")

    # Gap Analysis Chart
    st.plotly_chart(
        create_gap_chart(monthly_summary),
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("---")

    # PROJECT SANDBOX (st.data_editor)
    st.subheader("📝 Project Pipeline")
    st.caption("✏️ Klik in een cel om te bewerken. Gebruik **+** onderaan om projecten toe te voegen. Wijzigingen worden direct verwerkt.")

    # Column configuration
    column_config = {
        "Project": st.column_config.TextColumn(
            "Projectnaam",
            help="Naam van het project of de klant",
            width="large",
            required=True
        ),
        "Startmaand": st.column_config.DateColumn(
            "Startmaand",
            help="Startdatum van het project",
            format="MMM YYYY",
            required=True
        ),
        "Duur": st.column_config.NumberColumn(
            "Duur (mnd)",
            help="Projectduur in maanden",
            min_value=1,
            max_value=12,
            step=1,
            required=True
        ),
        "Uurtarief": st.column_config.NumberColumn(
            "Tarief (€/u)",
            help="Uurtarief in euro's",
            min_value=50,
            max_value=300,
            step=5,
            format="€ %d",
            required=True
        ),
        "Uren_mnd": st.column_config.NumberColumn(
            "Uren/mnd",
            help="Verwacht aantal uren per maand",
            min_value=0,
            max_value=200,
            step=4,
            required=True
        ),
        "Kans": st.column_config.NumberColumn(
            "Kans (%)",
            help="Waarschijnlijkheid (0-100%)",
            min_value=0,
            max_value=100,
            step=10,
            format="%d%%",
            required=True
        ),
        "Status": st.column_config.SelectboxColumn(
            "Status",
            help="Pipeline = nog niet getekend, Getekend = contract rond",
            options=["Pipeline", "Getekend"],
            required=True
        )
    }

    # Data editor
    edited_df = st.data_editor(
        st.session_state.sandbox_projects,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="sandbox_editor"
    )

    # Update session state when data changes
    if not edited_df.equals(st.session_state.sandbox_projects):
        st.session_state.sandbox_projects = edited_df
        st.rerun()

    # Monthly breakdown table
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("#### 📋 Maandoverzicht")
        display_df = monthly_summary[["Maand_Label", "Forecast_Hard", "Forecast_Zacht", "Target", "Gap"]].copy()
        display_df.columns = ["Maand", "Getekend", "Pipeline", "Target", "Gap"]

        st.dataframe(
            display_df.style.format({
                "Getekend": "€{:,.0f}",
                "Pipeline": "€{:,.0f}",
                "Target": "€{:,.0f}",
                "Gap": "€{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True,
            height=280
        )

    with col_left:
        st.markdown("#### 📊 Verdeling Hard vs Zacht")

        # Pie chart for hard vs soft
        labels = ["Getekend", "Pipeline", "Gap"]
        values = [total_hard, total_forecast - total_hard, max(0, total_gap)]
        colors = ["#22c55e", "#3b82f6", "#fbbf24"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo="label+percent",
            textposition="outside"
        )])

        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=280
        )

        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 2: PROJECTEN (Existing category view)
# =============================================================================
with tab2:
    st.subheader("Projecten per categorie")
    st.caption("Bestaande projecten uit Google Sheets")

    # Load existing projects
    projects = load_projects()
    regie_projects = load_regie()
    all_projects = projects + regie_projects

    if all_projects:
        df = pd.DataFrame(all_projects)

        # Ensure columns exist
        if "Categorie" not in df.columns:
            df["Categorie"] = "-"
        df["Categorie"] = df["Categorie"].fillna("-")

        # Calculate totals
        for month in MONTH_COLS:
            if month not in df.columns:
                df[month] = 0

        df["Totaal"] = df[MONTH_COLS].sum(axis=1)

        # Display by category
        for row_idx in range(3):
            cols = st.columns(2)
            for col_idx in range(2):
                cat_idx = row_idx * 2 + col_idx
                if cat_idx < len(CATEGORIES):
                    emoji, cat_name = CATEGORIES[cat_idx]

                    if cat_name == "Regie":
                        cat_projects = df[df["Type"] == "Regie"]
                    else:
                        cat_projects = df[df["Categorie"] == cat_name]

                    cat_total = cat_projects["Totaal"].sum() if len(cat_projects) > 0 else 0

                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="category-header">
                            <h3>{emoji} {cat_name}</h3>
                            <div style="font-size: 24px; font-weight: bold; margin-top: 8px;">
                                €{cat_total/1000:.0f}k
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if len(cat_projects) > 0:
                            for _, row in cat_projects.iterrows():
                                st.markdown(f"**{row.get('Klant', '-')}** - {row.get('Opdracht', '-')}")
                                st.caption(f"€{row.get('Totaal', 0):,.0f} | {row.get('Partner', '-')}")
                        else:
                            st.info("Geen projecten")
    else:
        st.info("Geen projecten gevonden in Google Sheets")


# =============================================================================
# TAB 3: DATA BEHEER
# =============================================================================
with tab3:
    st.subheader("Data beheer & Upload")

    tab_upload, tab_edit, tab_export = st.tabs(["📤 Upload", "✏️ Bewerk projecten", "📥 Export"])

    with tab_upload:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Consultancy Uren (Excel)**")
            st.caption("Upload de SharePoint Excel met uren per partner")

            uren_file = st.file_uploader("Selecteer Excel", type=['xlsx'], key="uren_upload")

            if uren_file is not None:
                with st.spinner("Verwerken..."):
                    regie_data = parse_regie_excel(uren_file)

                if regie_data:
                    st.success(f"✅ {len(regie_data)} partners gevonden")
                    if save_regie_to_gsheets(regie_data):
                        st.success("Opgeslagen naar Google Sheets!")
                        st.rerun()

        with col2:
            st.markdown("**👥 Pipedrive Klanten**")
            st.caption("Upload Pipedrive organisatie export")

            pipedrive_file = st.file_uploader("Selecteer export", type=['xlsx'], key="pipedrive_upload")

            if pipedrive_file is not None:
                with st.spinner("Verwerken..."):
                    klanten_data = parse_pipedrive_excel(pipedrive_file)

                if klanten_data:
                    st.success(f"✅ {len(klanten_data)} klanten gevonden")
                    if save_klanten_to_gsheets(klanten_data):
                        st.success("Opgeslagen!")

    with tab_edit:
        st.caption("Bewerk alle projecten in een tabel")

        projects = load_projects()
        df_edit = pd.DataFrame(projects) if projects else pd.DataFrame(
            columns=["Klant", "Opdracht", "Type", "Partner", "Categorie"] + MONTH_COLS
        )

        edit_cols = ["Klant", "Opdracht", "Type", "Partner", "Categorie"] + MONTH_COLS
        for col in edit_cols:
            if col not in df_edit.columns:
                df_edit[col] = 0 if col in MONTH_COLS else ""

        edited_projects = st.data_editor(
            df_edit[edit_cols],
            column_config={
                "Klant": st.column_config.TextColumn("Klant", width="medium"),
                "Opdracht": st.column_config.TextColumn("Opdracht", width="large"),
                "Type": st.column_config.SelectboxColumn("Type", options=["Regie", "Afgeprijsd", "Product"]),
                "Partner": st.column_config.TextColumn("Partner"),
                "Categorie": st.column_config.SelectboxColumn(
                    "Categorie",
                    options=["-", "Groei in Corporates", "Migratie", "AI & Innovatie", "Data Analyse"]
                ),
                **{m: st.column_config.NumberColumn(m, format="€%.0f", min_value=0) for m in MONTH_COLS}
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("💾 Opslaan naar Google Sheets", type="primary"):
            edited_projects["Totaal"] = edited_projects[MONTH_COLS].sum(axis=1)
            new_projects = edited_projects.to_dict('records')
            if save_projects(new_projects):
                st.success("✅ Opgeslagen!")
                st.rerun()

    with tab_export:
        projects = load_projects()
        if projects:
            df_export = pd.DataFrame(projects)
            cols = ["Klant", "Opdracht", "Type", "Partner", "Categorie"] + MONTH_COLS + ["Totaal"]
            df_export = df_export.reindex(columns=[c for c in cols if c in df_export.columns])

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, sheet_name='Projecten', index=False)
            output.seek(0)

            st.download_button(
                label="⬇️ Download Excel",
                data=output,
                file_name="consultancy_projecten.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Geen projecten om te exporteren")


# Footer
st.markdown("---")
st.caption(f"Notifica Consultancy Forecast | Scenario: prijs {price_impact:+d}%, capaciteit {capacity_pct}%")
