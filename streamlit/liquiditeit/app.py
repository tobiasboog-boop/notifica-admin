"""
Liquiditeit Dashboard - Main Application
========================================
Streamlit app voor liquiditeitsanalyse en cashflow prognoses.

Notifica - Business Intelligence voor installatiebedrijven

Filteropties gebaseerd op Notifica Thin Reports:
- Administratie filter
- Relatie (debiteur/crediteur) filter
- Datum bereik filter
- Ouderdomscategorie filter

Transparantie:
- Duidelijk onderscheid tussen realisatie en prognose
- Historische data vs verwachte cashflows
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple

# Local imports
from config import AppConfig, COLORS, LIQUIDITY_THRESHOLDS
from src.database import get_database, MockDatabase, SyntessDWHConnection, FailedConnectionDatabase
from src.calculations import (
    calculate_liquidity_metrics,
    create_weekly_cashflow_forecast,
    calculate_aging_buckets,
    create_enhanced_cashflow_forecast,
    calculate_recurring_costs_per_week,
    create_ml_forecast,
    backtest_forecast_model,
    ForecastModelMetrics,
    LiquidityMetrics,
)

# Page configuration
st.set_page_config(
    page_title="Liquiditeit Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with additional styles for transparency indicators
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .transparency-legend {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    .transparency-legend .realisatie {
        color: #1E3A5F;
        font-weight: 600;
    }
    .transparency-legend .forecast {
        color: #6c757d;
        font-style: italic;
    }
    .filter-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FILTERING FUNCTIONS (gebaseerd op Thin Reports filteropties)
# =============================================================================

def apply_data_filters(
    data: Dict[str, pd.DataFrame],
    filters: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Pas filters toe op de data.

    Ondersteunde filters:
    - administratie: Filter op administratie naam
    - bedrijfseenheid: Filter op bedrijfseenheid
    - bankrekeningen: Filter op specifieke bankrekeningen
    """
    filtered_data = {k: v.copy() for k, v in data.items()}

    # Administratie filter
    admin_filter = filters.get("administratie")
    if admin_filter and admin_filter != "Alle":
        # Filter debiteuren
        if not filtered_data["debiteuren"].empty and "administratie" in filtered_data["debiteuren"].columns:
            filtered_data["debiteuren"] = filtered_data["debiteuren"][
                filtered_data["debiteuren"]["administratie"] == admin_filter
            ]

        # Filter crediteuren
        if not filtered_data["crediteuren"].empty and "administratie" in filtered_data["crediteuren"].columns:
            filtered_data["crediteuren"] = filtered_data["crediteuren"][
                filtered_data["crediteuren"]["administratie"] == admin_filter
            ]

    # Bedrijfseenheid filter
    be_filter = filters.get("bedrijfseenheid")
    if be_filter and be_filter != "Alle":
        # Filter debiteuren
        if not filtered_data["debiteuren"].empty and "bedrijfseenheid" in filtered_data["debiteuren"].columns:
            filtered_data["debiteuren"] = filtered_data["debiteuren"][
                filtered_data["debiteuren"]["bedrijfseenheid"] == be_filter
            ]

        # Filter crediteuren
        if not filtered_data["crediteuren"].empty and "bedrijfseenheid" in filtered_data["crediteuren"].columns:
            filtered_data["crediteuren"] = filtered_data["crediteuren"][
                filtered_data["crediteuren"]["bedrijfseenheid"] == be_filter
            ]

    # Filter banksaldo (op rekening indien filter actief)
    if not filtered_data["banksaldo"].empty:
        bank = filtered_data["banksaldo"]

        if filters.get("bankrekeningen") and len(filters["bankrekeningen"]) > 0:
            if "bank_naam" in bank.columns:
                bank = bank[bank["bank_naam"].isin(filters["bankrekeningen"])]

        filtered_data["banksaldo"] = bank

    return filtered_data


def get_filter_options(data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Haal beschikbare filteropties op uit de data."""
    options = {
        "debiteuren": [],
        "crediteuren": [],
        "bankrekeningen": [],
    }

    if not data["debiteuren"].empty and "debiteur_naam" in data["debiteuren"].columns:
        options["debiteuren"] = sorted(data["debiteuren"]["debiteur_naam"].unique().tolist())

    if not data["crediteuren"].empty and "crediteur_naam" in data["crediteuren"].columns:
        options["crediteuren"] = sorted(data["crediteuren"]["crediteur_naam"].unique().tolist())

    if not data["banksaldo"].empty and "bank_naam" in data["banksaldo"].columns:
        options["bankrekeningen"] = sorted(data["banksaldo"]["bank_naam"].unique().tolist())

    return options


# =============================================================================
# TRANSPARANTIE: REALISATIE vs FORECAST
# =============================================================================

def create_transparent_cashflow_forecast(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    salarissen: pd.DataFrame,
    historisch: pd.DataFrame,
    weeks_history: int = 4,
    weeks_forecast: int = 13,
    debiteur_delay_days: int = 0,
    crediteur_delay_days: int = 0,
    reference_date: date = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Maak cashflow overzicht met duidelijk onderscheid tussen:
    - REALISATIE: Historische data (afgelopen X weken vanaf standdatum)
    - PROGNOSE: Verwachte cashflow (komende Y weken vanaf standdatum)

    Args:
        reference_date: De standdatum - startpunt voor de forecast (default: vandaag)

    Returns:
        Tuple van (DataFrame met alle data, index waar forecast begint)
    """
    # Gebruik standdatum als referentiepunt (niet vandaag)
    today = reference_date if reference_date else datetime.now().date()

    # -------------------------------------------------------------------------
    # DEEL 1: HISTORISCHE DATA (REALISATIE) - uit historisch betalingsgedrag
    # -------------------------------------------------------------------------
    history_rows = []

    if not historisch.empty and "maand" in historisch.columns:
        # Gebruik laatste 4 weken van historische data
        hist = historisch.copy()
        hist["maand"] = pd.to_datetime(hist["maand"])
        hist = hist.sort_values("maand", ascending=False).head(weeks_history)

        for i, row in enumerate(hist.itertuples()):
            week_num = -(weeks_history - i)
            week_start = today + timedelta(weeks=week_num)
            history_rows.append({
                "week_nummer": week_num,
                "week_label": f"Week {week_num}",
                "week_start": week_start,
                "week_eind": week_start + timedelta(days=7),
                "inkomsten_debiteuren": getattr(row, "inkomsten", 0) / 4,  # Schatting per week
                "uitgaven_crediteuren": getattr(row, "uitgaven", 0) / 4,
                "uitgaven_salarissen": 0.0,
                "uitgaven_overig": 0.0,
                "netto_cashflow": (getattr(row, "inkomsten", 0) - getattr(row, "uitgaven", 0)) / 4,
                "is_realisatie": True,
                "data_type": "Realisatie",
            })
    else:
        # Genereer placeholder historische data als er geen historiek is
        for i in range(weeks_history):
            week_num = -(weeks_history - i)
            week_start = today + timedelta(weeks=week_num)
            history_rows.append({
                "week_nummer": week_num,
                "week_label": f"Week {week_num}",
                "week_start": week_start,
                "week_eind": week_start + timedelta(days=7),
                "inkomsten_debiteuren": 0.0,
                "uitgaven_crediteuren": 0.0,
                "uitgaven_salarissen": 0.0,
                "uitgaven_overig": 0.0,
                "netto_cashflow": 0.0,
                "is_realisatie": True,
                "data_type": "Realisatie",
            })

    # -------------------------------------------------------------------------
    # DEEL 2: PROGNOSE DATA (FORECAST) - uit openstaande posten
    # -------------------------------------------------------------------------
    forecast_rows = []
    week_starts = [today + timedelta(weeks=i) for i in range(weeks_forecast + 1)]

    for i in range(weeks_forecast):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]

        # Inkomsten van debiteuren (met vertraging scenario)
        deb_income = 0.0
        if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
            deb = debiteuren.copy()
            deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
            deb["verwachte_betaling"] = deb["vervaldatum"].apply(
                lambda x: x + timedelta(days=debiteur_delay_days) if x else x
            )
            mask = (deb["verwachte_betaling"] >= week_start) & (deb["verwachte_betaling"] < week_end)
            deb_income = deb.loc[mask, "openstaand"].sum()

        # Uitgaven aan crediteuren (met vertraging scenario)
        cred_expense = 0.0
        if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
            cred = crediteuren.copy()
            cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
            cred["verwachte_betaling"] = cred["vervaldatum"].apply(
                lambda x: x + timedelta(days=crediteur_delay_days) if x else x
            )
            mask = (cred["verwachte_betaling"] >= week_start) & (cred["verwachte_betaling"] < week_end)
            cred_expense = cred.loc[mask, "openstaand"].sum()

        # Salarissen
        sal_expense = 0.0
        if not salarissen.empty and "betaaldatum" in salarissen.columns:
            sal = salarissen.copy()
            sal["betaaldatum"] = pd.to_datetime(sal["betaaldatum"]).dt.date
            mask = (sal["betaaldatum"] >= week_start) & (sal["betaaldatum"] < week_end)
            sal_expense = sal.loc[mask, "bedrag"].sum()

        forecast_rows.append({
            "week_nummer": i + 1,
            "week_label": f"Week {i + 1}",
            "week_start": week_start,
            "week_eind": week_end,
            "inkomsten_debiteuren": deb_income,
            "uitgaven_crediteuren": cred_expense,
            "uitgaven_salarissen": sal_expense,
            "uitgaven_overig": 0.0,
            "netto_cashflow": deb_income - cred_expense - sal_expense,
            "is_realisatie": False,
            "data_type": "Prognose",
        })

    # Combineer historie en forecast
    all_rows = history_rows + forecast_rows
    df = pd.DataFrame(all_rows)

    # Bereken cumulatief saldo
    start_balance = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    df["cumulatief_saldo"] = start_balance + df["netto_cashflow"].cumsum()

    # Index waar forecast begint
    forecast_start_idx = len(history_rows)

    return df, forecast_start_idx


# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================

def load_data(use_mock: bool = True, customer_code: Optional[str] = None, standdatum: date = None, administratie: str = None):
    """Load data from database or mock at a specific reference date."""
    # Debug info in sidebar
    st.sidebar.caption(f"Data source: {'Demo' if use_mock else customer_code}")

    if standdatum is None:
        standdatum = datetime.now().date()

    db = get_database(use_mock=use_mock, customer_code=customer_code)

    # Check if connection failed
    if isinstance(db, FailedConnectionDatabase):
        st.error(f"Verbindingsfout: {db.error_msg}. Controleer de netwerk/VPN verbinding.")

    # Bereken periode voor historische data (12 maanden voor standdatum)
    hist_startdatum = date(standdatum.year - 1, standdatum.month, 1)
    hist_einddatum = standdatum

    data = {
        "banksaldo": db.get_banksaldo(standdatum=standdatum, administratie=administratie),
        "debiteuren": db.get_openstaande_debiteuren(standdatum=standdatum, administratie=administratie),
        "crediteuren": db.get_openstaande_crediteuren(standdatum=standdatum),
        "salarissen": db.get_geplande_salarissen(),
        "historisch": db.get_historisch_betalingsgedrag(),
    }

    # Laad extra data voor verbeterde prognose
    # Voor MockDatabase: altijd laden (geen administratie filter nodig)
    # Voor echte DB: alleen als administratie is ingesteld
    if hasattr(db, 'get_historische_cashflow_per_week'):
        if isinstance(db, MockDatabase):
            # MockDatabase: laad altijd
            data["terugkerende_kosten"] = db.get_terugkerende_kosten()
            data["historische_cashflow"] = db.get_historische_cashflow_per_week()
        elif administratie:
            # Echte DB: alleen met administratie filter
            data["terugkerende_kosten"] = db.get_terugkerende_kosten(
                startdatum=hist_startdatum,
                einddatum=hist_einddatum,
                administratie=administratie
            )
            data["historische_cashflow"] = db.get_historische_cashflow_per_week(
                startdatum=hist_startdatum,
                einddatum=hist_einddatum,
                administratie=administratie
            )
        else:
            data["terugkerende_kosten"] = pd.DataFrame()
            data["historische_cashflow"] = pd.DataFrame()
    else:
        data["terugkerende_kosten"] = pd.DataFrame()
        data["historische_cashflow"] = pd.DataFrame()

    return data


def render_kpi_cards(metrics: LiquidityMetrics):
    """Render KPI cards at top of dashboard."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Current Ratio",
            value=f"{metrics.current_ratio:.2f}",
            delta="Gezond" if metrics.current_ratio >= 1.5 else "Let op",
            delta_color="normal" if metrics.current_ratio >= 1.5 else "inverse"
        )

    with col2:
        st.metric(
            label="Quick Ratio",
            value=f"{metrics.quick_ratio:.2f}",
            delta="OK" if metrics.quick_ratio >= 1.0 else "Laag",
            delta_color="normal" if metrics.quick_ratio >= 1.0 else "inverse"
        )

    with col3:
        st.metric(
            label="Liquiditeit",
            value=f"€ {metrics.cash_position:,.0f}",
            help="Totaal banksaldo"
        )

    with col4:
        st.metric(
            label="Debiteuren",
            value=f"€ {metrics.total_receivables:,.0f}",
            help="Openstaande vorderingen"
        )

    with col5:
        st.metric(
            label="Crediteuren",
            value=f"€ {metrics.total_payables:,.0f}",
            help="Openstaande schulden"
        )


def render_transparent_cashflow_chart(forecast: pd.DataFrame, forecast_start_idx: int, reference_date: date = None):
    """
    Render cashflow chart met duidelijk onderscheid tussen REALISATIE en PROGNOSE.

    - Realisatie: Solid kleuren, volle opacity
    - Prognose: Gestreept patroon, lagere opacity
    """
    fig = go.Figure()

    # Split data in realisatie en prognose
    realisatie = forecast[forecast["is_realisatie"] == True]
    prognose = forecast[forecast["is_realisatie"] == False]

    # --- REALISATIE BARS (Solid) ---
    if not realisatie.empty:
        colors_real = [COLORS["primary"] if x >= 0 else COLORS["danger"] for x in realisatie["netto_cashflow"]]
        fig.add_trace(go.Bar(
            x=realisatie["week_label"],
            y=realisatie["netto_cashflow"],
            name="Realisatie",
            marker_color=colors_real,
            opacity=1.0,
            hovertemplate="<b>REALISATIE</b><br>%{x}<br>Cashflow: €%{y:,.0f}<extra></extra>",
            legendgroup="realisatie",
        ))

    # --- PROGNOSE BARS (Semi-transparent) ---
    if not prognose.empty:
        colors_prog = [COLORS["success"] if x >= 0 else COLORS["warning"] for x in prognose["netto_cashflow"]]
        fig.add_trace(go.Bar(
            x=prognose["week_label"],
            y=prognose["netto_cashflow"],
            name="Prognose",
            marker_color=colors_prog,
            opacity=0.6,
            marker_pattern_shape="/",
            hovertemplate="<b>PROGNOSE</b><br>%{x}<br>Verwacht: €%{y:,.0f}<extra></extra>",
            legendgroup="prognose",
        ))

    # --- CUMULATIEF SALDO LIJN ---
    # Realisatie deel (solid lijn)
    if not realisatie.empty:
        fig.add_trace(go.Scatter(
            x=realisatie["week_label"],
            y=realisatie["cumulatief_saldo"],
            name="Saldo (Realisatie)",
            line=dict(color=COLORS["primary"], width=3),
            mode="lines+markers",
            marker=dict(size=8),
            hovertemplate="<b>REALISATIE</b><br>%{x}<br>Saldo: €%{y:,.0f}<extra></extra>",
            legendgroup="realisatie",
        ))

    # Prognose deel (dashed lijn)
    if not prognose.empty:
        fig.add_trace(go.Scatter(
            x=prognose["week_label"],
            y=prognose["cumulatief_saldo"],
            name="Saldo (Prognose)",
            line=dict(color=COLORS["secondary"], width=3, dash="dash"),
            mode="lines+markers",
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>PROGNOSE</b><br>%{x}<br>Verwacht saldo: €%{y:,.0f}<extra></extra>",
            legendgroup="prognose",
        ))

    # Verbindingslijn tussen realisatie en prognose
    if not realisatie.empty and not prognose.empty:
        fig.add_trace(go.Scatter(
            x=[realisatie["week_label"].iloc[-1], prognose["week_label"].iloc[0]],
            y=[realisatie["cumulatief_saldo"].iloc[-1], prognose["cumulatief_saldo"].iloc[0]],
            line=dict(color=COLORS["neutral"], width=2, dash="dot"),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Minimum buffer lijn
    if not forecast.empty:
        start_balance = forecast["cumulatief_saldo"].iloc[0]
        min_buffer = start_balance * 0.2
        fig.add_hline(
            y=min_buffer,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text="Minimum buffer (20%)",
            annotation_position="right"
        )

    # Verticale scheidingslijn tussen realisatie en prognose
    if forecast_start_idx > 0 and forecast_start_idx < len(forecast):
        # Label met standdatum als die is ingesteld
        date_label = reference_date.strftime("%d-%m-%Y") if reference_date else "Vandaag"
        fig.add_vline(
            x=forecast_start_idx - 0.5,
            line_dash="solid",
            line_color="#333",
            line_width=2,
            annotation_text=f"Standdatum: {date_label}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Cashflow Overzicht: Realisatie + Prognose",
        xaxis_title="",
        yaxis_title="Bedrag (€)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        height=500,
        barmode="relative",
    )

    return fig


def render_transparency_legend():
    """Render legenda die uitlegt wat realisatie vs prognose betekent."""
    st.markdown("""
    <div class="transparency-legend">
        <strong>📊 Transparantie in dit dashboard:</strong><br>
        <span class="realisatie">■ REALISATIE</span> = Werkelijke historische data (betalingen die al zijn gedaan)<br>
        <span class="forecast">◇ PROGNOSE</span> = Verwachte toekomstige cashflow (op basis van openstaande posten en vervaldatums)
    </div>
    """, unsafe_allow_html=True)


def render_pie_charts_with_filter(debiteuren: pd.DataFrame, crediteuren: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Render pie charts showing composition of debiteuren and crediteuren.
    Returns selected debiteur/crediteur names for filtering the aging analysis.
    """
    st.subheader("Openstaande Posten per Relatie")
    st.caption("Selecteer een relatie om de ouderdomsanalyse te filteren")

    selected_deb = None
    selected_cred = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Debiteuren**")
        if not debiteuren.empty and "debiteur_naam" in debiteuren.columns:
            # Group by debiteur and sum openstaand
            deb_grouped = debiteuren.groupby("debiteur_naam")["openstaand"].sum().reset_index()
            deb_grouped = deb_grouped.sort_values("openstaand", ascending=False)

            # Top 10 + Others
            if len(deb_grouped) > 10:
                top10 = deb_grouped.head(10)
                others = pd.DataFrame({
                    "debiteur_naam": ["Overige"],
                    "openstaand": [deb_grouped.iloc[10:]["openstaand"].sum()]
                })
                deb_grouped = pd.concat([top10, others], ignore_index=True)

            totaal = deb_grouped["openstaand"].sum()

            fig_deb = px.pie(
                deb_grouped,
                values="openstaand",
                names="debiteur_naam",
                hole=0.4,
            )
            fig_deb.update_traces(textposition='outside', textinfo='percent+label')
            fig_deb.update_layout(
                showlegend=False,
                height=350,
                annotations=[dict(text=f"EUR {totaal:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig_deb, key="pie_deb_chart", use_container_width=True)

            # Selectbox als filter
            deb_options = ["Alle debiteuren"] + deb_grouped["debiteur_naam"].tolist()
            selected_deb = st.selectbox(
                "Filter op debiteur:",
                deb_options,
                key="deb_select",
                help="Selecteer een debiteur om de ouderdomsanalyse hieronder te filteren"
            )

            if selected_deb == "Alle debiteuren" or selected_deb == "Overige":
                selected_deb = None

            # Toon factuurdetails als er een specifieke debiteur is geselecteerd
            if selected_deb:
                detail_df = debiteuren[debiteuren["debiteur_naam"] == selected_deb].copy()
                if not detail_df.empty:
                    st.markdown(f"**Facturen van {selected_deb}:**")
                    detail_cols = ["factuurnummer", "factuurdatum", "vervaldatum", "openstaand"]
                    available_cols = [c for c in detail_cols if c in detail_df.columns]
                    st.dataframe(
                        detail_df[available_cols].style.format({"openstaand": "EUR {:,.2f}"}),
                        hide_index=True,
                        use_container_width=True
                    )
        else:
            st.info("Geen debiteuren data beschikbaar")

    with col2:
        st.markdown("**Crediteuren**")
        if not crediteuren.empty and "crediteur_naam" in crediteuren.columns:
            # Group by crediteur and sum openstaand
            cred_grouped = crediteuren.groupby("crediteur_naam")["openstaand"].sum().reset_index()
            cred_grouped = cred_grouped.sort_values("openstaand", ascending=False)

            # Top 10 + Others
            if len(cred_grouped) > 10:
                top10 = cred_grouped.head(10)
                others = pd.DataFrame({
                    "crediteur_naam": ["Overige"],
                    "openstaand": [cred_grouped.iloc[10:]["openstaand"].sum()]
                })
                cred_grouped = pd.concat([top10, others], ignore_index=True)

            totaal = cred_grouped["openstaand"].sum()

            fig_cred = px.pie(
                cred_grouped,
                values="openstaand",
                names="crediteur_naam",
                hole=0.4,
            )
            fig_cred.update_traces(textposition='outside', textinfo='percent+label')
            fig_cred.update_layout(
                showlegend=False,
                height=350,
                annotations=[dict(text=f"EUR {totaal:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig_cred, key="pie_cred_chart", use_container_width=True)

            # Selectbox als filter
            cred_options = ["Alle crediteuren"] + cred_grouped["crediteur_naam"].tolist()
            selected_cred = st.selectbox(
                "Filter op crediteur:",
                cred_options,
                key="cred_select",
                help="Selecteer een crediteur om de ouderdomsanalyse hieronder te filteren"
            )

            if selected_cred == "Alle crediteuren" or selected_cred == "Overige":
                selected_cred = None

            # Toon factuurdetails als er een specifieke crediteur is geselecteerd
            if selected_cred:
                detail_df = crediteuren[crediteuren["crediteur_naam"] == selected_cred].copy()
                if not detail_df.empty:
                    st.markdown(f"**Facturen van {selected_cred}:**")
                    detail_cols = ["factuurnummer", "factuurdatum", "vervaldatum", "openstaand"]
                    available_cols = [c for c in detail_cols if c in detail_df.columns]
                    st.dataframe(
                        detail_df[available_cols].style.format({"openstaand": "EUR {:,.2f}"}),
                        hide_index=True,
                        use_container_width=True
                    )
        else:
            st.info("Geen crediteuren data beschikbaar")

    return selected_deb, selected_cred


def render_aging_chart(
    aging_deb: pd.DataFrame,
    aging_cred: pd.DataFrame,
    selected_deb: Optional[str] = None,
    selected_cred: Optional[str] = None
):
    """Render aging analysis charts, optionally filtered by selected relation."""
    st.subheader("Ouderdomsanalyse")

    # Toon filter indicatie als er een filter actief is
    if selected_deb or selected_cred:
        filter_parts = []
        if selected_deb:
            filter_parts.append(f"Debiteur: **{selected_deb}**")
        if selected_cred:
            filter_parts.append(f"Crediteur: **{selected_cred}**")
        st.info(f"Gefilterd op: {', '.join(filter_parts)}")

    col1, col2 = st.columns(2)

    with col1:
        title = f"**Debiteuren per vervaldatum**"
        if selected_deb:
            title += f" ({selected_deb})"
        st.markdown(title)

        fig_deb = px.bar(
            aging_deb,
            x="bucket",
            y="bedrag",
            color="bucket",
            color_discrete_sequence=[COLORS["success"], COLORS["secondary"],
                                     COLORS["warning"], COLORS["danger"], "#8B0000"],
            text="percentage",
        )
        fig_deb.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_deb.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="Bedrag (€)")
        st.plotly_chart(fig_deb, key="aging_deb_chart")

    with col2:
        title = f"**Crediteuren per vervaldatum**"
        if selected_cred:
            title += f" ({selected_cred})"
        st.markdown(title)

        fig_cred = px.bar(
            aging_cred,
            x="bucket",
            y="bedrag",
            color="bucket",
            color_discrete_sequence=[COLORS["success"], COLORS["secondary"],
                                     COLORS["warning"], COLORS["danger"], "#8B0000"],
            text="percentage",
        )
        fig_cred.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_cred.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="Bedrag (€)")
        st.plotly_chart(fig_cred, key="aging_cred_chart")


def render_cashflow_details(forecast: pd.DataFrame):
    """Render detailed cashflow table with realisatie/prognose indicator."""
    st.subheader("Cashflow Details per Week")

    # Bepaal welke kolommen beschikbaar zijn
    has_vaste_lasten = "uitgaven_vaste_lasten" in forecast.columns
    has_salarissen = "uitgaven_salarissen" in forecast.columns

    # Selecteer kolommen afhankelijk van wat beschikbaar is
    base_cols = ["data_type", "week_label", "week_start", "inkomsten_debiteuren", "uitgaven_crediteuren"]
    if has_vaste_lasten:
        base_cols.append("uitgaven_vaste_lasten")
    elif has_salarissen:
        base_cols.append("uitgaven_salarissen")
    base_cols.extend(["netto_cashflow", "cumulatief_saldo"])

    display_df = forecast[base_cols].copy()

    # Rename kolommen
    if has_vaste_lasten:
        display_df.columns = ["Type", "Week", "Startdatum", "Inkomsten", "Crediteuren", "Vaste Lasten", "Netto", "Saldo"]
        currency_cols = ["Inkomsten", "Crediteuren", "Vaste Lasten", "Netto", "Saldo"]
    elif has_salarissen:
        display_df.columns = ["Type", "Week", "Startdatum", "Inkomsten", "Crediteuren", "Salarissen", "Netto", "Saldo"]
        currency_cols = ["Inkomsten", "Crediteuren", "Salarissen", "Netto", "Saldo"]
    else:
        display_df.columns = ["Type", "Week", "Startdatum", "Inkomsten", "Crediteuren", "Netto", "Saldo"]
        currency_cols = ["Inkomsten", "Crediteuren", "Netto", "Saldo"]

    def highlight_type(row):
        if row["Type"] == "Realisatie":
            return ["background-color: #e3f2fd"] * len(row)
        else:
            return ["background-color: #fff8e1"] * len(row)

    st.dataframe(
        display_df.style.format({
            col: "EUR {:,.0f}" for col in currency_cols
        }).apply(highlight_type, axis=1),
        hide_index=True,
        height=400,
    )

    # Legenda voor tabel
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Blauw = Realisatie (historische data)")
    with col2:
        st.caption("Geel = Prognose (verwachte cashflow)")


def render_filter_sidebar(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Render filter controls in sidebar.
    Filters: Administratie, Bedrijfseenheid, Standdatum, Bankrekening
    """
    st.sidebar.header("🔍 Filters")

    filters = {}

    # Administratie filter
    administraties = set()
    if not data["debiteuren"].empty and "administratie" in data["debiteuren"].columns:
        administraties.update(data["debiteuren"]["administratie"].dropna().unique())
    if not data["crediteuren"].empty and "administratie" in data["crediteuren"].columns:
        administraties.update(data["crediteuren"]["administratie"].dropna().unique())

    if administraties:
        admin_options = sorted([a for a in administraties if a and a != "Onbekend"])
        if admin_options:
            filters["administratie"] = st.sidebar.selectbox(
                "Administratie",
                options=["Alle"] + admin_options,
                index=0,
                help="Filter op administratie/bedrijfsonderdeel"
            )

    # Bedrijfseenheid filter
    bedrijfseenheden = set()
    if not data["debiteuren"].empty and "bedrijfseenheid" in data["debiteuren"].columns:
        bedrijfseenheden.update(data["debiteuren"]["bedrijfseenheid"].dropna().unique())
    if not data["crediteuren"].empty and "bedrijfseenheid" in data["crediteuren"].columns:
        bedrijfseenheden.update(data["crediteuren"]["bedrijfseenheid"].dropna().unique())

    if bedrijfseenheden:
        be_options = sorted([b for b in bedrijfseenheden if b and b != "Onbekend"])
        if be_options:
            filters["bedrijfseenheid"] = st.sidebar.selectbox(
                "Bedrijfseenheid",
                options=["Alle"] + be_options,
                index=0,
                help="Filter op bedrijfseenheid"
            )

    # Bankrekening filter (standdatum is nu in main() gedefinieerd, vóór data laden)
    options = get_filter_options(data)
    if options["bankrekeningen"]:
        filters["bankrekeningen"] = st.sidebar.multiselect(
            "Bankrekeningen",
            options=options["bankrekeningen"],
            default=[],
            help="Filter op specifieke bankrekeningen"
        )

    return filters


def render_scenario_sidebar():
    """Render scenario analysis controls in sidebar."""
    st.sidebar.header("Scenario Analyse")

    st.sidebar.markdown("**Betalingsgedrag simulatie:**")

    debiteur_delay = st.sidebar.slider(
        "Debiteuren betalen later (dagen)",
        min_value=0,
        max_value=30,
        value=0,
        help="Simuleer wat er gebeurt als klanten X dagen later betalen"
    )

    crediteur_delay = st.sidebar.slider(
        "Crediteuren later betalen (dagen)",
        min_value=0,
        max_value=30,
        value=0,
        help="Simuleer wat er gebeurt als we leveranciers X dagen later betalen"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Prognose instellingen:**")

    forecast_weeks = st.sidebar.selectbox(
        "Weken prognose vooruit",
        options=[8, 13, 26, 52],
        index=1,
        help="Aantal weken voor de cashflow prognose"
    )

    history_weeks = st.sidebar.selectbox(
        "Weken historie terug",
        options=[0, 4, 8, 13],
        index=1,
        help="Aantal weken historische data tonen (realisatie)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Prognose methodiek:**")

    forecast_method = st.sidebar.radio(
        "Kies forecast model",
        options=["ML Ensemble (aanbevolen)", "Blend (oud)", "Alleen openstaande posten"],
        index=0,
        help="""
        **ML Ensemble**: Combineert seizoenspatroon, trend, weighted moving average en bekende posten.
        Inclusief automatische backtesting voor betrouwbaarheidsmeting.

        **Blend**: Eenvoudige combinatie van bekende posten en historisch gemiddelde.

        **Alleen openstaande posten**: Puur gebaseerd op facturen met vervaldatum.
        """
    )

    use_ml = forecast_method == "ML Ensemble (aanbevolen)"
    use_historical_profile = forecast_method in ["ML Ensemble (aanbevolen)", "Blend (oud)"]
    use_seasonality = forecast_method == "ML Ensemble (aanbevolen)"

    return debiteur_delay, crediteur_delay, forecast_weeks, history_weeks, use_historical_profile, use_seasonality, use_ml


def render_alerts(forecast: pd.DataFrame, metrics: LiquidityMetrics):
    """Render alert messages based on liquidity status."""
    alerts = []

    # Check for negative cash balance in forecast (only prognose weeks)
    prognose_weeks = forecast[forecast["is_realisatie"] == False]
    if not prognose_weeks.empty:
        negative_weeks = prognose_weeks[prognose_weeks["cumulatief_saldo"] < 0]
        if not negative_weeks.empty:
            first_negative = negative_weeks.iloc[0]
            alerts.append({
                "type": "danger",
                "message": f"⚠️ **Let op:** Verwacht negatief saldo in {first_negative['week_label']} "
                           f"(€ {first_negative['cumulatief_saldo']:,.0f})"
            })

    # Check current ratio
    if metrics.current_ratio < LIQUIDITY_THRESHOLDS["current_ratio_danger"]:
        alerts.append({
            "type": "danger",
            "message": f"🚨 **Current ratio te laag:** {metrics.current_ratio:.2f} "
                       f"(minimum: {LIQUIDITY_THRESHOLDS['current_ratio_danger']})"
        })
    elif metrics.current_ratio < LIQUIDITY_THRESHOLDS["current_ratio_warning"]:
        alerts.append({
            "type": "warning",
            "message": f"⚡ **Current ratio onder streefwaarde:** {metrics.current_ratio:.2f} "
                       f"(streef: {LIQUIDITY_THRESHOLDS['current_ratio_warning']})"
        })

    # Check days cash on hand
    if metrics.days_cash_on_hand < LIQUIDITY_THRESHOLDS["min_cash_buffer_days"]:
        alerts.append({
            "type": "warning",
            "message": f"💰 **Beperkte kasbuffer:** {metrics.days_cash_on_hand:.0f} dagen "
                       f"(aanbevolen: {LIQUIDITY_THRESHOLDS['min_cash_buffer_days']} dagen)"
        })

    # Render alerts
    for alert in alerts:
        if alert["type"] == "danger":
            st.error(alert["message"])
        elif alert["type"] == "warning":
            st.warning(alert["message"])
        else:
            st.info(alert["message"])

    if not alerts:
        st.success("✅ Liquiditeitspositie is gezond")


def render_datamodel_tab():
    """Render datamodel tab showing the database structure used by this dashboard."""
    st.header("Datamodel")
    st.caption("Overzicht van de database tabellen en relaties die dit dashboard gebruikt")

    # === SECTIE 1: OVERZICHT ===
    st.subheader("1. Database Structuur")

    st.markdown("""
    Dit dashboard haalt data uit de **Syntess DWH** (Data Warehouse). De data is georganiseerd in verschillende schema's:

    | Schema | Doel | Voorbeelden |
    |--------|------|-------------|
    | `notifica` | SSM (Self-Service Model) views | Verkoopfacturen, Inkoopfacturen, Administraties |
    | `financieel` | Financiële boekingen | Journaalregels, Rubrieken |
    | `stam` | Stamgegevens | Documenten, Dagboeken |
    """)

    # === SECTIE 2: ENTITEITEN DIAGRAM ===
    st.subheader("2. Entiteiten & Relaties")

    # Mermaid-achtige visualisatie met Streamlit
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           SYNTESS DWH DATAMODEL                              │
    └─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐       ┌──────────────────────┐
    │  SSM Administraties  │───────│  SSM Bedrijfseenheden│
    │──────────────────────│  1:N  │──────────────────────│
    │ • AdministratieKey   │       │ • BedrijfseenheidKey │
    │ • Administratie      │       │ • Bedrijfseenheid    │
    └──────────────────────┘       │ • AdministratieKey   │
            │                      └──────────────────────┘
            │ 1:N                           │ 1:N
            ▼                               ▼
    ┌──────────────────────┐       ┌──────────────────────┐
    │    SSM Documenten    │◄──────│ Verkoopfactuur       │
    │──────────────────────│  N:1  │ termijnen            │
    │ • DocumentKey        │       │──────────────────────│
    │ • Document code      │       │ • VerkoopfactuurKey  │
    │ • BedrijfseenheidKey │       │ • Debiteur           │
    └──────────────────────┘       │ • Vervaldatum        │
            │                      │ • Bedrag             │
            │                      │ • Alloc_datum        │
            │                      └──────────────────────┘
            │
            │                      ┌──────────────────────┐
            └─────────────────────►│ Inkoopfactuur        │
                              N:1  │ termijnen            │
                                   │──────────────────────│
                                   │ • InkoopFactuurKey   │
                                   │ • Crediteur          │
                                   │ • Vervaldatum        │
                                   │ • Bedrag             │
                                   │ • Bankafschrift status│
                                   └──────────────────────┘

    ┌──────────────────────┐       ┌──────────────────────┐
    │      Dagboeken       │───────│    Journaalregels    │
    │──────────────────────│  1:N  │──────────────────────│
    │ • DagboekKey         │       │ • JournaalregelKey   │
    │ • Dagboek            │       │ • DocumentKey        │
    │ • AdministratieKey   │       │ • Boekdatum          │
    │ • DagboekRubriekKey  │       │ • Bedrag             │
    └──────────────────────┘       │ • Debet/Credit       │
                                   │ • RubriekKey         │
                                   │ • AdministratieKey   │
                                   └──────────────────────┘
                                           │
                                           │ N:1
                                           ▼
                                   ┌──────────────────────┐
                                   │      Rubrieken       │
                                   │──────────────────────│
                                   │ • RubriekKey         │
                                   │ • Rubriek Code       │
                                   │ • Rubriek            │
                                   └──────────────────────┘
    ```
    """)

    # === SECTIE 3: TABELLEN DETAIL ===
    st.subheader("3. Tabel Details")

    with st.expander("📊 notifica.\"SSM Verkoopfactuur termijnen\"", expanded=False):
        st.markdown("""
        **Doel:** Openstaande debiteuren (verkoopfacturen)

        | Kolom | Type | Beschrijving |
        |-------|------|--------------|
        | `VerkoopfactuurDocumentKey` | bigint | FK naar SSM Documenten |
        | `Debiteur` | varchar | Naam van de debiteur |
        | `Vervaldatum` | date | Verwachte betaaldatum |
        | `Bedrag` | decimal | Factuurbedrag |
        | `Alloc_datum` | date | Datum van allocatie (gebruikt voor standdatum filter) |

        **Filter logica:**
        - `Alloc_datum <= standdatum` - alleen posten t/m de standdatum
        - `HAVING ABS(SUM(Bedrag)) > 0.01` - alleen niet-nul saldi
        """)

    with st.expander("📊 notifica.\"SSM Inkoopfactuur termijnen\"", expanded=False):
        st.markdown("""
        **Doel:** Openstaande crediteuren (inkoopfacturen)

        | Kolom | Type | Beschrijving |
        |-------|------|--------------|
        | `InkoopFactuurKey` | bigint | FK naar SSM Documenten |
        | `Crediteur` | varchar | Naam van de crediteur |
        | `Vervaldatum` | date | Verwachte betaaldatum |
        | `Bedrag` | decimal | Factuurbedrag |
        | `Alloc_datum` | date | Datum van allocatie |
        | `Bankafschrift status` | varchar | 'Openstaand' of 'Betaald' |

        **Filter logica:**
        - `Alloc_datum <= standdatum`
        - `Bankafschrift status = 'Openstaand'` - alleen onbetaalde facturen
        """)

    with st.expander("📊 financieel.\"Journaalregels\"", expanded=False):
        st.markdown("""
        **Doel:** Alle financiële boekingen (basis voor banksaldi en cashflow analyse)

        | Kolom | Type | Beschrijving |
        |-------|------|--------------|
        | `DocumentKey` | bigint | FK naar stam.Documenten |
        | `Boekdatum` | date | Datum van de boeking |
        | `Bedrag` | decimal | Bedrag van de boeking |
        | `Debet/Credit` | char(1) | 'D' = Debet, 'C' = Credit |
        | `RubriekKey` | bigint | FK naar Rubrieken |
        | `AdministratieKey` | bigint | FK naar Administraties |

        **Gebruik:**
        - **Banksaldi:** Filter op `StandaardEntiteitKey = 10` (bankdocumenten)
        - **Terugkerende kosten:** Filter op rubriekcodes (4xxx, 61xx, etc.)
        - **Historische cashflow:** Aggregatie per week/maand
        """)

    with st.expander("📊 stam.\"Dagboeken\"", expanded=False):
        st.markdown("""
        **Doel:** Dagboekdefinities (banken, kas, memoriaal, etc.)

        | Kolom | Type | Beschrijving |
        |-------|------|--------------|
        | `DagboekKey` | bigint | Primary key |
        | `Dagboek` | varchar | Naam van het dagboek (bijv. "ABN Bank") |
        | `AdministratieKey` | bigint | FK naar Administraties |
        | `DagboekRubriekKey` | bigint | Grootboekrekening van het dagboek |

        **Gebruik:**
        - Banksaldi worden berekend per dagboek
        - `DagboekRubriekKey` wordt gebruikt om alleen boekingen OP de bankrekening te selecteren
        """)

    with st.expander("📊 financieel.\"Rubrieken\"", expanded=False):
        st.markdown("""
        **Doel:** Grootboekrekeningen (rubrieken)

        | Kolom | Type | Beschrijving |
        |-------|------|--------------|
        | `RubriekKey` | bigint | Primary key |
        | `Rubriek Code` | varchar | Rekeningnummer (bijv. "1230", "4000") |
        | `Rubriek` | varchar | Omschrijving |

        **Belangrijke rubriekcodes:**
        - `1230` - Voorziening debiteuren
        - `4xxx` - Personeelskosten
        - `61xx` - Huisvestingskosten
        - `62xx` - Machinekosten
        - `65xx` - Autokosten
        """)

    # === SECTIE 4: KEY FILTERS ===
    st.subheader("4. Belangrijke Filters")

    st.markdown("""
    Het dashboard gebruikt de volgende key filters:

    | Filter | Tabel | Kolom | Waarde |
    |--------|-------|-------|--------|
    | Bank documenten | stam.Documenten | `StandaardEntiteitKey` | `= 10` |
    | Alleen bankrekening boekingen | financieel.Journaalregels | `RubriekKey` | `= dag.DagboekRubriekKey` |
    | Openstaande crediteuren | notifica.SSM Inkoopfactuur termijnen | `Bankafschrift status` | `= 'Openstaand'` |
    | Administratie filter | notifica.SSM Administraties | `Administratie` of `AdministratieKey` | Geselecteerde waarde |
    """)

    # === SECTIE 5: PROGNOSE METHODIEK ===
    st.subheader("5. Prognose Methodiek")

    st.markdown("""
    **Stap 1: Realisatie data**
    - Historische bankmutaties per week uit `financieel.Journaalregels`
    - Filter: laatste X weken voor standdatum

    **Stap 2: Bekende toekomstige cashflow**
    - Openstaande debiteuren → verwachte inkomsten (per vervaldatum)
    - Openstaande crediteuren → verwachte uitgaven (per vervaldatum)

    **Stap 3: Historisch profiel (optioneel, 25% nauwkeuriger)**
    - Gemiddelde inkomsten/uitgaven per maand uit historische data
    - Seizoenscorrectie per maand
    - Combinatie: `max(bekende_post, 80% * historisch_gemiddelde)`

    **Stap 4: Cumulatief saldo**
    - Start met huidig banksaldo
    - Tel netto cashflow per week op
    """)


def render_validation_tab(data: dict, customer_code: str, load_timestamp: datetime):
    """Render validation tab with totals, SQL queries, and forecast logic."""
    from src.database import (
        QUERY_BANKSALDO,
        QUERY_OPENSTAANDE_DEBITEUREN,
        QUERY_OPENSTAANDE_CREDITEUREN,
        QUERY_HISTORISCH_BETALINGSGEDRAG,
    )

    st.header("Validatie & Aansluiting")
    st.caption(f"Data geladen op: **{load_timestamp.strftime('%Y-%m-%d %H:%M:%S')}** | Klant: **{customer_code}**")

    # === SECTIE 1: CONTROLETOTALEN ===
    st.subheader("1. Controletotalen")
    st.markdown("*Vergelijk deze totalen met Power BI om de aansluiting te valideren.*")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Banksaldi**")
        if not data["banksaldo"].empty:
            totaal_bank = data["banksaldo"]["saldo"].sum()
            st.metric("Totaal liquide middelen", f"€ {totaal_bank:,.2f}")
            st.dataframe(
                data["banksaldo"][["bank_naam", "rekeningnummer", "saldo"]].style.format({"saldo": "€ {:,.2f}"}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Geen banksaldi gevonden")

    with col2:
        st.markdown("**Openstaande Debiteuren**")
        if not data["debiteuren"].empty:
            totaal_deb = data["debiteuren"]["openstaand"].sum()
            aantal_deb = len(data["debiteuren"])
            st.metric("Totaal openstaand", f"€ {totaal_deb:,.2f}")
            st.metric("Aantal facturen", f"{aantal_deb}")
        else:
            st.warning("Geen openstaande debiteuren")

    with col3:
        st.markdown("**Openstaande Crediteuren**")
        if not data["crediteuren"].empty:
            totaal_cred = data["crediteuren"]["openstaand"].sum()
            aantal_cred = len(data["crediteuren"])
            st.metric("Totaal openstaand", f"€ {totaal_cred:,.2f}")
            st.metric("Aantal facturen", f"{aantal_cred}")
        else:
            st.warning("Geen openstaande crediteuren")

    # === SECTIE 2: SQL QUERIES ===
    st.markdown("---")
    st.subheader("2. Gebruikte SQL Queries")
    st.markdown("*Dit zijn de exacte queries die gebruikt worden om data uit het DWH te halen.*")

    with st.expander("🏦 Banksaldi Query", expanded=False):
        st.code(QUERY_BANKSALDO, language="sql")
        st.caption("Haalt alle liquide middelen (rubriek 1xxx) op uit de journaalregels.")

    with st.expander("📥 Openstaande Debiteuren Query", expanded=False):
        st.code(QUERY_OPENSTAANDE_DEBITEUREN, language="sql")
        st.caption("Haalt alle openstaande vorderingen op (niet betaald, niet geannuleerd).")

    with st.expander("📤 Openstaande Crediteuren Query", expanded=False):
        st.code(QUERY_OPENSTAANDE_CREDITEUREN, language="sql")
        st.caption("Haalt alle openstaande schulden op (niet betaald, niet geannuleerd).")

    with st.expander("📊 Historisch Betalingsgedrag Query", expanded=False):
        st.code(QUERY_HISTORISCH_BETALINGSGEDRAG, language="sql")
        st.caption("Haalt maandelijkse omzet op voor trendanalyse (laatste 12 maanden).")

    # === SECTIE 3: PROGNOSE LOGICA ===
    st.markdown("---")
    st.subheader("3. Prognose Methodologie")
    st.markdown("*Hoe de cashflow prognose wordt berekend.*")

    st.markdown("""
    #### Inkomende Cashflow (Debiteuren)

    De prognose voor inkomende betalingen is gebaseerd op:

    1. **Openstaande facturen**: Elke openstaande debiteurenfactuur wordt meegenomen
    2. **Verwachte betaaldatum**: Berekend als `vervaldatum + scenario_vertraging`
       - De vervaldatum komt uit het ERP systeem
       - De scenario-vertraging (standaard 0 dagen) kan aangepast worden in de sidebar
    3. **Weekbucket**: Facturen worden gegroepeerd per week waarin betaling verwacht wordt

    ```python
    verwachte_betaaldatum = factuur.vervaldatum + timedelta(days=debiteur_vertraging)
    week_nummer = (verwachte_betaaldatum - vandaag).days // 7
    ```

    #### Uitgaande Cashflow (Crediteuren)

    De prognose voor uitgaande betalingen werkt identiek:

    1. **Openstaande facturen**: Elke openstaande crediteurenfactuur
    2. **Verwachte betaaldatum**: `vervaldatum + scenario_vertraging`
    3. **Weekbucket**: Groepering per verwachte betalingsweek

    #### Cumulatief Saldo

    Het verwachte banksaldo per week wordt als volgt berekend:

    ```python
    saldo_week_n = saldo_week_n-1 + inkomsten_week_n - uitgaven_week_n
    ```

    Waarbij:
    - `saldo_week_0` = huidige banksaldo (som van alle liquide middelen)
    - `inkomsten_week_n` = som van alle debiteurenfacturen die in week n verwacht worden
    - `uitgaven_week_n` = som van alle crediteurenfacturen die in week n betaald moeten worden

    #### Realisatie vs Prognose

    In de grafiek wordt onderscheid gemaakt tussen:
    - **Realisatie** (donkere kleuren): Historische data uit het verleden
    - **Prognose** (lichtere kleuren/gestreept): Verwachte toekomstige cashflows

    De scheidslijn is de huidige datum.
    """)

    # === SECTIE 4: DATA BRONNEN ===
    st.markdown("---")
    st.subheader("4. Data Bronnen")

    st.markdown(f"""
    | Bron | Tabel | Schema |
    |------|-------|--------|
    | Banksaldi | `SSM Journaalregels` + `SSM Rubrieken` | `notifica` |
    | Debiteuren | `SSM Betalingen per opbrengstregel` + `SSM Relaties` | `notifica` |
    | Crediteuren | `SSM Betalingen per inkoopregel` | `notifica` |
    | Historisch | `SSM Betalingen per opbrengstregel` | `notifica` |

    **Database:** `{customer_code}` op `10.3.152.9:5432`
    """)


def get_available_customers() -> list:
    """Return list of available customers from DWH."""
    # Alle klanten met een data warehouse
    return [
        "1054", "1096", "1138", "1142", "1164", "1172", "1177", "1190", "1198",
        "1209", "1210", "1211", "1212", "1214", "1217", "1222", "1224", "1231",
        "1234", "1241", "1243", "1246", "1247", "1249", "1251", "1252", "1253",
        "1255", "1256", "1257", "1258", "1263", "1264", "1265", "1267", "1268",
        "1269", "1270", "1271", "1272", "1273"
    ]


def render_customer_selector():
    """Render customer code selector in sidebar."""
    st.sidebar.header("🏢 Klant Selectie")

    # Demo mode toggle - use key to track changes
    use_mock = st.sidebar.checkbox("Demo modus (mock data)", value=False, key="demo_mode")

    customer_code = None
    if not use_mock:
        # Haal beschikbare klanten op
        customers = get_available_customers()

        # Dropdown met klanten - use key to track changes and trigger rerun
        customer_code = st.sidebar.selectbox(
            "Selecteer klant",
            options=[""] + customers,
            index=customers.index("1241") + 1 if "1241" in customers else 0,
            format_func=lambda x: "-- Kies een klant --" if x == "" else f"Klant {x}",
            help="Selecteer een klant uit de beschikbare databases",
            key="selected_customer"
        )

        if customer_code:
            st.sidebar.success(f"Verbonden met database {customer_code}")
        else:
            st.sidebar.warning("Selecteer een klant")
            customer_code = None
    else:
        st.sidebar.info("📊 Demo modus actief - fictieve data wordt getoond")

    return use_mock, customer_code


def render_data_summary(data: Dict[str, pd.DataFrame], filtered: bool = False):
    """Render samenvatting van de data met aantallen."""
    prefix = "Gefilterd: " if filtered else ""

    col1, col2, col3 = st.columns(3)

    with col1:
        n_deb = len(data["debiteuren"]) if not data["debiteuren"].empty else 0
        st.caption(f"{prefix}{n_deb} openstaande debiteuren")

    with col2:
        n_cred = len(data["crediteuren"]) if not data["crediteuren"].empty else 0
        st.caption(f"{prefix}{n_cred} openstaande crediteuren")

    with col3:
        n_bank = len(data["banksaldo"]) if not data["banksaldo"].empty else 0
        st.caption(f"{prefix}{n_bank} bankrekeningen")


def main():
    """Main application entry point."""
    # Header
    st.title("Liquiditeit Dashboard")
    st.markdown("*Real-time inzicht in uw cashflow en liquiditeitspositie*")

    # Sidebar controls - Logo linksboven
    import os
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "notifica-logo.svg")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    st.sidebar.markdown("---")

    # Customer selection
    use_mock, customer_code = render_customer_selector()

    st.sidebar.markdown("---")

    # Standdatum selector (BEFORE loading data - affects which data is fetched)
    st.sidebar.header("Peildatum")
    standdatum = st.sidebar.date_input(
        "Standdatum",
        value=datetime.now().date(),
        help="Peildatum voor openstaande posten. Toont facturen die op deze datum nog openstonden."
    )

    # Administratie selectie (voor prognose met vaste lasten)
    # Standaard administratie voor bekende klanten
    default_admins = {
        "1273": "Kronenburg Techniek B.V",
    }
    selected_admin = None
    if customer_code and customer_code in default_admins:
        selected_admin = default_admins[customer_code]
        st.sidebar.text_input(
            "Administratie",
            value=selected_admin,
            disabled=True,
            help="Administratie voor deze klant"
        )

    st.sidebar.markdown("---")

    # Toon actieve database in header
    if not use_mock and customer_code:
        st.caption(f"Database: **{customer_code}** | Standdatum: **{standdatum}**")

    # Load data first (needed for filter options)
    load_timestamp = datetime.now()
    with st.spinner(f"Data laden voor klant {customer_code} per {standdatum}..." if customer_code else "Data laden..."):
        data = load_data(
            use_mock=use_mock,
            customer_code=customer_code,
            standdatum=standdatum,
            administratie=selected_admin
        )

    # Filter controls
    filters = render_filter_sidebar(data)

    st.sidebar.markdown("---")

    # Scenario controls
    debiteur_delay, crediteur_delay, forecast_weeks, history_weeks, use_historical_profile, use_seasonality, use_ml = render_scenario_sidebar()

    # Apply filters to data
    has_active_filters = any([
        filters.get("bankrekeningen"),
        filters.get("administratie") and filters.get("administratie") != "Alle",
        filters.get("bedrijfseenheid") and filters.get("bedrijfseenheid") != "Alle",
    ])

    if has_active_filters:
        filtered_data = apply_data_filters(data, filters)
        st.info("Filters actief - data is gefilterd")
    else:
        filtered_data = data

    # Calculate metrics
    metrics = calculate_liquidity_metrics(
        filtered_data["banksaldo"],
        filtered_data["debiteuren"],
        filtered_data["crediteuren"]
    )

    # Kies de juiste prognose methode
    has_historical_data = (
        "historische_cashflow" in filtered_data
        and not filtered_data.get("historische_cashflow", pd.DataFrame()).empty
    )

    # Debug info - toon welke methode wordt gebruikt
    hist_rows = len(filtered_data.get("historische_cashflow", pd.DataFrame()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debug info:**")
    st.sidebar.caption(f"Historische data: {hist_rows} weken")
    st.sidebar.caption(f"use_ml={use_ml}, has_hist={has_historical_data}")

    # Model metrics placeholder
    model_metrics = None

    if use_ml and has_historical_data:
        # ML Ensemble model met automatische backtesting
        forecast, forecast_start_idx, model_metrics = create_ml_forecast(
            filtered_data["banksaldo"],
            filtered_data["debiteuren"],
            filtered_data["crediteuren"],
            filtered_data.get("historische_cashflow", pd.DataFrame()),
            weeks=forecast_weeks,
            weeks_history=history_weeks,
            debiteur_delay_days=debiteur_delay,
            crediteur_delay_days=crediteur_delay,
            reference_date=standdatum,
        )

        # Toon model performance in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Model Performance (backtest):**")
        if model_metrics and model_metrics.n_test_weeks > 0:
            # Color code MAPE
            mape_color = "green" if model_metrics.mape_netto < 20 else ("orange" if model_metrics.mape_netto < 40 else "red")
            st.sidebar.markdown(f"MAPE netto: **:{mape_color}[{model_metrics.mape_netto:.1f}%]**")
            st.sidebar.caption(f"MAPE in: {model_metrics.mape_inkomsten:.1f}% | uit: {model_metrics.mape_uitgaven:.1f}%")
            st.sidebar.caption(f"Bias: €{model_metrics.bias:,.0f} | Tests: {model_metrics.n_test_weeks} weken")
        else:
            st.sidebar.caption("Onvoldoende data voor backtest")

        st.sidebar.caption("Methodiek: ML Ensemble (seizoen + trend + WMA)")

    elif has_historical_data and use_historical_profile:
        # Gebruik verbeterde prognose met historisch profiel
        forecast, forecast_start_idx = create_enhanced_cashflow_forecast(
            filtered_data["banksaldo"],
            filtered_data["debiteuren"],
            filtered_data["crediteuren"],
            filtered_data.get("terugkerende_kosten", pd.DataFrame()),
            filtered_data.get("historische_cashflow", pd.DataFrame()),
            weeks=forecast_weeks,
            weeks_history=history_weeks,
            debiteur_delay_days=debiteur_delay,
            crediteur_delay_days=crediteur_delay,
            reference_date=standdatum,
            include_recurring_costs=False,
            use_seasonality=use_seasonality,
        )
        st.sidebar.caption("Methodiek: blend historisch + openstaande posten")
    else:
        # Gebruik standaard prognose (alleen openstaande posten)
        forecast, forecast_start_idx = create_transparent_cashflow_forecast(
            filtered_data["banksaldo"],
            filtered_data["debiteuren"],
            filtered_data["crediteuren"],
            filtered_data["salarissen"],
            filtered_data["historisch"],
            weeks_history=history_weeks,
            weeks_forecast=forecast_weeks,
            debiteur_delay_days=debiteur_delay,
            crediteur_delay_days=crediteur_delay,
            reference_date=standdatum,
        )
        st.sidebar.caption("Methodiek: alleen openstaande posten")

    # === TABS ===
    tab_dashboard, tab_validatie, tab_datamodel = st.tabs(["📊 Dashboard", "🔍 Validatie & Aansluiting", "🗃️ Datamodel"])

    with tab_dashboard:
        # Alerts section
        st.markdown("---")
        render_alerts(forecast, metrics)

        # KPI Cards
        st.markdown("---")
        render_kpi_cards(metrics)
        render_data_summary(filtered_data, filtered=has_active_filters)

        # Transparency legend
        render_transparency_legend()

        # Main cashflow chart with realisatie/prognose distinction
        st.markdown("---")
        cashflow_fig = render_transparent_cashflow_chart(forecast, forecast_start_idx, reference_date=standdatum)
        st.plotly_chart(cashflow_fig, key="cashflow_main_chart")

        # Scenario impact info
        if debiteur_delay > 0 or crediteur_delay > 0:
            st.info(
                f"📈 **Scenario actief:** Debiteuren +{debiteur_delay} dagen, "
                f"Crediteuren +{crediteur_delay} dagen vertraging"
            )

        # Pie charts voor compositie debiteuren/crediteuren (met filter functie)
        st.markdown("---")
        selected_deb, selected_cred = render_pie_charts_with_filter(
            filtered_data["debiteuren"],
            filtered_data["crediteuren"]
        )

        # Aging analysis - gefilterd op geselecteerde relatie
        st.markdown("---")

        # Filter data op geselecteerde relatie voor ouderdomsanalyse
        aging_deb_data = filtered_data["debiteuren"]
        aging_cred_data = filtered_data["crediteuren"]

        if selected_deb and not filtered_data["debiteuren"].empty:
            aging_deb_data = filtered_data["debiteuren"][
                filtered_data["debiteuren"]["debiteur_naam"] == selected_deb
            ]

        if selected_cred and not filtered_data["crediteuren"].empty:
            aging_cred_data = filtered_data["crediteuren"][
                filtered_data["crediteuren"]["crediteur_naam"] == selected_cred
            ]

        aging_deb = calculate_aging_buckets(aging_deb_data, reference_date=standdatum)
        aging_cred = calculate_aging_buckets(aging_cred_data, reference_date=standdatum)
        render_aging_chart(aging_deb, aging_cred, selected_deb, selected_cred)

        # Detailed table
        st.markdown("---")
        render_cashflow_details(forecast)

    with tab_validatie:
        render_validation_tab(data, customer_code or "Demo", load_timestamp)

    with tab_datamodel:
        render_datamodel_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Liquiditeit Dashboard v0.3 | Notifica - Business Intelligence voor installatiebedrijven<br>"
        "<em>Met transparantie: realisatie vs prognose onderscheid</em>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
