"""
Voorraad Dashboard - Main Application
=====================================
Streamlit app voor voorraadanalyse, min-max bewaking en magazijnbeheer.

Notifica - Business Intelligence voor installatiebedrijven

Gebaseerd op Syntess DWH voorraad tabellen:
- Voorraad magazijnen
- Voorraad locaties
- Voorraad posities
- Voorraad mutaties
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple

# Local imports
from config import AppConfig, COLORS, VOORRAAD_THRESHOLDS
from src.database import get_database, MockDatabase, LocalDatabase
from src.auth import check_password, logout
from src.calculations import (
    calculate_voorraad_metrics,
    calculate_min_max_status,
    calculate_omloopsnelheid,
    calculate_gemiddelde_ligduur,
    get_top_slow_moving,
    get_top_fast_moving,
    check_balans_aansluiting,
    validate_p_times_q,
    VoorraadMetrics,
    BalansAansluiting,
)

# Page configuration
st.set_page_config(
    page_title="Voorraad Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #16136F;
    }
    .status-ok { color: #27ae60; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-danger { color: #e74c3c; font-weight: bold; }
    .magazijn-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def render_kpi_cards(metrics: VoorraadMetrics):
    """Render KPI cards at top of dashboard."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Voorraadwaarde",
            value=f"€ {metrics.totale_waarde:,.0f}",
            help="Totale waarde op kostprijs"
        )

    with col2:
        st.metric(
            label="Stuks op Voorraad",
            value=f"{metrics.totaal_aantal:,.0f}",
            help="Totaal aantal stuks in alle magazijnen"
        )

    with col3:
        st.metric(
            label="Onder Minimum",
            value=f"{metrics.onder_minimum}",
            delta="Kritiek" if metrics.onder_minimum > 10 else None,
            delta_color="inverse" if metrics.onder_minimum > 10 else "off",
            help="Aantal posities onder minimum voorraad"
        )

    with col4:
        st.metric(
            label="Omloopsnelheid",
            value=f"{metrics.omloopsnelheid:.1f}x",
            help="Jaarlijkse omloopsnelheid (hoger = beter)"
        )

    with col5:
        st.metric(
            label="Dagen Voorraad",
            value=f"{metrics.dagen_voorraad:.0f}",
            help="Gemiddeld aantal dagen voorraad (DSI)"
        )


def render_min_max_overview(posities: pd.DataFrame):
    """Render min-max status overview."""
    st.subheader("Min-Max Bewaking")

    if posities.empty or "status" not in posities.columns:
        st.info("Geen voorraadposities beschikbaar")
        return

    # Calculate status counts
    status_counts = posities.groupby("status").size().reset_index(name="aantal")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        geen = status_counts[status_counts["status"] == "Geen voorraad"]["aantal"].sum()
        st.metric("Geen voorraad", geen, delta=None)

    with col2:
        onder = status_counts[status_counts["status"] == "Onder minimum"]["aantal"].sum()
        st.metric("Onder minimum", onder, delta="Actie vereist" if onder > 0 else None, delta_color="inverse")

    with col3:
        ok = status_counts[status_counts["status"] == "OK"]["aantal"].sum()
        st.metric("OK", ok)

    with col4:
        boven = status_counts[status_counts["status"] == "Boven maximum"]["aantal"].sum()
        st.metric("Boven maximum", boven, delta="Let op" if boven > 0 else None, delta_color="inverse")

    # Pie chart of status distribution
    fig = px.pie(
        status_counts,
        values="aantal",
        names="status",
        color="status",
        color_discrete_map={
            "Geen voorraad": COLORS["danger"],
            "Onder minimum": COLORS["warning"],
            "OK": COLORS["success"],
            "Boven maximum": COLORS["info"],
        },
        hole=0.4,
    )
    fig.update_layout(height=300, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def render_magazijn_overzicht(magazijnen: pd.DataFrame, posities: pd.DataFrame):
    """Render overview of all warehouses."""
    st.subheader("Magazijnen Overzicht")

    if magazijnen.empty:
        st.info("Geen magazijnen beschikbaar")
        return

    # Calculate waarde per magazijn from posities
    if not posities.empty and "waarde" in posities.columns and "magazijn" in posities.columns:
        waarde_per_magazijn = posities.groupby("magazijn")["waarde"].sum().reset_index()
        magazijnen = magazijnen.merge(waarde_per_magazijn, on="magazijn", how="left")
        magazijnen["waarde"] = magazijnen["waarde"].fillna(0)
    else:
        magazijnen["waarde"] = 0

    # Bar chart of stock value per warehouse
    fig = px.bar(
        magazijnen,
        x="magazijn",
        y="waarde",
        color="projectmagazijn",
        color_discrete_map={"Ja": COLORS["warning"], "Nee": COLORS["primary"]},
        labels={"waarde": "Voorraadwaarde (€)", "magazijn": "Magazijn"},
    )
    fig.update_layout(height=400, showlegend=True, legend_title="Projectmagazijn")
    st.plotly_chart(fig, use_container_width=True)


def render_kritieke_posities(posities: pd.DataFrame):
    """Render table of critical stock positions."""
    st.subheader("Kritieke Posities")

    if posities.empty or "status" not in posities.columns:
        st.info("Geen positie data beschikbaar")
        return

    # Filter critical positions
    kritiek = posities[posities["status"].isin(["Geen voorraad", "Onder minimum"])]

    if kritiek.empty:
        st.success("Geen kritieke posities gevonden!")
        return

    # Sort by urgency
    kritiek = kritiek.sort_values(["status", "tekort"], ascending=[True, False])

    # Display table
    display_cols = ["artikel", "magazijn", "locatie", "stand", "minimum", "tekort", "status"]
    available_cols = [c for c in display_cols if c in kritiek.columns]

    st.dataframe(
        kritiek[available_cols].head(50),
        hide_index=True,
        use_container_width=True,
        column_config={
            "stand": st.column_config.NumberColumn("Stand", format="%d"),
            "minimum": st.column_config.NumberColumn("Min", format="%d"),
            "tekort": st.column_config.NumberColumn("Tekort", format="%d"),
            "status": st.column_config.TextColumn("Status"),
        }
    )


def render_mutaties_trend(mutaties: pd.DataFrame):
    """Render stock movement trend chart."""
    st.subheader("Voorraadbeweging")

    if mutaties.empty:
        st.info("Geen mutaties beschikbaar")
        return

    # Aggregate by week
    mutaties["week"] = pd.to_datetime(mutaties["boekdatum"]).dt.to_period("W").dt.start_time

    weekly = mutaties.groupby("week").agg({
        "ontvangst_aantal": "sum",
        "uitgifte_aantal": "sum",
    }).reset_index()

    weekly["netto"] = weekly["ontvangst_aantal"] - weekly["uitgifte_aantal"]

    # Line chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekly["week"],
        y=weekly["ontvangst_aantal"],
        name="Ontvangsten",
        marker_color=COLORS["success"],
    ))

    fig.add_trace(go.Bar(
        x=weekly["week"],
        y=-weekly["uitgifte_aantal"],
        name="Uitgiftes",
        marker_color=COLORS["danger"],
    ))

    fig.add_trace(go.Scatter(
        x=weekly["week"],
        y=weekly["netto"].cumsum(),
        name="Cumulatief",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=COLORS["primary"], width=2),
    ))

    fig.update_layout(
        height=400,
        barmode="relative",
        yaxis=dict(title="Aantal"),
        yaxis2=dict(title="Cumulatief", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_analyse_tab(posities: pd.DataFrame, mutaties: pd.DataFrame, metrics: VoorraadMetrics):
    """Render analysis tab with turnover insights per feedback."""
    st.header("Voorraadanalyse")
    st.caption("Omloopsnelheid, ligduur en top artikelen")

    # === KPI ROW ===
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Omloopsnelheid (jaar)",
            value=f"{metrics.omloopsnelheid:.1f}x",
            help="Hoe vaak de voorraad per jaar omloopt. Hoger = beter"
        )

    with col2:
        st.metric(
            label="Gem. Ligduur",
            value=f"{metrics.dagen_voorraad:.0f} dagen",
            help="Gemiddeld aantal dagen dat artikelen op voorraad liggen"
        )

    with col3:
        st.metric(
            label="Onder Minimum",
            value=f"{metrics.onder_minimum}",
            delta="Actie vereist" if metrics.onder_minimum > 5 else None,
            delta_color="inverse",
            help="Aantal posities onder minimumvoorraad"
        )

    with col4:
        st.metric(
            label="Boven Maximum",
            value=f"{metrics.boven_maximum}",
            delta="Let op" if metrics.boven_maximum > 5 else None,
            delta_color="inverse",
            help="Aantal posities boven maximumvoorraad"
        )

    st.markdown("---")

    # === TOP 10 SECTIONS ===
    col_slow, col_fast = st.columns(2)

    with col_slow:
        st.subheader("🐢 Top 10 Langzaam Lopend")
        st.caption("Artikelen met hoogste ligduur (laagste omloopsnelheid)")

        slow_df = get_top_slow_moving(posities, mutaties, top_n=10, min_waarde=50)

        if slow_df.empty:
            st.info("Geen data beschikbaar")
        else:
            # Format for display
            display_df = slow_df[["artikel", "stand", "waarde", "omloopsnelheid", "gemiddelde_ligduur_dagen"]].copy()
            display_df.columns = ["Artikel", "Stand", "Waarde", "Omloop", "Ligduur (dagen)"]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Waarde": st.column_config.NumberColumn("Waarde", format="€ %.0f"),
                    "Stand": st.column_config.NumberColumn("Stand", format="%d"),
                    "Omloop": st.column_config.NumberColumn("Omloop", format="%.1fx"),
                    "Ligduur (dagen)": st.column_config.NumberColumn("Ligduur", format="%d"),
                }
            )

            # Warning for very slow items
            very_slow = slow_df[slow_df["gemiddelde_ligduur_dagen"] >= 365]
            if not very_slow.empty:
                st.warning(f"⚠️ {len(very_slow)} artikelen met ligduur > 1 jaar (waarde: € {very_slow['waarde'].sum():,.0f})")

    with col_fast:
        st.subheader("🐇 Top 10 Snel Lopend")
        st.caption("Artikelen met laagste ligduur (hoogste omloopsnelheid)")

        fast_df = get_top_fast_moving(posities, mutaties, top_n=10, min_waarde=50)

        if fast_df.empty:
            st.info("Geen data beschikbaar")
        else:
            display_df = fast_df[["artikel", "stand", "waarde", "omloopsnelheid", "gemiddelde_ligduur_dagen"]].copy()
            display_df.columns = ["Artikel", "Stand", "Waarde", "Omloop", "Ligduur (dagen)"]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Waarde": st.column_config.NumberColumn("Waarde", format="€ %.0f"),
                    "Stand": st.column_config.NumberColumn("Stand", format="%d"),
                    "Omloop": st.column_config.NumberColumn("Omloop", format="%.1fx"),
                    "Ligduur (dagen)": st.column_config.NumberColumn("Ligduur", format="%d"),
                }
            )

            # Info for fastest movers
            if not fast_df.empty:
                avg_days = fast_df["gemiddelde_ligduur_dagen"].mean()
                st.info(f"✅ Gemiddelde ligduur top 10: {avg_days:.0f} dagen")


def render_datakwaliteit_tab(posities: pd.DataFrame, balans_waarde: float, tarieven: pd.DataFrame):
    """Render data quality tab with balance reconciliation and P x Q validation."""
    st.header("Datakwaliteit & Aansluiting")
    st.caption("Controle van voorraadberekening en balansaansluiting")

    # === BALANSAANSLUITING ===
    st.subheader("📊 Balansaansluiting")

    aansluiting = check_balans_aansluiting(posities, balans_waarde, tolerantie_percentage=1.0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Berekende Voorraadwaarde",
            value=f"€ {aansluiting.voorraad_berekend:,.0f}",
            help="Som van alle voorraadposities (Aantal × Kostprijs)"
        )

    with col2:
        if balans_waarde and balans_waarde > 0:
            st.metric(
                label="Balanswaarde (Grootboek)",
                value=f"€ {aansluiting.voorraad_balans:,.0f}",
                help="Voorraadwaarde uit balans (grootboekrekeningen 3xxx)"
            )
        else:
            st.metric(
                label="Balanswaarde (Grootboek)",
                value="Niet beschikbaar",
                help="Geen balansdata beschikbaar"
            )

    with col3:
        if aansluiting.status == "OK":
            st.metric(
                label="Verschil",
                value=f"€ {aansluiting.verschil:,.0f}",
                delta=f"{aansluiting.verschil_percentage:.1f}%",
                delta_color="off"
            )
            st.success("✅ Aansluiting OK")
        elif aansluiting.status == "Geen balansdata":
            st.info("ℹ️ Geen balansdata beschikbaar voor vergelijking")
        elif aansluiting.status == "Kleine afwijking":
            st.metric(
                label="Verschil",
                value=f"€ {aansluiting.verschil:,.0f}",
                delta=f"{aansluiting.verschil_percentage:.1f}%",
                delta_color="inverse"
            )
            st.warning("⚠️ Kleine afwijking geconstateerd")
        else:
            st.metric(
                label="Verschil",
                value=f"€ {aansluiting.verschil:,.0f}",
                delta=f"{aansluiting.verschil_percentage:.1f}%",
                delta_color="inverse"
            )
            st.error("❌ Significante afwijking - onderzoek nodig")

    st.markdown("---")

    # === NEGATIEVE VOORRAAD ===
    st.subheader("🔴 Negatieve Voorraad")

    if aansluiting.artikelen_met_negatief > 0:
        st.warning(f"⚠️ {aansluiting.artikelen_met_negatief} posities met negatieve voorraad")

        # Show items with negative stock
        negatief = posities[posities["stand"] < 0].copy() if "stand" in posities.columns else pd.DataFrame()

        if not negatief.empty:
            display_cols = ["artikel", "magazijn", "stand", "waarde"]
            available = [c for c in display_cols if c in negatief.columns]

            st.dataframe(
                negatief[available].head(20),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "stand": st.column_config.NumberColumn("Stand", format="%d"),
                    "waarde": st.column_config.NumberColumn("Waarde", format="€ %.2f"),
                }
            )

            st.caption("""
            **Mogelijke oorzaken negatieve voorraad:**
            - Uitgiftes geboekt vóór ontvangsten
            - Historische mutaties ontbreken
            - Beginstand niet correct ingeladen
            """)
    else:
        st.success("✅ Geen posities met negatieve voorraad")

    st.markdown("---")

    # === P x Q VALIDATIE ===
    st.subheader("🧮 P × Q Validatie")
    st.caption("Controle: Waarde = Aantal × Kostprijs per stuk")

    issues = validate_p_times_q(posities, tarieven)

    if issues.empty:
        st.success("✅ Geen significante prijsafwijkingen gevonden")
    else:
        st.warning(f"⚠️ {len(issues)} posities met mogelijke prijsafwijkingen")

        display_cols = ["artikel", "stand", "waarde", "kostprijs_per_stuk"]
        if "referentie_prijs" in issues.columns:
            display_cols.append("referentie_prijs")
        if "prijs_afwijking_pct" in issues.columns:
            display_cols.append("prijs_afwijking_pct")

        available = [c for c in display_cols if c in issues.columns]

        st.dataframe(
            issues[available].head(20),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    # === UITLEG BEREKENING ===
    with st.expander("ℹ️ Hoe wordt de voorraadwaarde berekend?"):
        st.markdown("""
        **Voorraadberekening in Syntess DWH:**

        De voorraadstand en -waarde worden berekend op basis van mutaties:

        ```
        Stand = SUM(Ontvangsten) - SUM(Uitgiftes)
        Waarde = SUM(Ontvangst Kostprijs) - SUM(Uitgifte Kostprijs)
        ```

        **Belangrijke opmerkingen:**

        1. **Historische data**: Als historische mutaties ontbreken, klopt de berekende
           stand niet. Een beginstand-correctie kan nodig zijn.

        2. **Kostprijs**: De kostprijs in mutaties is de werkelijke inkoopprijs op dat
           moment, niet de actuele verrekenprijs uit tarieven.

        3. **Balansaansluiting**: De voorraadwaarde zou moeten aansluiten op de
           grootboekrekeningen voor voorraden (typisch 3xxx rekeningen).
        """)


def render_datamodel_tab():
    """Render datamodel tab showing the database structure."""
    st.header("Datamodel")
    st.caption("Overzicht van de voorraad tabellen in Syntess DWH")

    st.subheader("1. Database Structuur")

    st.markdown("""
    Dit dashboard haalt data uit de **Syntess DWH** voorraadtabellen:

    | Tabel | Beschrijving | Key |
    |-------|--------------|-----|
    | `Voorraad magazijnen` | Magazijn definities | VoorraadmagazijnKey |
    | `Voorraad locaties` | Locaties binnen magazijnen (hiërarchisch) | VoorraadlocatieKey |
    | `Voorraad posities` | Artikel + locatie combinatie met min/max | VoorraadpositieKey |
    | `Voorraad mutaties` | Alle in/uit bewegingen | VoorraadmutatieKey |
    """)

    st.subheader("2. Entiteiten & Relaties")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        VOORRAAD DATA MODEL                               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   ADMINISTRATIES     │
    └──────────┬───────────┘
               │ 1:N
               ▼
    ┌──────────────────────┐         ┌──────────────────────┐
    │    MAGAZIJNEN        │────────►│     TARIEVEN         │
    │──────────────────────│         │    (Artikelen)       │
    │ • VoorraadmagazijnKey│         └──────────┬───────────┘
    │ • Magazijn           │                    │
    │ • Projectmagazijn    │                    │ 1:N
    └──────────┬───────────┘                    ▼
               │ 1:N                 ┌──────────────────────┐
               ▼                     │     POSITIES         │
    ┌──────────────────────┐         │──────────────────────│
    │     LOCATIES         │────────►│ • VoorraadpositieKey │
    │──────────────────────│   N:1   │ • TariefKey          │
    │ • VoorraadlocatieKey │         │ • MinimumVoorraad    │
    │ • ParentKey (self)   │         │ • MaximumVoorraad    │
    └──────────────────────┘         └──────────┬───────────┘
                                                │
                                                │ 1:N (dubbel!)
                                                ▼
                                     ┌──────────────────────┐
                                     │     MUTATIES         │
                                     │──────────────────────│
                                     │ • OntvangstPositieKey│ ◄─── ACTIEF
                                     │ • UitgiftePositieKey │ ◄─── INACTIEF!
                                     │ • Aantal             │
                                     │ • Kostprijs          │
                                     │ • Boekdatum          │
                                     └──────────────────────┘
    ```
    """)

    st.warning("""
    **Kritieke relatie:** De relatie `UitgifteVoorraadpositieKey → VoorraadpositieKey`
    is **INACTIEF** in het model. Alle uitgifte-berekeningen moeten `USERELATIONSHIP()` gebruiken!
    """)

    st.subheader("3. Belangrijke Berekeningen")

    st.markdown("""
    **Voorraadstand** wordt berekend als:
    ```
    Stand = SUM(Ontvangsten) - SUM(Uitgiftes via USERELATIONSHIP)
    ```

    **Min-Max Status:**
    - `Geen voorraad`: Stand <= 0
    - `Onder minimum`: Stand < MinimumVoorraad (en Min > 0)
    - `OK`: Stand tussen Min en Max
    - `Boven maximum`: Stand > MaximumVoorraad (en Max > 0)
    """)


def get_available_customers() -> list:
    """Return list of available customers from DWH."""
    return [
        "1054", "1096", "1138", "1142", "1164", "1172", "1177", "1190", "1198",
        "1209", "1210", "1211", "1212", "1214", "1217", "1222", "1224", "1231",
        "1234", "1241", "1243", "1246", "1247", "1249", "1251", "1252", "1253",
        "1255", "1256", "1257", "1258", "1263", "1264", "1265", "1267", "1268",
        "1269", "1270", "1271", "1272", "1273"
    ]


def get_customer_from_secrets() -> tuple:
    """Get customer config from Streamlit secrets if available."""
    try:
        if hasattr(st, 'secrets'):
            if 'customer' in st.secrets:
                return st.secrets['customer'].get('code'), st.secrets['customer'].get('name')
            if 'database' in st.secrets:
                return st.secrets['database'].get('database'), None
    except Exception:
        pass
    return None, None


def is_local_data_mode() -> bool:
    """Check if app is running with local data (no database connection needed)."""
    from src.database import get_local_data_path
    local_path = get_local_data_path()
    return local_path is not None


def get_local_customer_info() -> tuple:
    """Get customer info from local data or secrets."""
    from src.database import get_local_customer_code

    # First try secrets
    secret_code, secret_name = get_customer_from_secrets()

    # Then try local data folder
    local_code, _ = get_local_customer_code()

    # Prefer secrets for name, local data for code as fallback
    customer_code = secret_code or local_code
    customer_name = secret_name

    return customer_code, customer_name


def render_customer_selector():
    """Render customer code selector in sidebar."""
    # Check if running with local data (deployed mode)
    local_mode = is_local_data_mode()

    if local_mode:
        # LOCAL DATA MODE: No customer selection - single customer only!
        customer_code, customer_name = get_local_customer_info()

        # Notifica logo
        from pathlib import Path
        logo_path = Path(__file__).parent / "assets" / "notifica_logo.jpg"
        if logo_path.exists():
            st.sidebar.image(str(logo_path), use_container_width=True)
        else:
            st.sidebar.markdown("**NOTIFICA**")

        if customer_name:
            st.sidebar.markdown(f"**{customer_name}**")

        # Logout button (only if auth is configured)
        try:
            if 'auth' in st.secrets:
                st.sidebar.markdown("---")
                if st.sidebar.button("🚪 Uitloggen", use_container_width=True):
                    logout()
        except:
            pass

        return False, customer_code

    # Check for single-customer mode via secrets (database mode)
    secret_customer, secret_name = get_customer_from_secrets()

    if secret_customer:
        # Single-customer mode with database: show customer info
        st.sidebar.header("🏢 Klant")
        if secret_name:
            st.sidebar.success(f"**{secret_name}**")
        else:
            st.sidebar.success(f"Klant {secret_customer}")

        # Still allow demo mode for testing
        use_mock = st.sidebar.checkbox("Demo modus", value=False, key="demo_mode",
                                       help="Gebruik fictieve data voor demonstratie")
        if use_mock:
            st.sidebar.info("📊 Demo modus actief")

        return use_mock, secret_customer

    # Multi-customer mode: show selector
    st.sidebar.header("🏢 Klant Selectie")

    # Demo mode toggle
    use_mock = st.sidebar.checkbox("Demo modus (mock data)", value=False, key="demo_mode")

    customer_code = None
    if not use_mock:
        customers = get_available_customers()

        customer_code = st.sidebar.selectbox(
            "Selecteer klant",
            options=[""] + customers,
            index=customers.index("1256") + 1 if "1256" in customers else 0,
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


def render_filter_sidebar(data: dict) -> dict:
    """Render filter controls in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")

    filters = {}

    # Magazijn filter
    if "magazijnen" in data and not data["magazijnen"].empty:
        mag_options = ["Alle"] + data["magazijnen"]["magazijn"].tolist()
        filters["magazijn"] = st.sidebar.selectbox(
            "Magazijn",
            mag_options,
            index=0,
        )

    # Projectmagazijn filter
    filters["excl_projectmagazijn"] = st.sidebar.checkbox(
        "Excl. projectmagazijnen",
        value=True,
        help="Sluit projectmagazijnen (OHW) uit voor zuivere voorraadcijfers"
    )

    # Status filter
    status_options = ["Alle", "Geen voorraad", "Onder minimum", "OK", "Boven maximum"]
    filters["status"] = st.sidebar.selectbox(
        "Status filter",
        status_options,
        index=0,
    )

    return filters


@st.cache_data(ttl=300)
def load_data(use_mock: bool = True, customer_code: str = None) -> Tuple[dict, datetime]:
    """Load all data from database with caching."""
    db = get_database(customer_code if not use_mock else None)

    data = {
        "magazijnen": db.get_magazijnen(),
        "locaties": db.get_locaties(),
        "posities": db.get_posities(),
        "mutaties": db.get_mutaties(),
        "tarieven": db.get_tarieven(),
        "balans_waarde": db.get_balans_voorraad(),
    }

    return data, datetime.now()


def main():
    """Main application entry point."""
    # Check authentication first
    if not check_password():
        st.stop()

    # Check mode (local vs database)
    local_mode = is_local_data_mode()

    # Get customer info based on mode
    if local_mode:
        secret_customer, secret_name = get_local_customer_info()
    else:
        secret_customer, secret_name = get_customer_from_secrets()

    # Header - show customer name in local mode
    if local_mode and secret_name:
        st.title(f"📦 Voorraad Dashboard")
        st.markdown(f"### {secret_name}")
    else:
        st.title("📦 Voorraad Dashboard")

    st.caption("Voorraadanalyse en min-max bewaking | Notifica BI")

    # Prototype disclaimer
    st.warning("⚠️ **Prototype** - Dit dashboard bevat ongevalideerde data en is bedoeld voor demonstratiedoeleinden.")

    # Customer selector in sidebar
    use_mock, customer_code = render_customer_selector()

    # Only show database info when NOT in local mode
    if not local_mode and not use_mock and customer_code:
        st.caption(f"Database: **{customer_code}** op 10.3.152.9")

    # Show data snapshot info in local mode
    if local_mode:
        from src.database import get_local_data_path, LocalDatabase
        local_path = get_local_data_path()
        if local_path:
            local_db = LocalDatabase(local_path, customer_code)
            export_date = local_db.get_export_date()
            if export_date != "Onbekend":
                try:
                    dt = datetime.fromisoformat(export_date.replace('Z', '+00:00'))
                    st.caption(f"📅 Data snapshot: {dt.strftime('%d-%m-%Y %H:%M')}")
                except:
                    pass

    # Load data
    with st.spinner("Data laden..." if local_mode else f"Data laden voor klant {customer_code}..." if customer_code else "Demo data laden..."):
        try:
            data, load_timestamp = load_data(use_mock, customer_code)
        except Exception as e:
            st.error(f"Fout bij laden data: {e}")
            st.info("Demo modus wordt gebruikt")
            data, load_timestamp = load_data(True, None)

    st.sidebar.markdown(f"*Data geladen: {load_timestamp.strftime('%H:%M:%S')}*")

    # Filter controls
    filters = render_filter_sidebar(data)

    # Apply filters (simplified - would need full implementation)
    filtered_data = data.copy()

    # Calculate metrics
    if not data["posities"].empty:
        metrics = calculate_voorraad_metrics(
            filtered_data["posities"],
            filtered_data.get("mutaties", pd.DataFrame())
        )
    else:
        metrics = VoorraadMetrics(
            totale_waarde=0,
            totaal_aantal=0,
            onder_minimum=0,
            boven_maximum=0,
            omloopsnelheid=0,
            dagen_voorraad=0,
            aantal_artikelen=0,
            aantal_magazijnen=0,
        )

    # === TABS ===
    tab_dashboard, tab_minmax, tab_analyse, tab_beweging, tab_kwaliteit, tab_datamodel = st.tabs([
        "📊 Dashboard",
        "⚠️ Min-Max Bewaking",
        "📈 Analyse",
        "📉 Beweging",
        "🔍 Datakwaliteit",
        "🗃️ Datamodel"
    ])

    with tab_dashboard:
        # KPI Cards
        st.markdown("---")
        render_kpi_cards(metrics)

        # Magazijn overview
        st.markdown("---")
        render_magazijn_overzicht(
            filtered_data.get("magazijnen", pd.DataFrame()),
            filtered_data.get("posities", pd.DataFrame())
        )

    with tab_minmax:
        render_min_max_overview(filtered_data.get("posities", pd.DataFrame()))
        st.markdown("---")
        render_kritieke_posities(filtered_data.get("posities", pd.DataFrame()))

    with tab_analyse:
        render_analyse_tab(
            filtered_data.get("posities", pd.DataFrame()),
            filtered_data.get("mutaties", pd.DataFrame()),
            metrics
        )

    with tab_beweging:
        render_mutaties_trend(filtered_data.get("mutaties", pd.DataFrame()))

    with tab_kwaliteit:
        render_datakwaliteit_tab(
            filtered_data.get("posities", pd.DataFrame()),
            filtered_data.get("balans_waarde", 0),
            filtered_data.get("tarieven", pd.DataFrame())
        )

    with tab_datamodel:
        render_datamodel_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Voorraad Dashboard v0.1 | Notifica - Business Intelligence voor installatiebedrijven"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
