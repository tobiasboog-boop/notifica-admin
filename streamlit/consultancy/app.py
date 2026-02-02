"""
Notifica Consultancy Forecast
Sturen op 50% facturabele tijd - Target vs Committed vs Gerealiseerd
"""
import streamlit as st
import pandas as pd
import altair as alt
import json
from pathlib import Path
from datetime import date

# Page config
st.set_page_config(
    page_title="Consultancy Forecast",
    page_icon="📊",
    layout="wide"
)

# Constants
PARTNERS = ["Tobias", "Arthur", "Dolf", "Mark"]
DAILY_RATE = 1150
TARGET_PER_PARTNER = 11 * DAILY_RATE  # 11 dagen × €1150 = €12.650/maand

# SharePoint Excel path - gesynchroniseerd via OneDrive
EXCEL_PATH = Path.home() / "OneDrive - Notifica B.V" / "Documenten - Sharepoint Notifica intern" / "112. Consultancy uren" / "Uren Consultancy_v2.xlsx"

# Data files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_FILE = DATA_DIR / "projects_simple.json"
NACALC_FILE = DATA_DIR / "nacalculatie.json"
REGIE_FILE = DATA_DIR / "regie_uren.json"  # Fallback lokale opslag
PIPEDRIVE_FILE = DATA_DIR / "pipedrive_klanten.xlsx"


def load_projects():
    """Laad projecten uit Excel sheet 'Projecten', met JSON als fallback"""
    # Probeer eerst uit Excel te laden
    if EXCEL_PATH.exists():
        try:
            xlsx = pd.ExcelFile(EXCEL_PATH)
            if "Projecten" in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name="Projecten")
                if len(df) > 0:
                    # Convert DataFrame to list of dicts
                    projects = df.to_dict('records')
                    # Ensure numeric columns are proper floats
                    for p in projects:
                        for col in ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]:
                            if col in p:
                                p[col] = float(p[col]) if pd.notna(p[col]) else 0
                    return projects
        except Exception as e:
            st.sidebar.warning(f"Excel laden mislukt: {e}")

    # Fallback naar JSON
    if PROJECTS_FILE.exists():
        with open(PROJECTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_projects(projects):
    """Sla projecten op naar Excel sheet 'Projecten' EN lokale JSON backup"""
    # Sla altijd lokale JSON backup op
    with open(PROJECTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)

    # Probeer naar Excel te schrijven
    if EXCEL_PATH.exists():
        try:
            # Lees bestaande Excel
            with pd.ExcelFile(EXCEL_PATH) as xlsx:
                # Lees alle bestaande sheets
                existing_sheets = {}
                for sheet_name in xlsx.sheet_names:
                    existing_sheets[sheet_name] = pd.read_excel(xlsx, sheet_name=sheet_name)

            # Maak DataFrame van projecten
            df_projects = pd.DataFrame(projects)

            # Zorg voor juiste kolomvolgorde
            cols = ["Klant", "Opdracht", "Type", "Partner", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]
            df_projects = df_projects.reindex(columns=cols)

            # Schrijf alles terug naar Excel
            with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='w') as writer:
                # Schrijf eerst alle bestaande sheets
                for sheet_name, df in existing_sheets.items():
                    if sheet_name != "Projecten":  # Skip oude Projecten sheet
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                # Schrijf Projecten sheet
                df_projects.to_excel(writer, sheet_name="Projecten", index=False)

            st.sidebar.success("✅ Opgeslagen naar Excel")
        except PermissionError:
            st.sidebar.error("❌ Excel is geopend - sluit het bestand en probeer opnieuw")
        except Exception as e:
            st.sidebar.error(f"❌ Excel opslaan mislukt: {e}")


def load_nacalculatie():
    if NACALC_FILE.exists():
        with open(NACALC_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_nacalculatie(data):
    with open(NACALC_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@st.cache_data
def load_pipedrive_klanten():
    """Laad klantenlijst uit Pipedrive export"""
    if not PIPEDRIVE_FILE.exists():
        return []

    try:
        df = pd.read_excel(PIPEDRIVE_FILE)
        # Filter op Label = "Klant"
        klanten_df = df[df['Label'] == 'Klant'][['Naam', 'Branche', 'Gewonnen deals', 'Openstaande deals']].copy()
        klanten_df = klanten_df.sort_values('Naam')
        return klanten_df.to_dict('records')
    except Exception:
        return []


@st.cache_data
def load_revenue_2025_per_client():
    """Laad gerealiseerde omzet per klant uit 2025 vanuit Excel voor segmentatie

    Returns:
        dict: {klantnaam: totale_omzet_2025}
    """
    if not EXCEL_PATH.exists():
        return {}

    try:
        xlsx = pd.ExcelFile(EXCEL_PATH)
        revenue_per_client = {}

        # Alle partners sheets parsen
        for partner, sheet_name in [("Arthur", "Uren Arthur"), ("Tobias", "Uren Tobias"), ("Mark", "Uren Mark")]:
            if sheet_name not in xlsx.sheet_names:
                continue

            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            date_col = df.columns[0]
            omzet_col = None
            klant_col = None
            jaar_col = None

            # Zoek de juiste kolommen
            for col in df.columns:
                col_str = str(col).lower()
                if 'omzet' in col_str:
                    omzet_col = col
                if 'klant' in col_str:
                    klant_col = col
                if 'jaar' in col_str:
                    jaar_col = col

            # Arthur heeft speciale structuur: kolom 4 = klant, kolom 7 = omzet
            if partner == "Arthur":
                if klant_col is None and len(df.columns) > 4:
                    klant_col = df.columns[4]
                if omzet_col is None and len(df.columns) > 7:
                    omzet_col = df.columns[7]

            # Als geen expliciete jaar kolom, zoek kolom die jaartallen bevat
            if jaar_col is None:
                for col in df.columns:
                    if isinstance(col, int) and 2020 <= col <= 2030:
                        unique_vals = df[col].dropna().unique()
                        if any(isinstance(v, (int, float)) and 2020 <= v <= 2030 for v in unique_vals[:10]):
                            jaar_col = col
                            break

            if omzet_col is None or klant_col is None:
                continue

            # Filter op 2025
            df_work = df.copy()
            if jaar_col is not None:
                df_work = df_work[df_work[jaar_col] == 2025]
            else:
                # Filter op datum jaar
                df_work['_datum'] = pd.to_datetime(df_work[date_col], errors='coerce')
                df_work = df_work.dropna(subset=['_datum'])
                df_work = df_work[df_work['_datum'].dt.year == 2025]

            if len(df_work) == 0:
                continue

            # Convert omzet to numeric
            df_work['_omzet'] = pd.to_numeric(df_work[omzet_col], errors='coerce').fillna(0)
            df_work['_klant'] = df_work[klant_col].astype(str).str.strip()

            # Aggregeer per klant
            klant_totals = df_work.groupby('_klant')['_omzet'].sum()

            for klant, omzet in klant_totals.items():
                if pd.notna(omzet) and omzet > 0 and klant and klant != 'nan':
                    # Normaliseer klantnaam
                    klant_clean = klant.strip()
                    if klant_clean in revenue_per_client:
                        revenue_per_client[klant_clean] += float(omzet)
                    else:
                        revenue_per_client[klant_clean] = float(omzet)

        return revenue_per_client
    except Exception as e:
        st.sidebar.warning(f"2025 omzet fout: {e}")
        return {}


def get_customer_segment(revenue_2025: float) -> tuple:
    """Bepaal klantsegment op basis van 2025 omzet

    Returns:
        tuple: (segment_naam, segment_emoji)
    """
    if revenue_2025 >= 10000:
        return ("A - Top", "🥇")
    elif revenue_2025 >= 5000:
        return ("B - Actief", "🥈")
    elif revenue_2025 > 0:
        return ("C - Incidenteel", "🥉")
    else:
        return ("D - Nieuw/Inactief", "⭐")


def get_pipedrive_klant_namen():
    """Haal alleen de klantnamen op voor dropdown"""
    klanten = load_pipedrive_klanten()
    return sorted([k['Naam'] for k in klanten])


def load_regie_from_excel():
    """Laad regie-uren uit de SharePoint Excel voor huidig jaar

    Excel structuur (alle partners):
    - Eerste kolom = datum
    - Jaar kolom kan 'Jaar' heten of een jaartal als naam hebben (2024)
    - Omzet kolom kan 'Omzet' heten of een Unnamed kolom zijn (index 7)
    """
    # Huidig jaar
    current_year = date.today().year

    # Debug: toon het pad in de sidebar
    st.sidebar.caption(f"Excel pad: {EXCEL_PATH}")
    st.sidebar.caption(f"Bestaat: {EXCEL_PATH.exists()}")

    if not EXCEL_PATH.exists():
        return None

    try:
        xlsx = pd.ExcelFile(EXCEL_PATH)
        regie_data = {"Tobias": {}, "Arthur": {}, "Mark": {}, "Dolf": {}}

        # Alle partners hebben dezelfde datum-gebaseerde structuur
        for partner, sheet_name in [("Arthur", "Uren Arthur"), ("Tobias", "Uren Tobias"), ("Mark", "Uren Mark")]:
            if sheet_name not in xlsx.sheet_names:
                continue

            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            date_col = df.columns[0]
            omzet_col = None
            jaar_col = None

            # Zoek de juiste kolommen
            for col in df.columns:
                col_str = str(col).lower()
                if 'omzet' in col_str:
                    omzet_col = col
                if 'jaar' in col_str:
                    jaar_col = col

            # Als geen expliciete jaar kolom, zoek kolom die jaartallen bevat (naam is een jaartal)
            if jaar_col is None:
                for col in df.columns:
                    if isinstance(col, int) and 2020 <= col <= 2030:
                        # Dit is waarschijnlijk een jaar-kolom (naam is jaartal, waarden zijn jaartallen)
                        # Check of de waarden ook jaartallen zijn
                        unique_vals = df[col].dropna().unique()
                        if any(isinstance(v, (int, float)) and 2020 <= v <= 2030 for v in unique_vals[:10]):
                            jaar_col = col
                            break

            # Als geen expliciete omzet kolom, gebruik kolom index 7 (standaard positie)
            if omzet_col is None and len(df.columns) > 7:
                omzet_col = df.columns[7]

            if omzet_col is None:
                continue

            # Filter op huidig jaar
            if jaar_col is not None:
                df_year = df[df[jaar_col] == current_year].copy()
            else:
                df_year = df.copy()

            # Parse datum
            df_year['_datum'] = pd.to_datetime(df_year[date_col], errors='coerce')
            df_year = df_year.dropna(subset=['_datum'])

            # Filter nogmaals op jaar (voor het geval jaar_col niet goed werkt)
            df_year = df_year[df_year['_datum'].dt.year == current_year]

            if len(df_year) == 0:
                continue

            df_year['_maand'] = df_year['_datum'].dt.month

            # Convert omzet to numeric (some sheets have string values like '``')
            df_year['_omzet_numeric'] = pd.to_numeric(df_year[omzet_col], errors='coerce').fillna(0)

            monthly = df_year.groupby('_maand')['_omzet_numeric'].sum()

            for month, total in monthly.items():
                if pd.notna(total) and total > 0:
                    month_key = f"{current_year}-{int(month):02d}"
                    regie_data[partner][month_key] = float(total)

        return regie_data
    except Exception as e:
        st.sidebar.error(f"Excel fout: {e}")
        return None


def get_all_klanten():
    """Haal alle klanten uit Pipedrive + eventuele extra uit projecten"""
    # Start met Pipedrive klanten
    pipedrive_klanten = set(get_pipedrive_klant_namen())

    # Voeg klanten uit bestaande projecten toe (voor het geval er nieuwe zijn)
    projects = load_projects()
    for p in projects:
        klant = p.get("Klant", "")
        if klant and klant != "Target" and klant != "Diverse":
            pipedrive_klanten.add(klant)

    return sorted(list(pipedrive_klanten))


def get_default_projects():
    """Jouw besproken projecten - zonder Arthur regie (komt uit Excel)"""
    return [
        # Klantprojecten - Afgeprijsd
        {"Klant": "Barth Groep", "Opdracht": "Migratie", "Type": "Afgeprijsd", "Partner": "Tobias, Dolf",
         "Jan": 0, "Feb": 0, "Mrt": 3750, "Apr": 3750, "Mei": 3750, "Jun": 3750, "Totaal": 15000},
        {"Klant": "Castellum", "Opdracht": "Oplevering + PowerApps", "Type": "Afgeprijsd", "Partner": "Tobias, Mark",
         "Jan": 0, "Feb": 0, "Mrt": 0, "Apr": 6667, "Mei": 6667, "Jun": 6666, "Totaal": 20000},
        {"Klant": "WVC", "Opdracht": "AI Contract Checker (pilot)", "Type": "Afgeprijsd", "Partner": "Tobias",
         "Jan": 0, "Feb": 5000, "Mrt": 0, "Apr": 0, "Mei": 0, "Jun": 0, "Totaal": 5000},
        # Klantprojecten - Regie (Unica specifiek, Arthur/Tobias/Mark regie komt uit Excel)
        {"Klant": "Unica", "Opdracht": "Regie", "Type": "Regie", "Partner": "Mark, Dolf",
         "Jan": 11500, "Feb": 11500, "Mrt": 11500, "Apr": 0, "Mei": 0, "Jun": 0, "Totaal": 34500},
        # Product targets (nog te verkopen)
        {"Klant": "Target", "Opdracht": "Liquiditeitsprognose (3×)", "Type": "Product", "Partner": "Tobias",
         "Jan": 0, "Feb": 5000, "Mrt": 5000, "Apr": 5000, "Mei": 0, "Jun": 0, "Totaal": 15000},
        {"Klant": "Target", "Opdracht": "Voorraad Analyse (3×)", "Type": "Product", "Partner": "Tobias",
         "Jan": 0, "Feb": 5000, "Mrt": 5000, "Apr": 5000, "Mei": 0, "Jun": 0, "Totaal": 15000},
        {"Klant": "Target", "Opdracht": "AI Contract Checker (2× extra)", "Type": "Product", "Partner": "Tobias",
         "Jan": 0, "Feb": 0, "Mrt": 5000, "Apr": 5000, "Mei": 0, "Jun": 0, "Totaal": 10000},
        {"Klant": "Target", "Opdracht": "BLOP Analyse (3×)", "Type": "Product", "Partner": "Tobias",
         "Jan": 0, "Feb": 5000, "Mrt": 5000, "Apr": 5000, "Mei": 0, "Jun": 0, "Totaal": 15000},
    ]


def get_regie_from_excel():
    """Haal regie omzet uit Excel en maak er projectregels van"""
    regie_data = load_regie_from_excel()
    if not regie_data:
        return []

    # Huidig jaar
    current_year = date.today().year

    # Maand mapping van YYYY-MM naar Jan, etc.
    month_map = {
        f"{current_year}-01": "Jan", f"{current_year}-02": "Feb", f"{current_year}-03": "Mrt",
        f"{current_year}-04": "Apr", f"{current_year}-05": "Mei", f"{current_year}-06": "Jun"
    }

    regie_projects = []
    for partner in ["Tobias", "Arthur", "Mark"]:
        partner_data = regie_data.get(partner, {})
        if not partner_data:
            continue

        # Bouw project regel met maandelijkse omzet
        project = {
            "Klant": "Diverse",
            "Opdracht": f"{partner} regie (Excel)",
            "Type": "Regie",
            "Partner": partner,
            "Jan": 0, "Feb": 0, "Mrt": 0, "Apr": 0, "Mei": 0, "Jun": 0
        }

        for month_key, amount in partner_data.items():
            if month_key in month_map:
                project[month_map[month_key]] = amount

        # Bereken totaal
        project["Totaal"] = sum([project[m] for m in ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun"]])

        # Alleen toevoegen als er daadwerkelijk omzet is
        if project["Totaal"] > 0:
            regie_projects.append(project)

    return regie_projects


# Load data
projects = load_projects()
if not projects:
    projects = get_default_projects()
    save_projects(projects)

nacalculatie = load_nacalculatie()

# Load regie from Excel
excel_regie = get_regie_from_excel()

# Header
st.title("📊 Consultancy Forecast")
st.caption(f"Target: 50% facturabel = €{TARGET_PER_PARTNER:,}/partner/maand | Team: €{TARGET_PER_PARTNER * 4:,}/maand")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overzicht", "🎯 Upsell", "✏️ Projecten", "⏱️ Nacalculatie"])

# =============================================================================
# TAB 1: OVERZICHT
# =============================================================================
with tab1:
    st.subheader("Committed omzet per maand")

    # Toon Excel status
    if excel_regie:
        st.success(f"✅ Regie-uren geladen uit Excel ({len(excel_regie)} partners)")
    else:
        st.warning("⚠️ Excel niet gevonden - regie-uren worden niet meegenomen. Verwacht pad: " + str(EXCEL_PATH))

    # Combineer projecten met Excel regie data
    all_projects = projects + excel_regie
    df = pd.DataFrame(all_projects)
    month_cols = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun"]
    display_cols = ["Klant", "Opdracht", "Type", "Partner"] + month_cols + ["Totaal"]

    def fmt(x):
        if isinstance(x, (int, float)) and x > 0:
            return f"€{x:,.0f}"
        elif isinstance(x, (int, float)):
            return "-"
        return x

    def highlight_totaal(col):
        """Maak Totaal kolom bold"""
        if col.name == "Totaal":
            return ["font-weight: bold"] * len(col)
        return [""] * len(col)

    styled = df[display_cols].style.format({col: fmt for col in month_cols + ["Totaal"]}).apply(highlight_totaal)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    # Calculate totals
    month_mapping = [("Jan", "Jan"), ("Feb", "Feb"), ("Mrt", "Mrt"),
                     ("Apr", "Apr"), ("Mei", "Mei"), ("Jun", "Jun")]

    monthly_committed = {}
    for month_short, month_label in month_mapping:
        monthly_committed[month_label] = round(df[month_short].sum() / 1000) * 1000  # Afgerond op duizendtallen

    total_committed = sum(monthly_committed.values())
    total_target = round(TARGET_PER_PARTNER * 4 * 6 / 1000) * 1000
    total_gap = total_target - total_committed
    pct = total_committed / total_target * 100 if total_target > 0 else 0

    # Gap Analysis Waterfall Chart
    st.markdown("---")
    st.subheader("Gap Analyse")

    # Build waterfall data
    waterfall_data = []
    running_total = 0

    # Start with Target
    waterfall_data.append({
        "label": "Target",
        "value": total_target,
        "start": 0,
        "end": total_target,
        "type": "target",
        "display": f"€{total_target/1000:.0f}k"
    })

    # Add each month's committed
    running_total = total_target
    for month_short, month_label in month_mapping:
        committed = monthly_committed[month_label]
        gap_month = round((TARGET_PER_PARTNER * 4 - df[month_short].sum()) / 1000) * 1000
        new_total = running_total - gap_month

        waterfall_data.append({
            "label": month_label,
            "value": -gap_month,
            "start": new_total,
            "end": running_total,
            "type": "gap" if gap_month > 0 else "surplus",
            "display": f"-€{gap_month/1000:.0f}k" if gap_month > 0 else f"+€{-gap_month/1000:.0f}k"
        })
        running_total = new_total

    # Final committed total
    waterfall_data.append({
        "label": "Committed",
        "value": total_committed,
        "start": 0,
        "end": total_committed,
        "type": "committed",
        "display": f"€{total_committed/1000:.0f}k"
    })

    df_waterfall = pd.DataFrame(waterfall_data)

    # Create waterfall chart
    bars = alt.Chart(df_waterfall).mark_bar(size=40).encode(
        x=alt.X("label:N",
                sort=["Target", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Committed"],
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("start:Q", title="Omzet (€)", axis=alt.Axis(format=",.0f")),
        y2="end:Q",
        color=alt.Color("type:N",
                        scale=alt.Scale(
                            domain=["target", "gap", "surplus", "committed"],
                            range=["#94a3b8", "#ef4444", "#22c55e", "#667eea"]
                        ),
                        legend=None)
    )

    # Add text labels
    text = alt.Chart(df_waterfall).mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=11,
        fontWeight="bold"
    ).encode(
        x=alt.X("label:N", sort=["Target", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Committed"]),
        y=alt.Y("end:Q"),
        text="display:N",
        color=alt.condition(
            alt.datum.type == "gap",
            alt.value("#ef4444"),
            alt.value("#374151")
        )
    )

    chart = (bars + text).properties(
        height=350
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=13
    )

    st.altair_chart(chart, use_container_width=True)

    # KPIs below chart
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Target", f"€{total_target/1000:.0f}k")
    col2.metric("✅ Committed", f"€{total_committed/1000:.0f}k")
    col3.metric("📊 Behaald", f"{pct:.0f}%")
    if total_gap > 0:
        col4.metric("⚠️ Gap", f"€{total_gap/1000:.0f}k")
    else:
        col4.metric("✅ Surplus", f"€{-total_gap/1000:.0f}k")


# =============================================================================
# TAB 2: KLANTSEGMENTATIE
# =============================================================================
with tab2:
    st.subheader("Klantsegmentatie")
    st.caption("Klanten gesegmenteerd op gerealiseerde consultancy omzet 2025")

    # Load Pipedrive klanten en 2025 omzet
    pipedrive_klanten = load_pipedrive_klanten()
    revenue_2025 = load_revenue_2025_per_client()

    if pipedrive_klanten:
        # Helper functie om 2025 omzet te matchen (fuzzy matching op klantnaam)
        def get_2025_revenue(klant_naam: str) -> float:
            """Zoek 2025 omzet voor klant met fuzzy matching"""
            klant_lower = klant_naam.lower().strip()

            # Directe match
            for excel_klant, omzet in revenue_2025.items():
                if excel_klant.lower().strip() == klant_lower:
                    return omzet

            # Partial match (Excel naam in Pipedrive naam of andersom)
            for excel_klant, omzet in revenue_2025.items():
                excel_lower = excel_klant.lower().strip()
                if excel_lower in klant_lower or klant_lower in excel_lower:
                    return omzet
                # Ook eerste woord matchen (bijv. "Barth" matcht "Barth Groep")
                if excel_lower.split()[0] == klant_lower.split()[0]:
                    return omzet

            return 0.0

        # Bouw klanttabel
        upsell_data = []
        for klant in pipedrive_klanten:
            naam = klant['Naam']

            # Bereken omzet 2026 (uit projecten)
            omzet_2026 = 0
            for p in projects + excel_regie:
                p_klant = p.get("Klant", "").lower()
                if p_klant == naam.lower() or naam.lower() in p_klant or p_klant in naam.lower():
                    omzet_2026 += p.get("Totaal", 0)

            # Haal 2025 omzet op
            omzet_2025 = get_2025_revenue(naam)

            # Bepaal segment op basis van 2025 omzet
            segment, emoji = get_customer_segment(omzet_2025)

            upsell_data.append({
                "Segment": f"{emoji} {segment}",
                "Klant": naam,
                "Branche": klant.get('Branche', '-'),
                "Omzet 2025": omzet_2025,
                "Omzet 2026": omzet_2026,
            })

        df_upsell = pd.DataFrame(upsell_data)

        # Sorteer op 2025 omzet (hoogste eerst)
        df_upsell = df_upsell.sort_values("Omzet 2025", ascending=False)

        # Stats per segment
        col1, col2, col3, col4 = st.columns(4)

        segment_a = len(df_upsell[df_upsell["Segment"].str.contains("A - Top")])
        segment_b = len(df_upsell[df_upsell["Segment"].str.contains("B - Actief")])
        segment_c = len(df_upsell[df_upsell["Segment"].str.contains("C - Incidenteel")])
        segment_d = len(df_upsell[df_upsell["Segment"].str.contains("D - Nieuw")])

        col1.metric("🥇 A - Top", f"{segment_a} klanten", help="Omzet 2025 > €10.000")
        col2.metric("🥈 B - Actief", f"{segment_b} klanten", help="Omzet 2025 €5.000-€10.000")
        col3.metric("🥉 C - Incidenteel", f"{segment_c} klanten", help="Omzet 2025 < €5.000")
        col4.metric("⭐ D - Nieuw", f"{segment_d} klanten", help="Geen consultancy omzet in 2025")

        # Filter
        st.markdown("---")
        segment_options = ["Alle segmenten"] + sorted(df_upsell["Segment"].unique().tolist())
        filter_segment = st.selectbox("Filter op segment", segment_options)

        df_filtered = df_upsell.copy()
        if filter_segment != "Alle segmenten":
            df_filtered = df_filtered[df_filtered["Segment"] == filter_segment]

        # Totalen
        total_2025 = df_filtered["Omzet 2025"].sum()
        total_2026 = df_filtered["Omzet 2026"].sum()
        st.caption(f"**Totaal omzet 2025: €{total_2025:,.0f}** | Omzet 2026: €{total_2026:,.0f}")

        # Tabel
        st.dataframe(
            df_filtered.style.format({
                "Omzet 2025": "€{:,.0f}",
                "Omzet 2026": "€{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Upsell focus
        st.markdown("---")
        st.subheader("🎯 Upsell prioriteit")
        st.caption("Klanten met omzet in 2025 maar nog geen projecten in 2026")

        priority_upsell = df_upsell[
            (df_upsell["Omzet 2025"] > 0) &
            (df_upsell["Omzet 2026"] == 0)
        ].head(10)

        if len(priority_upsell) > 0:
            st.dataframe(
                priority_upsell.style.format({
                    "Omzet 2025": "€{:,.0f}",
                    "Omzet 2026": "€{:,.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("Alle klanten met 2025 omzet hebben al 2026 projecten!")
    else:
        st.warning("Geen Pipedrive klantenlijst gevonden. Upload een export naar data/pipedrive_klanten.xlsx")


# =============================================================================
# TAB 3: PROJECTEN
# =============================================================================
with tab3:
    st.subheader("Projecten beheren")
    st.caption("Bewerk direct in de tabel. Totalen worden automatisch berekend bij opslaan.")

    # Prepare dataframe without Totaal column (we calculate it)
    df_edit = pd.DataFrame(projects)
    month_cols = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun"]

    # Remove Totaal for editing - will be calculated
    edit_cols = ["Klant", "Opdracht", "Type", "Partner"] + month_cols
    df_for_edit = df_edit[edit_cols].copy()

    edited_df = st.data_editor(
        df_for_edit,
        column_config={
            "Klant": st.column_config.TextColumn("Klant", width="medium"),
            "Opdracht": st.column_config.TextColumn("Opdracht", width="large"),
            "Type": st.column_config.SelectboxColumn("Type", options=["Regie", "Afgeprijsd", "Product"], width="small"),
            "Partner": st.column_config.TextColumn("Partner", width="medium"),
            "Jan": st.column_config.NumberColumn("Jan", format="€%.0f", min_value=0, step=500),
            "Feb": st.column_config.NumberColumn("Feb", format="€%.0f", min_value=0, step=500),
            "Mrt": st.column_config.NumberColumn("Mrt", format="€%.0f", min_value=0, step=500),
            "Apr": st.column_config.NumberColumn("Apr", format="€%.0f", min_value=0, step=500),
            "Mei": st.column_config.NumberColumn("Mei", format="€%.0f", min_value=0, step=500),
            "Jun": st.column_config.NumberColumn("Jun", format="€%.0f", min_value=0, step=500),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic"
    )

    # Calculate and show totals live
    edited_df["Totaal"] = edited_df[month_cols].sum(axis=1)
    total_all = edited_df["Totaal"].sum()

    st.caption(f"**Totaal alle projecten: €{total_all:,.0f}**")

    if st.button("💾 Opslaan", type="primary"):
        projects = edited_df.to_dict('records')
        save_projects(projects)
        st.success("✅ Opgeslagen!")
        st.rerun()

    # Quick add form
    st.markdown("---")
    st.subheader("➕ Snel toevoegen")

    # Haal alle klanten op (Pipedrive + projecten)
    all_klanten = get_all_klanten()

    with st.form("quick_add"):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Klant dropdown met alle 60+ Pipedrive klanten
            klant_options = ["-- Selecteer klant --"] + all_klanten + ["-- Nieuwe klant --"]
            klant_select = st.selectbox("Klant", klant_options)
            new_klant_input = st.text_input("Of nieuwe klant (typ hier)")
            new_opdracht = st.text_input("Opdracht")
        with col2:
            new_type = st.selectbox("Type", ["Afgeprijsd", "Regie", "Product"])
            new_partner = st.selectbox("Partner", PARTNERS + ["Tobias, Mark", "Tobias, Dolf", "Mark, Dolf"])
        with col3:
            new_value = st.number_input("Totale waarde (€)", min_value=0, step=1000, value=5000)
            new_months = st.multiselect("Maanden", ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun"])

        if st.form_submit_button("Toevoegen"):
            # Bepaal de klantnaam: nieuwe klant heeft prioriteit, anders selectie
            new_klant = new_klant_input.strip() if new_klant_input.strip() else (
                klant_select if klant_select not in ["-- Selecteer klant --", "-- Nieuwe klant --"] else ""
            )

            if new_klant and new_opdracht and new_months:
                per_month = new_value / len(new_months)
                new_project = {
                    "Klant": new_klant,
                    "Opdracht": new_opdracht,
                    "Type": new_type,
                    "Partner": new_partner,
                    "Jan": per_month if "Jan" in new_months else 0,
                    "Feb": per_month if "Feb" in new_months else 0,
                    "Mrt": per_month if "Mrt" in new_months else 0,
                    "Apr": per_month if "Apr" in new_months else 0,
                    "Mei": per_month if "Mei" in new_months else 0,
                    "Jun": per_month if "Jun" in new_months else 0,
                    "Totaal": new_value
                }
                projects.append(new_project)
                save_projects(projects)
                st.success(f"✅ '{new_klant} - {new_opdracht}' toegevoegd!")
                st.rerun()


# =============================================================================
# TAB 4: NACALCULATIE - Uren voor afgeprijsde opdrachten
# =============================================================================
with tab4:
    st.subheader("Nacalculatie - Uren op afgeprijsde opdrachten")
    st.caption("Registreer uren voor vaste-prijs projecten (niet voor facturatie, wel voor inzicht)")

    # Get afgeprijsde projecten
    afgeprijsd_projecten = [p for p in projects if p["Type"] == "Afgeprijsd"]

    if not afgeprijsd_projecten:
        st.info("Geen afgeprijsde projecten gevonden")
    else:
        # Entry form
        with st.form("nacalc_entry"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                nc_date = st.date_input("Datum", value=date.today())
            with col2:
                nc_partner = st.selectbox("Partner", PARTNERS)
            with col3:
                project_options = [f"{p['Klant']} - {p['Opdracht']}" for p in afgeprijsd_projecten]
                nc_project = st.selectbox("Project", project_options)
            with col4:
                nc_hours = st.number_input("Uren", min_value=0.5, max_value=12.0, step=0.5, value=8.0)

            nc_desc = st.text_input("Omschrijving (optioneel)")

            if st.form_submit_button("Registreren", type="primary"):
                nacalculatie.append({
                    "Datum": nc_date.strftime("%Y-%m-%d"),
                    "Partner": nc_partner,
                    "Project": nc_project,
                    "Uren": nc_hours,
                    "Omschrijving": nc_desc
                })
                save_nacalculatie(nacalculatie)
                st.success(f"✅ {nc_hours} uur geregistreerd")
                st.rerun()

        # Show nacalculatie per project
        st.markdown("---")

        if nacalculatie:
            df_nc = pd.DataFrame(nacalculatie)

            # Summary per project
            st.subheader("Overzicht per project")

            for proj in afgeprijsd_projecten:
                proj_name = f"{proj['Klant']} - {proj['Opdracht']}"
                proj_uren = df_nc[df_nc["Project"] == proj_name]["Uren"].sum() if not df_nc.empty else 0
                proj_dagen = proj_uren / 8
                proj_kosten = proj_dagen * DAILY_RATE
                proj_waarde = proj["Totaal"]
                proj_marge = proj_waarde - proj_kosten

                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f"**{proj_name}**")
                col2.metric("Uren", f"{proj_uren}u ({proj_dagen:.1f}d)")
                col3.metric("Kosten", f"€{proj_kosten:,.0f}")

                if proj_marge >= 0:
                    col4.metric("Marge", f"€{proj_marge:,.0f}", delta=f"{proj_marge/proj_waarde*100:.0f}%" if proj_waarde > 0 else "")
                else:
                    col4.metric("Marge", f"€{proj_marge:,.0f}", delta=f"{proj_marge/proj_waarde*100:.0f}%", delta_color="inverse")

                st.markdown("---")

            # Detail tabel
            with st.expander("📋 Alle uren entries"):
                st.dataframe(df_nc, use_container_width=True, hide_index=True)

                if st.button("🗑️ Laatste entry verwijderen"):
                    nacalculatie.pop()
                    save_nacalculatie(nacalculatie)
                    st.rerun()
        else:
            st.info("Nog geen uren geregistreerd voor nacalculatie")


# Sidebar
with st.sidebar:
    st.markdown("### 📊 Consultancy Forecast")
    st.markdown("---")
    st.markdown(f"""
    **Targets:**
    - Dagtarief: €{DAILY_RATE:,}
    - 50% = 11 dagen/mnd
    - Per partner: €{TARGET_PER_PARTNER:,}
    - Team: €{TARGET_PER_PARTNER * 4:,}
    """)

    st.markdown("---")
    st.markdown("**Legenda Types:**")
    st.markdown("""
    - **Afgeprijsd**: Vaste prijs
    - **Regie**: Uren × tarief
    - **Product**: Nog te verkopen
    """)

    st.markdown("---")
    if st.button("🔄 Reset projecten"):
        projects = get_default_projects()
        save_projects(projects)
        st.rerun()

    st.markdown("---")
    st.caption("Notifica 2025")
