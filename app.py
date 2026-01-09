"""
Notifica Customer Health Analytics
===================================
Een Streamlit dashboard voor het analyseren van Power BI rapportgebruik per klant.

Start met: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import os
import json
from dotenv import load_dotenv

# Laad .env bestand
load_dotenv()

# Anthropic API voor AI Assistent
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Notifica Customer Health",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .green-card { background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 5px 0; }
    .orange-card { background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 5px 0; }
    .red-card { background-color: #f8d7da; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545; margin: 5px 0; }
    .contact-card { background-color: #e7f3ff; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #0066cc; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
    a { color: #0066cc; }
</style>
""", unsafe_allow_html=True)

# === CONFIGURATIE ===
INTERNE_MEDEWERKERS = [
    'Arthur Gartz', 'Arthur Gartz - Notifica (GU)',
    'Tobias Boog - Notifica (GU)', 'Tobias boog',
    'Mark Haring - Notifica (GU)', 'mcharing@bisqq.nl',
    'Domien Parren', 'William de Rooij', 'Terence Sarmo',
    'Bas Vrielink', 'Bram Jansonius', 'Damion Verheij',
]

# De 13 officiële Notifica rapportclusters (exact zoals gedefinieerd)
ALLE_RAPPORTCLUSTERS = [
    'Financieel Overzicht',
    'Projectwaardering',
    'Bewaking productiviteit',
    'Projectbewaking',
    'Offerte en Sales proces',
    'Resource Management',
    'Liquiditeitsoverzicht',
    'Werkkapitaal (debiteuren, voorraad)',
    'Bewaking rendement S&O',
    'Onderhoudsplanning',
    'Openstaande Werkbonnen',
    'S&O Uitvoering',
    'Directierapport'
]

SCORE_THRESHOLDS = {
    # Functionarissen drempels
    'groen_func': 5,  # Minimaal 5 functionarissen voor groen
    'oranje_func': 2,  # Minimaal 2 functionarissen voor oranje
    # Views per maand drempels (gemiddeld)
    'groen_views': 50,  # Minimaal 50 views/maand voor groen
    'oranje_views': 15  # Minimaal 15 views/maand voor oranje
}

# === RAPPORT CATEGORISATIE REGELS ===
# Mapping van zoektermen naar de 13 officiële Notifica rapportclusters
# LET OP: De clusternamen moeten EXACT overeenkomen met ALLE_RAPPORTCLUSTERS
# Zoektermen zijn case-insensitive en matchen op substring
RAPPORT_CATEGORIE_REGELS = {
    'Financieel Overzicht': [
        'forecast resultaatrekening', 'resultatenrekening', 'financieel kpi',
        'dashboard financieel', 'kpi dashboard finance', 'grootboek',
        'salarissen', 'omzetrapportage', 'omzet', 'financieel', 'finance',
        'begroting', 'balans', 'winst', 'verlies', 'kosten', 'budget',
        'resultaat', 'exploitatie', 'kostenplaats', 'kostencategorie',
        'administratie', 'boekhouding', 'jaarrekening', 'kwartaal',
        'maandafsluiting', 'periodesluiting', 'consolidatie'
    ],
    'Projectwaardering': [
        'projectcontrol_waardering', 'nacalculatie', 'waardering',
        'projectwaardering', 'project waardering', 'waarde project'
    ],
    'Bewaking productiviteit': [
        'productiviteit', 'urenrapportage', 'urenoverzicht', 'kpi dashboard hr',
        'personeel', 'verlof', 'ziekte', 'uren', 'hr dashboard',
        'medewerker', 'bezetting', 'capaciteit uren', 'tijdregistratie',
        'declarabele uren', 'declarabiliteit', 'niet-declarabel',
        'verzuim', 'absentie', 'inzet', 'werktijd', 'rooster',
        'planning medewerker', 'functionaris', 'fte'
    ],
    'Projectbewaking': [
        'projectbeheer', 'projectcontrol', 'forecast project', 'ohw', 'onderhanden werk',
        'klimaatbalans', 'kpi dashboard project', 'rekenhulp projectbudget', 'meerwerken',
        'projecten ohw', 'projectoverzicht', 'projectstatus', 'projectvoortgang',
        'project dashboard', 'voortgang', 'meerwerk', 'minderwerk',
        'projectadministratie', 'projectanalyse', 'project analyse',
        'projectrapport', 'projectenlijst', 'projectportefeuille',
        'oplevering', 'milestone', 'fase', 'deelproject'
    ],
    'Offerte en Sales proces': [
        'kpi sales', 'sales', 'salesrapportage', 'offerte',
        'verkoop', 'acquisitie', 'lead', 'pipeline', 'funnel',
        'orderintake', 'order intake', 'opdracht', 'contract',
        'tender', 'aanbesteding', 'propositie', 'commercieel',
        'hitrate', 'hit rate', 'conversie', 'won', 'verloren'
    ],
    'Resource Management': [
        'resource management', 'accountmanagement', 'resource',
        'resourceplanning', 'capaciteitsplanning', 'capaciteit',
        'planning', 'bezettingsgraad', 'beschikbaarheid',
        'skills', 'competenties', 'toewijzing', 'allocatie'
    ],
    'Liquiditeitsoverzicht': [
        'liquiditeit', 'forecast cash', 'cash flow', 'cashflow',
        'cash', 'kasstroom', 'treasury', 'banking', 'bank',
        'rekening courant', 'krediet', 'financiering'
    ],
    'Werkkapitaal (debiteuren, voorraad)': [
        'openstaande posten', 'werkkapitaal', 'debiteuren', 'voorraad',
        'crediteuren', 'facturen', 'factuur', 'betalingen', 'incasso',
        'ouderdom', 'aging', 'openstaand', 'magazijn', 'stock',
        'inventaris', 'voorraadbeheer', 'inkooporder'
    ],
    'Bewaking rendement S&O': [
        'rendement s&o', 's&o rendement', 'rendement service',
        'service rendement', 'onderhoud rendement', 's&o kpi'
    ],
    'Onderhoudsplanning': [
        'onderhoudsplanning', 'onderhoud planning', 'maintenance',
        'preventief onderhoud', 'gepland onderhoud', 'onderhoudsschema',
        'onderhoudscontract', 'onderhoudstaak'
    ],
    'Openstaande Werkbonnen': [
        'werkbon', 'openstaande werkbonnen', 'werkbonnen',
        'werkorder', 'work order', 'serviceorder', 'service order',
        'werkopdracht', 'taak', 'ticket'
    ],
    'S&O Uitvoering': [
        'service uitvoering', 'servicedesk', 'maandagochtend service',
        'service analyse', 'service beheer', 'storingen', 'service corporaties',
        'service', 's&o', 'onderhoud', 'installatie', 'monteur',
        'technisch', 'reparatie', 'storing', 'melding', 'klacht',
        'response', 'sla', 'first time fix', 'ftf'
    ],
    'Directierapport': [
        'directierapport', 'dashboard mt', 'dashboard directie',
        'directie', 'management', 'bestuur', 'executive',
        'kpi overzicht', 'stuurinformatie', 'cockpit'
    ]
}

# === HELPER FUNCTIES ===

def categorize_report(name):
    """Categoriseer rapport in een van de 12 officiële Notifica rapportclusters"""
    if pd.isna(name):
        return 'Niet geclassificeerd'
    name_lower = str(name).lower()

    # Speciale gevallen eerst checken
    # Projectcontrol_waardering gaat naar Projectwaardering (niet Projectbewaking)
    if 'projectcontrol_waardering' in name_lower:
        return 'Projectwaardering'

    for cluster, zoektermen in RAPPORT_CATEGORIE_REGELS.items():
        for term in zoektermen:
            if term in name_lower:
                return cluster

    return 'Niet geclassificeerd'


def get_report_categorization(df):
    """Genereer een overzicht van alle rapporten en hun categorisatie"""
    reports = df['Report name'].dropna().unique()
    categorization = []

    for report in reports:
        cat = categorize_report(report)
        # Zoek welke zoekterm matchte
        matched_term = '-'
        if cat != 'Niet geclassificeerd':
            name_lower = str(report).lower()
            for term in RAPPORT_CATEGORIE_REGELS.get(cat, []):
                if term in name_lower:
                    matched_term = term
                    break

        categorization.append({
            'Rapport': report,
            'Rapportcluster': cat,
            'Matched op': matched_term
        })

    return pd.DataFrame(categorization).sort_values(['Rapportcluster', 'Rapport'])


def process_contacts(contacts_df, orgs_df):
    """Verwerk contactpersonen en koppel aan organisaties"""
    contacts = contacts_df.copy()
    contacts.columns = [c.replace('Contactpersoon - ', '') for c in contacts.columns]

    # Combineer email velden
    contacts['Email'] = contacts['E-mail - Werk'].fillna(contacts['E-mail - Privé']).fillna(contacts['E-mail - Anders'])
    contacts['Email'] = contacts['Email'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else None)

    # Koppel aan organisatie klantnummer
    orgs = orgs_df[['Organisatie - Klantnummer', 'Organisatie - Naam']].copy()
    orgs.columns = ['Klant_Code', 'Org_Naam']
    orgs['Klant_Code'] = orgs['Klant_Code'].astype(str)

    contacts = contacts.merge(orgs, left_on='Organisatie', right_on='Org_Naam', how='left')
    return contacts


def get_contacts_for_customer(contacts_df, klantnaam):
    """Haal contactpersonen op voor een specifieke klant"""
    if contacts_df is None or contacts_df.empty:
        return pd.DataFrame()

    matches = contacts_df[
        contacts_df['Organisatie'].str.contains(klantnaam.split()[0], case=False, na=False) |
        contacts_df['Org_Naam'].str.contains(klantnaam.split()[0], case=False, na=False)
    ]

    if matches.empty and len(klantnaam) > 3:
        matches = contacts_df[
            contacts_df['Organisatie'].str.contains(klantnaam[:5], case=False, na=False)
        ]

    return matches[['Naam', 'Email', 'Labels', 'Organisatie']].drop_duplicates()


def calculate_customer_scores(df, klantnamen_df, groen_threshold, oranje_threshold):
    """Bereken health scores per klant - gebaseerd op aantal functionarissen"""
    # Filter interne medewerkers
    df_extern = df[~df['DisplayName'].isin(INTERNE_MEDEWERKERS)].copy()

    # Basis stats per klant
    klant_stats = df_extern.groupby('Klant_Code').agg({
        'Aantal activity reportviews': 'sum',
        'DisplayName': 'nunique',
        'Report name': 'nunique'
    }).reset_index()
    klant_stats.columns = ['Klant_Code', 'Views', 'Functionarissen', 'Rapporten']

    # Views afronden op hele getallen
    klant_stats['Views'] = klant_stats['Views'].round(0).astype(int)

    # Rapportgroepen per klant
    df_extern['Categorie'] = df_extern['Report name'].apply(categorize_report)

    # Welke groepen worden gebruikt per klant
    groepen_per_klant = df_extern.groupby('Klant_Code')['Categorie'].apply(
        lambda x: list(x[x != 'Niet geclassificeerd'].unique())
    )
    klant_stats['Gebruikte_Groepen'] = klant_stats['Klant_Code'].map(groepen_per_klant)
    klant_stats['Aantal_Groepen'] = klant_stats['Gebruikte_Groepen'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Ontbrekende groepen
    klant_stats['Ontbrekende_Groepen'] = klant_stats['Gebruikte_Groepen'].apply(
        lambda x: [g for g in ALLE_RAPPORTCLUSTERS if g not in (x if isinstance(x, list) else [])]
    )

    # Kleur bepalen op basis van BEIDE functionarissen EN views
    # Een klant is GROEN als beide criteria voldoen, ROOD als beide slecht zijn
    def get_color(row):
        func = row['Functionarissen']
        views = row['Views']

        # Beide criteria moeten voldoen voor groen
        func_groen = func >= groen_threshold
        views_groen = views >= SCORE_THRESHOLDS['groen_views']

        func_oranje = func >= oranje_threshold
        views_oranje = views >= SCORE_THRESHOLDS['oranje_views']

        if func_groen and views_groen:
            return 'GROEN'
        elif func_oranje and views_oranje:
            return 'ORANJE'
        return 'ROOD'

    klant_stats['Kleur'] = klant_stats.apply(get_color, axis=1)

    # Merge met klantnamen
    klant_stats = klant_stats.merge(klantnamen_df, on='Klant_Code', how='left')

    return klant_stats.sort_values('Functionarissen', ascending=False)


def get_category_breakdown(df, klant_code):
    """Haal categorie breakdown op voor specifieke klant"""
    df_extern = df[~df['DisplayName'].isin(INTERNE_MEDEWERKERS)].copy()
    df_klant = df_extern[df_extern['Klant_Code'] == klant_code].copy()
    df_klant['Categorie'] = df_klant['Report name'].apply(categorize_report)

    breakdown = df_klant.groupby('Categorie').agg({
        'Aantal activity reportviews': 'sum',
        'DisplayName': 'nunique'
    }).reset_index()
    breakdown.columns = ['Categorie', 'Views', 'Functionarissen']
    breakdown['Views'] = breakdown['Views'].round(0).astype(int)
    return breakdown.sort_values('Views', ascending=False)


def get_user_breakdown(df, klant_code):
    """Haal gebruikers breakdown op voor specifieke klant"""
    df_extern = df[~df['DisplayName'].isin(INTERNE_MEDEWERKERS)].copy()
    df_klant = df_extern[df_extern['Klant_Code'] == klant_code]

    users = df_klant.groupby('DisplayName').agg({
        'Aantal activity reportviews': 'sum',
        'Report name': 'nunique'
    }).reset_index()
    users.columns = ['Naam', 'Views', 'Rapporten']
    users['Views'] = users['Views'].round(0).astype(int)
    return users.sort_values('Views', ascending=False)


def get_trend_data(df, klant_code):
    """Haal maandelijkse trend op voor specifieke klant"""
    df_extern = df[~df['DisplayName'].isin(INTERNE_MEDEWERKERS)].copy()
    df_klant = df_extern[df_extern['Klant_Code'] == klant_code].copy()

    # Maak een maand-sorteerbare kolom
    maand_order = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
        'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'december': 12
    }

    if 'Maand' in df_klant.columns and 'Jaar' in df_klant.columns:
        df_klant['Maand_Num'] = df_klant['Maand'].str.lower().map(maand_order)
        df_klant['Jaar'] = df_klant['Jaar'].fillna(0).astype(int)
        df_klant['Periode'] = df_klant['Jaar'].astype(str) + '-' + df_klant['Maand_Num'].fillna(0).astype(int).astype(str).str.zfill(2)

        trend = df_klant.groupby(['Periode', 'Jaar', 'Maand']).agg({
            'Aantal activity reportviews': 'sum',
            'DisplayName': 'nunique'
        }).reset_index()
        trend.columns = ['Periode', 'Jaar', 'Maand', 'Views', 'Functionarissen']
        trend['Views'] = trend['Views'].round(0).astype(int)
        trend = trend.sort_values('Periode')

        # Maak leesbare labels
        trend['Label'] = trend['Maand'].str[:3] + ' ' + trend['Jaar'].astype(str).str[-2:]

        return trend

    return pd.DataFrame()


def get_overall_trend(df):
    """Haal totale maandelijkse trend op over alle klanten"""
    df_extern = df[~df['DisplayName'].isin(INTERNE_MEDEWERKERS)].copy()

    maand_order = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
        'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'december': 12
    }

    if 'Maand' in df_extern.columns and 'Jaar' in df_extern.columns:
        df_extern['Maand_Num'] = df_extern['Maand'].str.lower().map(maand_order)
        df_extern['Jaar'] = df_extern['Jaar'].fillna(0).astype(int)
        df_extern['Periode'] = df_extern['Jaar'].astype(str) + '-' + df_extern['Maand_Num'].fillna(0).astype(int).astype(str).str.zfill(2)

        trend = df_extern.groupby(['Periode', 'Jaar', 'Maand']).agg({
            'Aantal activity reportviews': 'sum',
            'DisplayName': 'nunique',
            'Klant_Code': 'nunique'
        }).reset_index()
        trend.columns = ['Periode', 'Jaar', 'Maand', 'Views', 'Functionarissen', 'Klanten']
        trend['Views'] = trend['Views'].round(0).astype(int)
        trend = trend.sort_values('Periode')

        # Maak leesbare labels
        trend['Label'] = trend['Maand'].str[:3] + ' ' + trend['Jaar'].astype(str).str[-2:]

        return trend

    return pd.DataFrame()


# === STANDAARD DATA BESTANDEN ===
# Deze bestanden worden automatisch geladen als ze bestaan
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_PBI_FILE = os.path.join(DATA_DIR, 'powerbi_activity.xlsx')
DEFAULT_ORGS_FILE = os.path.join(DATA_DIR, 'organizations.xlsx')
DEFAULT_PEOPLE_FILE = os.path.join(DATA_DIR, 'people.xlsx')


# === MAIN APP ===

def main():
    st.title("📊 Notifica Customer Health Analytics")
    st.markdown("---")

    # Sidebar - File uploads
    with st.sidebar:
        st.header("📁 Data Import")

        # Check of standaard bestanden bestaan
        default_pbi_exists = os.path.exists(DEFAULT_PBI_FILE)
        default_orgs_exists = os.path.exists(DEFAULT_ORGS_FILE)
        default_people_exists = os.path.exists(DEFAULT_PEOPLE_FILE)

        if default_pbi_exists and default_orgs_exists:
            st.warning("⚠️ **Let op:** Standaard data geladen. Controleer of dit de laatste versies zijn!")

            # Toon welke bestanden geladen zijn
            with st.expander("📂 Geladen bestanden"):
                if default_pbi_exists:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(DEFAULT_PBI_FILE))
                    st.markdown(f"**Power BI:** `powerbi_activity.xlsx`\n\n*Laatste update: {mod_time.strftime('%d-%m-%Y %H:%M')}*")
                if default_orgs_exists:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(DEFAULT_ORGS_FILE))
                    st.markdown(f"**Organizations:** `organizations.xlsx`\n\n*Laatste update: {mod_time.strftime('%d-%m-%Y %H:%M')}*")
                if default_people_exists:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(DEFAULT_PEOPLE_FILE))
                    st.markdown(f"**People:** `people.xlsx`\n\n*Laatste update: {mod_time.strftime('%d-%m-%Y %H:%M')}*")

            st.markdown("---")
            st.markdown("**Nieuwe versies uploaden (optioneel):**")

        pbi_file = st.file_uploader(
            "Power BI Activity Export (.xlsx)",
            type=['xlsx'],
            help="Upload een nieuwere versie van de Power BI report views export"
        )

        pipedrive_orgs_file = st.file_uploader(
            "Pipedrive Organizations (.xlsx)",
            type=['xlsx'],
            help="Upload een nieuwere versie van de Pipedrive klanten export"
        )

        pipedrive_people_file = st.file_uploader(
            "Pipedrive People/Contacts (.xlsx)",
            type=['xlsx'],
            help="Upload een nieuwere versie van de Pipedrive contactpersonen export"
        )

        st.markdown("---")
        st.header("⚙️ Instellingen")

        st.markdown("""
        **Status wordt bepaald door combinatie van functionarissen EN views.**
        Beide criteria moeten voldoen voor de status.
        """)

        st.subheader("👥 Functionarissen drempels")
        groen_threshold = st.slider(
            "GROEN: minimaal functionarissen",
            1, 15,
            SCORE_THRESHOLDS['groen_func'],
            help="Klanten met >= dit aantal functionarissen kunnen GROEN worden"
        )
        oranje_threshold = st.slider(
            "ORANJE: minimaal functionarissen",
            1, 10,
            SCORE_THRESHOLDS['oranje_func'],
            help="Klanten met >= dit aantal functionarissen kunnen ORANJE worden"
        )

        st.subheader("👁️ Views drempels")
        groen_views = st.slider(
            "GROEN: minimaal views",
            10, 200,
            SCORE_THRESHOLDS['groen_views'],
            help="Klanten met >= dit aantal views kunnen GROEN worden"
        )
        oranje_views = st.slider(
            "ORANJE: minimaal views",
            5, 100,
            SCORE_THRESHOLDS['oranje_views'],
            help="Klanten met >= dit aantal views kunnen ORANJE worden"
        )

        # Update de thresholds voor de views
        SCORE_THRESHOLDS['groen_views'] = groen_views
        SCORE_THRESHOLDS['oranje_views'] = oranje_views

        st.caption("💡 Verwachte verdeling: ~70% groen, ~25% oranje, ~5% rood")

    # Bepaal welke bestanden te gebruiken (uploaded > default)
    use_pbi = pbi_file if pbi_file else (DEFAULT_PBI_FILE if default_pbi_exists else None)
    use_orgs = pipedrive_orgs_file if pipedrive_orgs_file else (DEFAULT_ORGS_FILE if default_orgs_exists else None)
    use_people = pipedrive_people_file if pipedrive_people_file else (DEFAULT_PEOPLE_FILE if default_people_exists else None)

    # Check if files are available
    if use_pbi is None or use_orgs is None:
        st.info("👆 Upload de Excel bestanden in de sidebar om te beginnen, of plaats standaard bestanden in de `data/` folder.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Power BI Export**")
            st.markdown("Report views activity data")
            st.markdown(f"*Standaard: {'✅' if default_pbi_exists else '❌'}*")
        with col2:
            st.markdown("**2. Pipedrive Organizations**")
            st.markdown("Klanten met klantnummer")
            st.markdown(f"*Standaard: {'✅' if default_orgs_exists else '❌'}*")
        with col3:
            st.markdown("**3. Pipedrive People** (optioneel)")
            st.markdown("Contactpersonen met email")
            st.markdown(f"*Standaard: {'✅' if default_people_exists else '❌'}*")

        return

    # Load data
    try:
        pbi_df = pd.read_excel(use_pbi)
        orgs_df = pd.read_excel(use_orgs)

        contacts_df = None
        if use_people is not None:
            contacts_df = pd.read_excel(use_people)
            contacts_df = process_contacts(contacts_df, orgs_df)
            st.sidebar.success(f"✅ {len(contacts_df)} contactpersonen geladen")

    except Exception as e:
        st.error(f"Fout bij laden van bestanden: {e}")
        return

    # Process data
    pbi_df['Klant_Code'] = pbi_df['Workspace name'].str.extract(r'^(\d{4})')
    pbi_df = pbi_df[pbi_df['Klant_Code'].notna()].copy()

    klantnamen = orgs_df[['Organisatie - Klantnummer', 'Organisatie - Naam']].copy()
    klantnamen.columns = ['Klant_Code', 'Klantnaam']
    klantnamen['Klant_Code'] = klantnamen['Klant_Code'].astype(str)

    # Calculate scores
    scores_df = calculate_customer_scores(pbi_df, klantnamen, groen_threshold, oranje_threshold)

    # === TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚦 Overzicht",
        "📋 Scorecard",
        "🔍 Klant Detail",
        "📧 Contactpersonen",
        "📑 Rapport Mapping",
        "📨 Upsell Emails",
        "🤖 AI Assistent"
    ])

    # === TAB 1: OVERVIEW ===
    with tab1:
        st.header("🚦 Overzicht")

        col1, col2, col3, col4 = st.columns(4)

        groen_count = len(scores_df[scores_df['Kleur'] == 'GROEN'])
        oranje_count = len(scores_df[scores_df['Kleur'] == 'ORANJE'])
        rood_count = len(scores_df[scores_df['Kleur'] == 'ROOD'])
        total_count = len(scores_df)

        with col1:
            st.metric("Totaal Klanten", total_count)
        with col2:
            st.metric("🟢 Groen", groen_count, f"{groen_count/total_count*100:.0f}%")
        with col3:
            st.metric("🟠 Oranje", oranje_count, f"{oranje_count/total_count*100:.0f}%")
        with col4:
            st.metric("🔴 Rood", rood_count, f"{rood_count/total_count*100:.0f}%")

        # Trend grafiek
        st.subheader("📈 Trend over tijd")
        overall_trend = get_overall_trend(pbi_df)
        if not overall_trend.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=overall_trend['Label'],
                y=overall_trend['Views'],
                mode='lines+markers',
                name='Views',
                line=dict(color='#0066cc', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=overall_trend['Label'],
                y=overall_trend['Functionarissen'],
                mode='lines+markers',
                name='Unieke functionarissen',
                yaxis='y2',
                line=dict(color='#28a745', width=2)
            ))
            fig.update_layout(
                title='Totaal views en actieve functionarissen per maand',
                yaxis=dict(title='Views', side='left'),
                yaxis2=dict(title='Functionarissen', side='right', overlaying='y'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                scores_df,
                names='Kleur',
                color='Kleur',
                color_discrete_map={'GROEN': '#28a745', 'ORANJE': '#ffc107', 'ROOD': '#dc3545'},
                title='Verdeling Customer Health'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Histogram van functionarissen
            fig = px.histogram(
                scores_df,
                x='Functionarissen',
                nbins=20,
                title='Verdeling aantal functionarissen per klant',
                labels={'Functionarissen': 'Aantal functionarissen', 'count': 'Aantal klanten'}
            )
            fig.add_vline(x=groen_threshold, line_dash="dash", line_color="green",
                         annotation_text=f"Groen grens ({groen_threshold})")
            fig.add_vline(x=oranje_threshold, line_dash="dash", line_color="orange",
                         annotation_text=f"Oranje grens ({oranje_threshold})")
            st.plotly_chart(fig, use_container_width=True)

        # Productgroepen analyse
        st.subheader("📊 Meest ontbrekende productgroepen")

        # Tel hoe vaak elke groep ontbreekt
        alle_ontbrekend = []
        for groepen in scores_df['Ontbrekende_Groepen']:
            if isinstance(groepen, list):
                alle_ontbrekend.extend(groepen)

        if alle_ontbrekend:
            ontbrekend_counts = pd.Series(alle_ontbrekend).value_counts().reset_index()
            ontbrekend_counts.columns = ['Productgroep', 'Aantal klanten zonder']

            fig = px.bar(
                ontbrekend_counts,
                x='Productgroep',
                y='Aantal klanten zonder',
                title='Kansen voor verbreding: welke productgroepen worden het minst gebruikt?',
                color='Aantal klanten zonder',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: SCORECARD ===
    with tab2:
        st.header("📋 Klanten Scorecard")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            kleur_filter = st.multiselect(
                "Filter op status",
                options=['GROEN', 'ORANJE', 'ROOD'],
                default=['GROEN', 'ORANJE', 'ROOD']
            )
        with col2:
            sort_by = st.selectbox("Sorteer op", ['Functionarissen', 'Views', 'Aantal_Groepen', 'Klantnaam'])

        # Apply filters
        filtered_df = scores_df[scores_df['Kleur'].isin(kleur_filter)]

        # Sort
        ascending = sort_by == 'Klantnaam'
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

        # Display table
        display_df = filtered_df[['Kleur', 'Klant_Code', 'Klantnaam', 'Functionarissen', 'Views', 'Aantal_Groepen', 'Ontbrekende_Groepen']].copy()
        display_df['Ontbrekende_Groepen'] = display_df['Ontbrekende_Groepen'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else '-'
        )
        display_df.columns = ['Status', 'Code', 'Klantnaam', 'Functionarissen', 'Views', 'Clusters', 'Ontbrekende clusters']

        def color_status(val):
            colors = {'GROEN': '#d4edda', 'ORANJE': '#fff3cd', 'ROOD': '#f8d7da'}
            return f'background-color: {colors.get(val, "")}'

        st.dataframe(
            display_df.style.applymap(color_status, subset=['Status']),
            use_container_width=True,
            height=500
        )

        # Export
        st.subheader("📤 Export")
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download als CSV", csv, "customer_scorecard.csv", "text/csv")
        with col2:
            buffer = BytesIO()
            filtered_df.to_excel(buffer, index=False)
            st.download_button("Download als Excel", buffer.getvalue(), "customer_scorecard.xlsx")

    # === TAB 3: KLANT DETAIL ===
    with tab3:
        st.header("🔍 Klant Detail")

        klant_options = scores_df.apply(lambda x: f"{x['Klant_Code']} - {x['Klantnaam']} ({x['Kleur']})", axis=1).tolist()
        selected_klant = st.selectbox("Selecteer klant", klant_options)

        if selected_klant:
            klant_code = selected_klant.split(' - ')[0]
            klant_info = scores_df[scores_df['Klant_Code'] == klant_code].iloc[0]

            # Klant header
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.subheader(f"{klant_info['Klantnaam']}")
                st.markdown(f"**Code:** {klant_code}")
            with col2:
                kleur = klant_info['Kleur']
                st.markdown(f"""
                <div class="{kleur.lower()}-card">
                    <h2 style="margin:0">{kleur}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                func_ok = klant_info['Functionarissen'] >= groen_threshold
                st.metric("Functionarissen", klant_info['Functionarissen'],
                         delta=f"{'✓' if func_ok else '✗'} groen ≥{groen_threshold}")
            with col4:
                views_ok = klant_info['Views'] >= SCORE_THRESHOLDS['groen_views']
                st.metric("Views", f"{klant_info['Views']:,}",
                         delta=f"{'✓' if views_ok else '✗'} groen ≥{SCORE_THRESHOLDS['groen_views']}")

            # Score breakdown
            func = klant_info['Functionarissen']
            views = klant_info['Views']
            func_groen = func >= groen_threshold
            views_groen = views >= SCORE_THRESHOLDS['groen_views']
            func_oranje = func >= oranje_threshold
            views_oranje = views >= SCORE_THRESHOLDS['oranje_views']

            if kleur == 'GROEN':
                st.success(f"✅ Beide criteria voldoen aan GROEN: functionarissen ({func} ≥ {groen_threshold}) en views ({views} ≥ {SCORE_THRESHOLDS['groen_views']})")
            elif kleur == 'ORANJE':
                issues = []
                if not func_groen:
                    issues.append(f"functionarissen ({func} < {groen_threshold})")
                if not views_groen:
                    issues.append(f"views ({views} < {SCORE_THRESHOLDS['groen_views']})")
                st.warning(f"⚠️ Niet groen vanwege: {', '.join(issues)}")
            else:
                issues = []
                if not func_oranje:
                    issues.append(f"functionarissen ({func} < {oranje_threshold})")
                if not views_oranje:
                    issues.append(f"views ({views} < {SCORE_THRESHOLDS['oranje_views']})")
                st.error(f"🔴 Rood vanwege: {', '.join(issues)}")

            st.metric("Rapportclusters", f"{klant_info['Aantal_Groepen']} / 13")

            # Ontbrekende rapportclusters highlight
            ontbrekend = klant_info['Ontbrekende_Groepen']
            if isinstance(ontbrekend, list) and len(ontbrekend) > 0:
                st.warning(f"**Kansen voor verbreding ({len(ontbrekend)} ontbrekende clusters):** {', '.join(ontbrekend)}")

            # Trend grafiek voor deze klant
            st.subheader("📈 Trend over tijd")
            klant_trend = get_trend_data(pbi_df, klant_code)
            if not klant_trend.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=klant_trend['Label'],
                    y=klant_trend['Views'],
                    mode='lines+markers',
                    name='Views',
                    line=dict(color='#0066cc', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 102, 204, 0.1)'
                ))
                fig.add_trace(go.Scatter(
                    x=klant_trend['Label'],
                    y=klant_trend['Functionarissen'],
                    mode='lines+markers',
                    name='Actieve functionarissen',
                    yaxis='y2',
                    line=dict(color='#28a745', width=2)
                ))
                fig.update_layout(
                    title=f'Views en actieve functionarissen per maand - {klant_info["Klantnaam"]}',
                    yaxis=dict(title='Views', side='left'),
                    yaxis2=dict(title='Functionarissen', side='right', overlaying='y'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Geen trenddata beschikbaar (Jaar/Maand kolommen ontbreken)")

            # Contactpersonen voor deze klant
            if contacts_df is not None:
                st.subheader("📧 Contactpersonen")
                klant_contacts = get_contacts_for_customer(contacts_df, klant_info['Klantnaam'])

                if not klant_contacts.empty:
                    for _, contact in klant_contacts.iterrows():
                        email = contact['Email'] if pd.notna(contact['Email']) else 'Geen email'
                        labels = contact['Labels'] if pd.notna(contact['Labels']) else ''
                        st.markdown(f"""
                        <div class="contact-card">
                            <strong>{contact['Naam']}</strong><br>
                            📧 <a href="mailto:{email}">{email}</a><br>
                            <small>{labels}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Geen contactpersonen gevonden voor deze klant")

            col1, col2 = st.columns(2)

            # Categorie breakdown
            with col1:
                st.subheader("📊 Gebruik per Productgroep")
                cat_breakdown = get_category_breakdown(pbi_df, klant_code)

                if not cat_breakdown.empty:
                    fig = px.bar(
                        cat_breakdown,
                        x='Categorie',
                        y='Views',
                        color='Functionarissen',
                        title=f'Views per productgroep'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # User breakdown
            with col2:
                st.subheader("👥 Functionarissen")
                user_breakdown = get_user_breakdown(pbi_df, klant_code)

                if not user_breakdown.empty:
                    fig = px.bar(
                        user_breakdown.head(10),
                        x='Naam',
                        y='Views',
                        title=f'Top 10 gebruikers'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # === TAB 4: CONTACTPERSONEN ===
    with tab4:
        st.header("📧 Contactpersonen Export")

        if contacts_df is None:
            st.warning("Upload de Pipedrive People export om contactpersonen te zien")
        else:
            # Filter opties
            col1, col2 = st.columns(2)
            with col1:
                contact_kleur_filter = st.multiselect(
                    "Filter op klant status",
                    options=['GROEN', 'ORANJE', 'ROOD'],
                    default=['ROOD', 'ORANJE'],
                    key='contact_kleur'
                )
            with col2:
                label_filter = st.multiselect(
                    "Filter op rol",
                    options=['Controller', 'Directie', 'Contactpersoon consultancy', 'Contactpersoon IT (intern)'],
                    default=['Controller', 'Directie']
                )

            # Bouw contactenlijst met klant status
            contact_list = []
            for _, klant in scores_df[scores_df['Kleur'].isin(contact_kleur_filter)].iterrows():
                klant_contacts = get_contacts_for_customer(contacts_df, klant['Klantnaam'])
                for _, contact in klant_contacts.iterrows():
                    contact_labels = str(contact['Labels']) if pd.notna(contact['Labels']) else ''
                    if any(label in contact_labels for label in label_filter) or len(label_filter) == 0:
                        contact_list.append({
                            'Klant_Code': klant['Klant_Code'],
                            'Klantnaam': klant['Klantnaam'],
                            'Status': klant['Kleur'],
                            'Functionarissen': klant['Functionarissen'],
                            'Contact_Naam': contact['Naam'],
                            'Email': contact['Email'],
                            'Rol': contact['Labels']
                        })

            if contact_list:
                contact_export_df = pd.DataFrame(contact_list)
                contact_export_df = contact_export_df.sort_values(['Status', 'Functionarissen'], ascending=[True, True])

                st.dataframe(contact_export_df, use_container_width=True, height=400)

                st.subheader("📤 Export voor Campagne")

                col1, col2, col3 = st.columns(3)
                with col1:
                    emails = contact_export_df['Email'].dropna().unique()
                    email_str = '; '.join(emails)
                    st.text_area("Email adressen (voor BCC)", email_str, height=100)

                with col2:
                    csv = contact_export_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "contacten_campagne.csv", "text/csv")

                with col3:
                    buffer = BytesIO()
                    contact_export_df.to_excel(buffer, index=False)
                    st.download_button("Download Excel", buffer.getvalue(), "contacten_campagne.xlsx")

                st.info(f"📊 {len(contact_export_df)} contactpersonen van {contact_export_df['Klant_Code'].nunique()} klanten")
            else:
                st.info("Geen contactpersonen gevonden met de huidige filters")

    # === TAB 5: RAPPORT MAPPING ===
    with tab5:
        st.header("📑 Rapport naar Rapportcluster Mapping")

        st.markdown("""
        Hieronder zie je welke rapporten onder welke rapportcluster vallen, en op welke zoekterm de match is gemaakt.
        Dit helpt bij het valideren en verbeteren van de categorisatie.
        """)

        # Genereer categorisatie overzicht MET klantinformatie
        pbi_df['Rapportcluster'] = pbi_df['Report name'].apply(categorize_report)

        # Maak een dataframe met rapport, cluster, en bij welke klanten het gebruikt wordt
        report_usage = pbi_df.groupby(['Report name', 'Rapportcluster']).agg({
            'Klant_Code': lambda x: ', '.join(sorted(x.dropna().unique())[:5]) + ('...' if len(x.dropna().unique()) > 5 else ''),
            'Aantal activity reportviews': 'sum'
        }).reset_index()
        report_usage.columns = ['Rapport', 'Rapportcluster', 'Gebruikt door klanten', 'Totaal views']
        report_usage['Totaal views'] = report_usage['Totaal views'].round(0).astype(int)

        # Zoek matched term
        def get_matched_term(row):
            if row['Rapportcluster'] == 'Niet geclassificeerd':
                return '-'
            name_lower = str(row['Rapport']).lower()
            for term in RAPPORT_CATEGORIE_REGELS.get(row['Rapportcluster'], []):
                if term in name_lower:
                    return term
            return '-'

        report_usage['Matched op'] = report_usage.apply(get_matched_term, axis=1)

        # Filter opties
        col1, col2, col3 = st.columns(3)
        with col1:
            # Klant filter met naam en code
            klant_mapping = {f"{row['Klantnaam']} ({row['Klant_Code']})": row['Klant_Code']
                           for _, row in scores_df[['Klant_Code', 'Klantnaam']].drop_duplicates().iterrows()}
            klant_display_options = ['Alle klanten'] + sorted(klant_mapping.keys())
            selected_klant_display = st.selectbox("Filter op klant", klant_display_options, key='mapping_klant')
            selected_klant_filter = klant_mapping.get(selected_klant_display, 'Alle klanten') if selected_klant_display != 'Alle klanten' else 'Alle klanten'
        with col2:
            cat_filter = st.multiselect(
                "Filter op rapportcluster",
                options=ALLE_RAPPORTCLUSTERS + ['Niet geclassificeerd'],
                default=['Niet geclassificeerd']
            )
        with col3:
            show_only_unclassified = st.checkbox("Toon alleen 'Niet geclassificeerd'", value=True)

        # Filter data
        if show_only_unclassified:
            filtered_reports = report_usage[report_usage['Rapportcluster'] == 'Niet geclassificeerd']
        else:
            filtered_reports = report_usage[report_usage['Rapportcluster'].isin(cat_filter)]

        # Extra filter op klant
        if selected_klant_filter != 'Alle klanten':
            # Haal rapporten op die door deze klant gebruikt worden
            klant_rapporten = pbi_df[pbi_df['Klant_Code'] == selected_klant_filter]['Report name'].unique()
            filtered_reports = filtered_reports[filtered_reports['Rapport'].isin(klant_rapporten)]

        # Sorteer op views (meest gebruikte eerst)
        filtered_reports = filtered_reports.sort_values('Totaal views', ascending=False)

        # Toon tabel
        st.dataframe(filtered_reports, use_container_width=True, height=500)

        # Samenvatting
        st.subheader("📊 Samenvatting categorisatie")
        summary = report_usage.groupby('Rapportcluster').size().reset_index(name='Aantal rapporten')
        summary = summary.sort_values('Aantal rapporten', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(summary, use_container_width=True)
        with col2:
            fig = px.pie(summary, values='Aantal rapporten', names='Rapportcluster',
                        title='Verdeling rapporten per rapportcluster')
            st.plotly_chart(fig, use_container_width=True)

        # Zoekregels tonen
        st.subheader("🔍 Huidige zoekregels per rapportcluster")

        for groep, termen in RAPPORT_CATEGORIE_REGELS.items():
            with st.expander(f"**{groep}** ({len(termen)} zoektermen)"):
                st.markdown(", ".join([f"`{t}`" for t in termen]))

        # Analyse niet-geclassificeerde rapporten
        unclassified = report_usage[report_usage['Rapportcluster'] == 'Niet geclassificeerd']['Rapport'].tolist()
        if unclassified:
            st.subheader("💡 Suggesties voor nieuwe zoektermen")
            st.markdown(f"Er zijn nog **{len(unclassified)}** niet-geclassificeerde rapporten.")

            # Extract woorden uit niet-geclassificeerde rapporten
            all_words = []
            for report in unclassified:
                words = str(report).lower().replace('_', ' ').replace('-', ' ').split()
                all_words.extend([w for w in words if len(w) > 3])

            # Tel woorden en filter op frequentie
            from collections import Counter
            word_counts = Counter(all_words)
            common_words = [(word, count) for word, count in word_counts.most_common(20)
                           if count >= 2 and word not in ['dashboard', 'rapport', 'report', 'kpi', 'overzicht']]

            if common_words:
                st.markdown("**Veelvoorkomende woorden in niet-geclassificeerde rapporten:**")
                word_df = pd.DataFrame(common_words, columns=['Woord', 'Frequentie'])
                st.dataframe(word_df, use_container_width=True, height=200)
                st.markdown("*Voeg relevante woorden toe aan de zoektermen in `RAPPORT_CATEGORIE_REGELS`*")

        # Classificatie percentage
        total_reports = len(report_usage)
        classified = total_reports - len(unclassified)
        pct = (classified / total_reports * 100) if total_reports > 0 else 0
        st.metric("Classificatie percentage", f"{pct:.1f}%", delta=f"{classified}/{total_reports} rapporten")

        # Export
        st.subheader("📤 Export")
        col1, col2 = st.columns(2)
        with col1:
            csv = report_usage.to_csv(index=False).encode('utf-8')
            st.download_button("Download mapping als CSV", csv, "rapport_mapping.csv", "text/csv")
        with col2:
            buffer = BytesIO()
            report_usage.to_excel(buffer, index=False)
            st.download_button("Download mapping als Excel", buffer.getvalue(), "rapport_mapping.xlsx")

    # === TAB 6: UPSELL EMAILS ===
    with tab6:
        st.header("📨 Upsell Emails")
        st.markdown("""
        Identificeer klanten die bepaalde rapportclusters nog niet gebruiken en genereer
        een email om ze te overtuigen deze functionaliteit wel te gaan gebruiken.
        """)

        # S&O clusters - sommige klanten hebben geen S&O
        SO_CLUSTERS = [
            'Bewaking rendement S&O',
            'Onderhoudsplanning',
            'Openstaande Werkbonnen',
            'S&O Uitvoering'
        ]

        # Standaard clusters - alle klanten zouden deze moeten gebruiken
        STANDAARD_CLUSTERS = [
            'Financieel Overzicht',
            'Projectwaardering',
            'Bewaking productiviteit',
            'Projectbewaking',
            'Offerte en Sales proces',
            'Resource Management',
            'Liquiditeitsoverzicht',
            'Werkkapitaal (debiteuren, voorraad)',
            'Directierapport'
        ]

        # Analyseer per klant welke clusters ze gebruiken
        klant_cluster_usage = {}

        for klant in scores_df['Klant_Code'].unique():
            klant_rapporten = pbi_df[pbi_df['Klant_Code'] == klant]['Report name'].unique()
            gebruikte_clusters = set()

            for rapport in klant_rapporten:
                cluster = categorize_report(rapport)
                if cluster != 'Niet geclassificeerd':
                    gebruikte_clusters.add(cluster)

            klant_cluster_usage[klant] = gebruikte_clusters

        # Filter opties
        st.subheader("🎯 Selecteer doelgroep")

        col1, col2 = st.columns(2)
        with col1:
            include_so = st.checkbox("Inclusief S&O clusters", value=False,
                help="Vink aan als je ook S&O clusters wilt meenemen (niet alle klanten hebben S&O)")
        with col2:
            min_missing = st.slider("Minimum aantal ontbrekende clusters", 1, 5, 2)

        # Bepaal welke clusters te checken
        clusters_to_check = STANDAARD_CLUSTERS.copy()
        if include_so:
            clusters_to_check.extend(SO_CLUSTERS)

        # Selecteer specifieke cluster om op te focussen
        selected_cluster = st.selectbox(
            "Focus op specifieke rapportcluster (optioneel)",
            options=['Alle clusters'] + clusters_to_check
        )

        # Vind klanten met ontbrekende clusters
        missing_data = []
        for klant, gebruikte in klant_cluster_usage.items():
            if selected_cluster == 'Alle clusters':
                ontbrekend = [c for c in clusters_to_check if c not in gebruikte]
            else:
                ontbrekend = [selected_cluster] if selected_cluster not in gebruikte else []

            if len(ontbrekend) >= min_missing or (selected_cluster != 'Alle clusters' and ontbrekend):
                # Haal klantnaam op
                klant_info = klantnamen[klantnamen['Klant_Code'] == klant]
                klant_naam = klant_info['Klantnaam'].iloc[0] if len(klant_info) > 0 else klant

                missing_data.append({
                    'Klant_Code': klant,
                    'Klantnaam': klant_naam,
                    'Aantal ontbrekend': len(ontbrekend),
                    'Ontbrekende clusters': ', '.join(ontbrekend),
                    'Wel in gebruik': ', '.join(sorted(gebruikte))
                })

        missing_df = pd.DataFrame(missing_data)

        if len(missing_df) > 0:
            st.subheader(f"📋 Klanten met ontbrekende clusters ({len(missing_df)} klanten)")
            missing_df = missing_df.sort_values('Aantal ontbrekend', ascending=False)
            st.dataframe(missing_df, use_container_width=True, height=300)

            # Email generatie sectie
            st.subheader("✉️ Email Genereren")

            # Selecteer klanten voor email
            selected_klanten = st.multiselect(
                "Selecteer klanten voor email",
                options=missing_df['Klantnaam'].tolist(),
                default=[]
            )

            if selected_klanten:
                # Haal ontbrekende clusters op voor geselecteerde klanten
                ontbrekend_per_klant = {}
                for klant_naam in selected_klanten:
                    row = missing_df[missing_df['Klantnaam'] == klant_naam].iloc[0]
                    ontbrekend_per_klant[klant_naam] = row['Ontbrekende clusters'].split(', ')

                # Bepaal gemeenschappelijke ontbrekende clusters
                all_missing = set()
                for clusters in ontbrekend_per_klant.values():
                    all_missing.update(clusters)

                st.markdown(f"**Ontbrekende clusters bij geselecteerde klanten:** {', '.join(all_missing)}")

                # Email template
                email_cluster = st.selectbox(
                    "Selecteer cluster voor email focus",
                    options=list(all_missing)
                ) if all_missing else None

                if email_cluster:
                    # Cluster beschrijvingen voor email (informele toon met jij/je)
                    CLUSTER_BESCHRIJVINGEN = {
                        'Financieel Overzicht': 'direct inzicht in je financiële positie, van resultatenrekening tot begroting',
                        'Projectwaardering': 'nauwkeurige waardering van je projecten met nacalculatie en rendementsanalyse',
                        'Bewaking productiviteit': 'grip op productiviteit, urenregistratie en personele bezetting',
                        'Projectbewaking': 'volledige controle over projectvoortgang, onderhanden werk en meerwerk',
                        'Offerte en Sales proces': 'inzicht in je sales pipeline, offertes en orderintake',
                        'Resource Management': 'optimale inzet van je resources met capaciteitsplanning',
                        'Liquiditeitsoverzicht': 'actueel zicht op je liquiditeitspositie en cashflow forecast',
                        'Werkkapitaal (debiteuren, voorraad)': 'beheer van debiteuren, crediteuren en voorraden',
                        'Bewaking rendement S&O': 'analyse van service & onderhoud rendement',
                        'Onderhoudsplanning': 'planning en bewaking van onderhoudswerkzaamheden',
                        'Openstaande Werkbonnen': 'overzicht van openstaande werkorders en tickets',
                        'S&O Uitvoering': 'monitoring van service uitvoering en storingen',
                        'Directierapport': 'compact overzicht voor directie met alle belangrijke KPIs'
                    }

                    beschrijving = CLUSTER_BESCHRIJVINGEN.get(email_cluster, 'waardevolle inzichten voor je organisatie')

                    # Genereer email
                    st.markdown("---")
                    st.markdown("### 📧 Gegenereerde Email")

                    # Bouw to: lijst via get_contacts_for_customer functie
                    to_emails = []
                    if contacts_df is not None and len(contacts_df) > 0:
                        for klant_naam in selected_klanten:
                            klant_contacts = get_contacts_for_customer(contacts_df, klant_naam)
                            if len(klant_contacts) > 0:
                                for _, contact in klant_contacts.iterrows():
                                    email = contact.get('Email')
                                    if pd.notna(email):
                                        emails = str(email).split(',')
                                        to_emails.extend([e.strip() for e in emails if '@' in e])

                    to_line = '; '.join(list(set(to_emails))[:10]) if to_emails else ''

                    # Email body (informele toon)
                    email_subject = f"Haal meer uit je Power BI dashboards - {email_cluster}"
                    email_body = f"""Hoi,

We zagen dat jullie nog geen gebruik maken van onze {email_cluster} dashboards.

Dit cluster biedt je {beschrijving}.

Wat kun je verwachten?
- Direct inzicht zonder handmatige rapportages
- Altijd actuele data vanuit je bronsysteem
- Speciaal ontworpen voor jullie branche

Wil je meer weten over de mogelijkheden? We plannen graag een korte demo in om te laten zien hoe dit dashboard jullie kan helpen.

Groet,

Het Notifica Team"""

                    # Display template
                    email_display = f"""**Aan:** {to_line if to_line else '[geen emails gevonden]'}

**Onderwerp:** {email_subject}

---

{email_body}
"""
                    st.text_area("Email template", email_display, height=400)

                    # Mailto link met body - URL encode de body
                    import urllib.parse
                    encoded_body = urllib.parse.quote(email_body)
                    encoded_subject = urllib.parse.quote(email_subject)

                    col1, col2 = st.columns(2)
                    with col1:
                        if to_line:
                            mailto_url = f"mailto:{to_line}?subject={encoded_subject}&body={encoded_body}"
                            st.markdown(f'''
                            <a href="{mailto_url}" target="_blank" style="
                                display: inline-block;
                                padding: 10px 20px;
                                background-color: #0066cc;
                                color: white;
                                text-decoration: none;
                                border-radius: 5px;
                                font-weight: bold;
                            ">📧 Open in Outlook</a>
                            ''', unsafe_allow_html=True)
                        else:
                            st.warning("Geen email adressen gevonden. Upload Pipedrive People data.")

                    with col2:
                        st.download_button(
                            "📋 Download als tekstbestand",
                            email_display,
                            file_name=f"upsell_email_{email_cluster.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

        else:
            st.info("Geen klanten gevonden met ontbrekende clusters op basis van de huidige filters.")

        # Samenvatting cluster adoptie
        st.subheader("📊 Cluster Adoptie Overzicht")

        adoptie_data = []
        for cluster in ALLE_RAPPORTCLUSTERS:
            gebruikers = sum(1 for usage in klant_cluster_usage.values() if cluster in usage)
            totaal = len(klant_cluster_usage)
            adoptie_data.append({
                'Rapportcluster': cluster,
                'Aantal klanten': gebruikers,
                'Percentage': f"{gebruikers/totaal*100:.0f}%" if totaal > 0 else "0%",
                'Type': 'S&O' if cluster in SO_CLUSTERS else 'Standaard'
            })

        adoptie_df = pd.DataFrame(adoptie_data)
        adoptie_df = adoptie_df.sort_values('Aantal klanten', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(adoptie_df, use_container_width=True)
        with col2:
            fig = px.bar(adoptie_df, x='Rapportcluster', y='Aantal klanten',
                        color='Type', title='Adoptie per rapportcluster',
                        color_discrete_map={'Standaard': '#3636A2', 'S&O': '#16136F'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 7: AI ASSISTENT ===
    with tab7:
        st.header("🤖 AI Assistent")
        st.markdown("""
        Stel vragen over je klantdata en krijg strategische inzichten van AI.
        De AI kan analyses maken, campagne-ideeën suggereren en helpen met klantbenadering.
        """)

        # API key uit .env (al geladen via load_dotenv)
        api_key = os.getenv('ANTHROPIC_API_KEY', '')

        if not ANTHROPIC_AVAILABLE:
            st.error("**Anthropic library niet geïnstalleerd** - Run: `pip install anthropic`")
        elif not api_key:
            st.warning("**Geen API key gevonden** - Voeg `ANTHROPIC_API_KEY` toe aan het `.env` bestand")

        if api_key and ANTHROPIC_AVAILABLE:
            # Bereid data context voor
            def prepare_data_context():
                """Maak een samenvatting van de data voor de AI"""
                groen_count = len(scores_df[scores_df['Kleur'] == 'GROEN'])
                oranje_count = len(scores_df[scores_df['Kleur'] == 'ORANJE'])
                rood_count = len(scores_df[scores_df['Kleur'] == 'ROOD'])
                total_count = len(scores_df)

                context = f"""## Notifica Customer Health Data Context

### Algemene Statistieken
- Totaal aantal klanten: {total_count}
- GROEN (gezond): {groen_count} ({groen_count/total_count*100:.1f}%)
- ORANJE (aandacht): {oranje_count} ({oranje_count/total_count*100:.1f}%)
- ROOD (risico): {rood_count} ({rood_count/total_count*100:.1f}%)

### Klant Health Criteria
- GROEN: minimaal {groen_threshold} functionarissen EN {SCORE_THRESHOLDS['groen_views']} views
- ORANJE: minimaal {oranje_threshold} functionarissen EN {SCORE_THRESHOLDS['oranje_views']} views
- ROOD: onder ORANJE drempels

### De 13 Notifica Rapportclusters
{chr(10).join([f'- {c}' for c in ALLE_RAPPORTCLUSTERS])}

### Top 10 Klanten (op functionarissen)
"""
                top_klanten = scores_df.nlargest(10, 'Functionarissen')[['Klantnaam', 'Klant_Code', 'Functionarissen', 'Views', 'Kleur', 'Aantal_Groepen']].to_string()
                context += top_klanten

                context += "\n\n### Bottom 10 Klanten (risico - laagste functionarissen)\n"
                bottom_klanten = scores_df.nsmallest(10, 'Functionarissen')[['Klantnaam', 'Klant_Code', 'Functionarissen', 'Views', 'Kleur', 'Aantal_Groepen']].to_string()
                context += bottom_klanten

                context += "\n\n### Cluster Adoptie (aantal klanten per cluster)\n"
                for cluster in ALLE_RAPPORTCLUSTERS:
                    gebruikers = sum(1 for _, row in scores_df.iterrows()
                                   if isinstance(row.get('Gebruikte_Groepen'), list) and cluster in row['Gebruikte_Groepen'])
                    context += f"- {cluster}: {gebruikers} klanten ({gebruikers/total_count*100:.0f}%)\n"

                context += "\n### Meest Ontbrekende Clusters (kansen voor upsell)\n"
                alle_ontbrekend = []
                for groepen in scores_df['Ontbrekende_Groepen']:
                    if isinstance(groepen, list):
                        alle_ontbrekend.extend(groepen)

                if alle_ontbrekend:
                    from collections import Counter
                    ontbrekend_counts = Counter(alle_ontbrekend).most_common(10)
                    for cluster, count in ontbrekend_counts:
                        context += f"- {cluster}: ontbreekt bij {count} klanten\n"

                return context

            # Functie om AI analyse uit te voeren
            def run_ai_analysis(prompt):
                """Voer AI analyse uit en sla resultaat op in session state"""
                try:
                    data_context = prepare_data_context()
                    client = anthropic.Anthropic(api_key=api_key)

                    system_prompt = """Je bent een strategische consultant voor Notifica, een bedrijf dat Power BI dashboards levert aan klanten.
Je analyseert customer health data en geeft concrete, actionable adviezen.

Jouw stijl:
- Informeel maar professioneel (je/jij, geen u)
- Concreet en to-the-point
- Data-gedreven met specifieke voorbeelden uit de data
- Focus op praktische acties die morgen gestart kunnen worden
- Denk mee als business partner, niet alleen als analist

Je hebt toegang tot klantdata inclusief:
- Health status (GROEN/ORANJE/ROOD) gebaseerd op functionarissen en views
- Welke rapportclusters elke klant gebruikt
- Welke clusters ontbreken (upsell kansen)
- Totale statistieken over alle klanten"""

                    message = client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Hier is de actuele data van onze klanten:

{data_context}

---

Mijn vraag: {prompt}"""
                            }
                        ]
                    )

                    result = message.content[0].text

                    # Sla op in history
                    if 'ai_history' not in st.session_state:
                        st.session_state['ai_history'] = []
                    st.session_state['ai_history'].append({
                        'vraag': prompt,
                        'antwoord': result,
                        'timestamp': datetime.now().strftime("%H:%M")
                    })

                    # Sla laatste resultaat op
                    st.session_state['ai_last_result'] = result
                    st.session_state['ai_last_question'] = prompt

                    return result

                except Exception as e:
                    error_msg = f"Fout bij AI analyse: {str(e)}"
                    st.session_state['ai_error'] = error_msg
                    return None

            # Chat interface
            st.subheader("💬 Chat met AI")

            # Suggestie knoppen
            st.markdown("**Snelle vragen:**")
            col1, col2, col3 = st.columns(3)

            suggested_prompts = {
                "📊 Analyse risico klanten": "Analyseer de rode klanten en geef concrete actie-suggesties per klant om ze te verbeteren naar oranje of groen. Wat zijn de belangrijkste patronen?",
                "📈 Webinar strategie": "Op basis van de cluster adoptie data, welke webinars zou je aanraden om te organiseren? Geef een top 3 met argumentatie en doelgroep per webinar.",
                "🎯 Upsell prioriteiten": "Welke rapportclusters hebben de meeste upsell potentie? Geef per cluster een concrete aanpak om klanten te overtuigen.",
                "📧 Campagne planning": "Stel een kwartaalplanning voor met verschillende campagnes (email, webinar, content) om klant adoptie te verhogen.",
                "🔍 Klant segmentatie": "Segmenteer de klanten in groepen op basis van hun gebruik patronen. Welke segmenten zie je en hoe zou je elke segment benaderen?",
                "💡 Quick wins": "Welke 5 acties kunnen we morgen starten die de meeste impact hebben op customer health?"
            }

            with col1:
                if st.button("📊 Analyse risico klanten", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["📊 Analyse risico klanten"]
                    st.rerun()
                if st.button("📈 Webinar strategie", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["📈 Webinar strategie"]
                    st.rerun()
            with col2:
                if st.button("🎯 Upsell prioriteiten", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["🎯 Upsell prioriteiten"]
                    st.rerun()
                if st.button("📧 Campagne planning", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["📧 Campagne planning"]
                    st.rerun()
            with col3:
                if st.button("🔍 Klant segmentatie", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["🔍 Klant segmentatie"]
                    st.rerun()
                if st.button("💡 Quick wins", use_container_width=True):
                    st.session_state['ai_pending_prompt'] = suggested_prompts["💡 Quick wins"]
                    st.rerun()

            st.markdown("---")

            # Tekstveld voor eigen vraag
            user_prompt = st.text_area(
                "Stel je vraag aan de AI",
                height=100,
                placeholder="Bijvoorbeeld: Welke klanten hebben de meeste potentie voor groei? Of: Hoe kunnen we de rode klanten het beste benaderen?",
                key="ai_user_input"
            )

            # Analyse knop
            if st.button("🚀 Analyseer", type="primary", use_container_width=True):
                if user_prompt:
                    st.session_state['ai_pending_prompt'] = user_prompt
                    st.rerun()
                else:
                    st.warning("Voer eerst een vraag in")

            # Verwerk pending prompt (na rerun)
            if 'ai_pending_prompt' in st.session_state:
                pending = st.session_state['ai_pending_prompt']
                del st.session_state['ai_pending_prompt']

                with st.spinner("AI analyseert je data..."):
                    result = run_ai_analysis(pending)

            # Toon laatste resultaat
            if 'ai_last_result' in st.session_state and st.session_state.get('ai_last_result'):
                st.markdown("---")
                st.markdown(f"### 🤖 AI Analyse")
                st.markdown(f"**Vraag:** {st.session_state.get('ai_last_question', '')}")
                st.markdown("---")
                st.markdown(st.session_state['ai_last_result'])

            # Toon eventuele error
            if 'ai_error' in st.session_state:
                st.error(st.session_state['ai_error'])
                del st.session_state['ai_error']

            # Toon chat history
            if 'ai_history' in st.session_state and len(st.session_state['ai_history']) > 1:
                st.markdown("---")
                st.subheader("📜 Eerdere analyses")

                # Toon alle behalve de laatste (die staat al bovenaan)
                for item in reversed(st.session_state['ai_history'][:-1]):
                    with st.expander(f"**{item['timestamp']}** - {item['vraag'][:50]}..."):
                        st.markdown(f"**Vraag:** {item['vraag']}")
                        st.markdown("**Antwoord:**")
                        st.markdown(item['antwoord'])

                if st.button("🗑️ Wis geschiedenis"):
                    st.session_state['ai_history'] = []
                    if 'ai_last_result' in st.session_state:
                        del st.session_state['ai_last_result']
                    if 'ai_last_question' in st.session_state:
                        del st.session_state['ai_last_question']
                    st.rerun()

            # Data preview voor debug
            with st.expander("📋 Data die naar AI wordt gestuurd"):
                st.code(prepare_data_context(), language="markdown")


if __name__ == "__main__":
    main()
