"""
Google Sheets integration for Consultancy Forecast
Handles persistent storage of projects, klanten, and regie data
"""
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Google Sheets scope
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


@st.cache_resource
def get_gspread_client():
    """Get authenticated gspread client using Streamlit secrets"""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets authenticatie mislukt: {e}")
        return None


def get_spreadsheet():
    """Get the Consultancy Forecast spreadsheet"""
    client = get_gspread_client()
    if not client:
        return None

    try:
        # Get spreadsheet ID from secrets
        spreadsheet_id = st.secrets.get("spreadsheet_id", None)
        if spreadsheet_id:
            return client.open_by_key(spreadsheet_id)
        else:
            # Fallback: try to open by name
            return client.open("Consultancy Forecast")
    except gspread.SpreadsheetNotFound:
        st.error("Spreadsheet 'Consultancy Forecast' niet gevonden. Maak deze eerst aan in Google Sheets.")
        return None
    except Exception as e:
        st.error(f"Spreadsheet openen mislukt: {e}")
        return None


def ensure_worksheet(spreadsheet, name: str, headers: list) -> gspread.Worksheet:
    """Ensure a worksheet exists with the given headers"""
    try:
        worksheet = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=name, rows=100, cols=len(headers))
        worksheet.update('A1', [headers])
    return worksheet


# =============================================================================
# PROJECTEN
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_projects_from_gsheets() -> list:
    """Load projects from Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return []

    try:
        headers = ["Klant", "Opdracht", "Type", "Partner", "Categorie", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]
        worksheet = ensure_worksheet(spreadsheet, "Projecten", headers)

        data = worksheet.get_all_records()
        if not data:
            return []

        # Ensure numeric columns are floats
        for row in data:
            for col in ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]:
                if col in row:
                    try:
                        row[col] = float(row[col]) if row[col] != "" else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0

        return data
    except Exception as e:
        st.sidebar.warning(f"Projecten laden mislukt: {e}")
        return []


def save_projects_to_gsheets(projects: list) -> bool:
    """Save projects to Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return False

    try:
        headers = ["Klant", "Opdracht", "Type", "Partner", "Categorie", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]
        worksheet = ensure_worksheet(spreadsheet, "Projecten", headers)

        # Clear and rewrite
        worksheet.clear()

        # Prepare data with headers
        if projects:
            df = pd.DataFrame(projects)
            # Ensure Categorie column exists
            if 'Categorie' not in df.columns:
                df['Categorie'] = '-'
            df = df.reindex(columns=headers)
            data = [headers] + df.values.tolist()
        else:
            data = [headers]

        worksheet.update('A1', data)
        # Clear cache after saving
        load_projects_from_gsheets.clear()
        return True
    except Exception as e:
        st.sidebar.error(f"Projecten opslaan mislukt: {e}")
        return False


# =============================================================================
# KLANTEN (Pipedrive)
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_klanten_from_gsheets() -> list:
    """Load klanten from Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return []

    try:
        headers = ["Naam", "Branche", "Gewonnen deals", "Openstaande deals"]
        worksheet = ensure_worksheet(spreadsheet, "Klanten", headers)

        data = worksheet.get_all_records()
        return data if data else []
    except Exception as e:
        st.sidebar.warning(f"Klanten laden mislukt: {e}")
        return []


def save_klanten_to_gsheets(klanten: list) -> bool:
    """Save klanten to Google Sheets (from Pipedrive upload)"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return False

    try:
        headers = ["Naam", "Branche", "Gewonnen deals", "Openstaande deals"]
        worksheet = ensure_worksheet(spreadsheet, "Klanten", headers)

        # Clear and rewrite
        worksheet.clear()

        if klanten:
            df = pd.DataFrame(klanten)
            df = df.reindex(columns=headers)
            data = [headers] + df.fillna("").values.tolist()
        else:
            data = [headers]

        worksheet.update('A1', data)
        # Clear cache after saving
        load_klanten_from_gsheets.clear()
        return True
    except Exception as e:
        st.sidebar.error(f"Klanten opslaan mislukt: {e}")
        return False


# =============================================================================
# REGIE UREN
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_regie_from_gsheets() -> list:
    """Load regie uren from Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return []

    try:
        headers = ["Partner", "Klant", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]
        worksheet = ensure_worksheet(spreadsheet, "Regie", headers)

        data = worksheet.get_all_records()
        if not data:
            return []

        # Convert to project format
        regie_projects = []
        for row in data:
            project = {
                "Klant": row.get("Klant", "Diverse"),
                "Opdracht": f"{row.get('Partner', '')} regie",
                "Type": "Regie",
                "Partner": row.get("Partner", ""),
                "Jan": float(row.get("Jan", 0) or 0),
                "Feb": float(row.get("Feb", 0) or 0),
                "Mrt": float(row.get("Mrt", 0) or 0),
                "Apr": float(row.get("Apr", 0) or 0),
                "Mei": float(row.get("Mei", 0) or 0),
                "Jun": float(row.get("Jun", 0) or 0),
                "Totaal": float(row.get("Totaal", 0) or 0),
            }
            if project["Totaal"] > 0:
                regie_projects.append(project)

        return regie_projects
    except Exception as e:
        st.sidebar.warning(f"Regie laden mislukt: {e}")
        return []


def save_regie_to_gsheets(regie_data: list) -> bool:
    """Save regie data to Google Sheets (from Excel upload)"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return False

    try:
        headers = ["Partner", "Klant", "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Totaal"]
        worksheet = ensure_worksheet(spreadsheet, "Regie", headers)

        # Clear and rewrite
        worksheet.clear()

        if regie_data:
            df = pd.DataFrame(regie_data)
            df = df.reindex(columns=headers)
            data = [headers] + df.fillna(0).values.tolist()
        else:
            data = [headers]

        worksheet.update('A1', data)
        # Clear cache after saving
        load_regie_from_gsheets.clear()
        return True
    except Exception as e:
        st.sidebar.error(f"Regie opslaan mislukt: {e}")
        return False


# =============================================================================
# OMZET 2025 (voor segmentatie)
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_omzet_2025_from_gsheets() -> dict:
    """Load 2025 revenue per client from Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return {}

    try:
        headers = ["Klant", "Omzet"]
        worksheet = ensure_worksheet(spreadsheet, "Omzet2025", headers)

        data = worksheet.get_all_records()
        if not data:
            return {}

        return {row["Klant"]: float(row.get("Omzet", 0) or 0) for row in data if row.get("Klant")}
    except Exception as e:
        st.sidebar.warning(f"Omzet 2025 laden mislukt: {e}")
        return {}


def save_omzet_2025_to_gsheets(omzet_data: dict) -> bool:
    """Save 2025 revenue data to Google Sheets"""
    spreadsheet = get_spreadsheet()
    if not spreadsheet:
        return False

    try:
        headers = ["Klant", "Omzet"]
        worksheet = ensure_worksheet(spreadsheet, "Omzet2025", headers)

        # Clear and rewrite
        worksheet.clear()

        if omzet_data:
            data = [headers] + [[k, v] for k, v in omzet_data.items()]
        else:
            data = [headers]

        worksheet.update('A1', data)
        # Clear cache after saving
        load_omzet_2025_from_gsheets.clear()
        return True
    except Exception as e:
        st.sidebar.error(f"Omzet 2025 opslaan mislukt: {e}")
        return False


# =============================================================================
# HELPER: Check if Google Sheets is available
# =============================================================================

def is_gsheets_available() -> bool:
    """Check if Google Sheets credentials are configured"""
    try:
        return "gcp_service_account" in st.secrets
    except Exception:
        return False
