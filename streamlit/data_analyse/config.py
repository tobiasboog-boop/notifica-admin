"""
Configuration for Data Analyse Dashboard
"""
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class AppConfig:
    """Application configuration."""
    app_name: str = "Data Analyse"
    version: str = "0.1.0"
    debug: bool = False

    # Default export path (can be overridden in UI)
    default_export_path: str = os.getenv(
        "EXPORT_PATH",
        r"C:\Users\tobia\OneDrive - Notifica B.V\Documenten - Sharepoint Notifica intern\106. Development"
    )


# Color scheme (consistent with other dashboards)
COLORS = {
    "primary": "#16136F",
    "primary_light": "#3636A2",
    "secondary": "#6c757d",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "light": "#f8f9fa",
    "dark": "#2c3e50",
}

# Entity type configuration
ENTITY_TYPES = {
    "AT_RELATIE": {
        "label": "Relaties",
        "icon": "🏢",
        "description": "Klant/leverancier informatie",
        "file_types": ["GC_INFORMATIE", "MELDING"],
    },
    "AT_DOCUMENT": {
        "label": "Documenten",
        "icon": "📄",
        "description": "Document notities",
        "file_types": ["GC_INFORMATIE"],
    },
    "AT_WERK": {
        "label": "Werkorders",
        "icon": "🔧",
        "description": "Werkorder notities",
        "file_types": ["GC_INFORMATIE"],
    },
    "AT_GEBOUW": {
        "label": "Gebouwen",
        "icon": "🏠",
        "description": "Locatie informatie",
        "file_types": ["GC_INFORMATIE"],
    },
    "AT_PERSOON": {
        "label": "Personen",
        "icon": "👤",
        "description": "Persoonsgegevens, foto's, handtekeningen",
        "file_types": ["AFBEELDING", "AFBEELDING_ID", "HANDTEKENING", "NOTITIE_VERTROUW"],
    },
    "AT_APPARAAT": {
        "label": "Apparaten",
        "icon": "⚙️",
        "description": "Apparaat informatie",
        "file_types": ["GC_INFORMATIE"],
    },
}

# Keywords for analysis
ANALYSIS_KEYWORDS = [
    "contract", "onderhoud", "factuur", "betaling", "credit", "storing",
    "reparatie", "offerte", "klacht", "defect", "garantie", "annuleren",
    "opzeggen", "korting", "telefonisch", "email", "brief", "bezoek",
    "franco", "ordernummer", "niet welkom", "blokkade"
]
