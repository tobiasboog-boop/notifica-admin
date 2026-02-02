"""
Configuration for Voorraad Dashboard
"""
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class AppConfig:
    """Application configuration."""
    app_name: str = "Voorraad Dashboard"
    version: str = "0.1.0"
    debug: bool = False

    # Database settings
    db_host: str = os.getenv("DB_HOST", "10.3.152.9")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "TQwSTtLM9bSaLD")

    # Cache settings
    cache_ttl: int = 300  # 5 minutes


# Color scheme (consistent with liquiditeit dashboard)
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

# Voorraad thresholds
VOORRAAD_THRESHOLDS = {
    "omloopsnelheid_laag": 2.0,      # Lager dan 2x per jaar = traag lopend
    "omloopsnelheid_goed": 4.0,      # Hoger dan 4x = goed
    "dagen_voorraad_hoog": 180,      # Meer dan 180 dagen = te veel
    "dagen_voorraad_laag": 30,       # Minder dan 30 dagen = let op
    "kritiek_threshold": 10,          # Meer dan 10 posities onder minimum = kritiek
}
