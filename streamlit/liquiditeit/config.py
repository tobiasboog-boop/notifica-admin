"""
Liquiditeit Dashboard - Configuration
=====================================
Configuratie voor database connecties en app settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class DatabaseConfig:
    """
    Database connection configuration for Syntess DWH.

    The Notifica DWH uses customer-specific databases:
    - Database name = klantnummer (bijv. 1241, 1209, etc.)

    Environment variables:
    - SYNTESS_DB_HOST: Database server hostname
    - SYNTESS_DB_PORT: Database port (default: 5432)
    - SYNTESS_DB_NAME: Database name (or use customer_code)
    - SYNTESS_DB_USER: Database username
    - SYNTESS_DB_PASSWORD: Database password
    """
    host: str
    port: int
    database: str
    username: str
    password: str

    @classmethod
    def from_env(cls, customer_code: Optional[str] = None) -> "DatabaseConfig":
        """
        Load database config from environment variables.

        Args:
            customer_code: Optional 4-digit customer code. If provided,
                          database name will be set to the customer code directly.
        """
        database = os.getenv("SYNTESS_DB_NAME", "1241")
        if customer_code and len(customer_code) == 4 and customer_code.isdigit():
            # Database name = klantnummer direct (niet dwh_XXXX)
            database = customer_code

        return cls(
            host=os.getenv("SYNTESS_DB_HOST", "10.3.152.9"),
            port=int(os.getenv("SYNTESS_DB_PORT", "5432")),
            database=database,
            username=os.getenv("SYNTESS_DB_USER", "postgres"),
            password=os.getenv("SYNTESS_DB_PASSWORD", "TQwSTtLM9bSaLD"),
        )

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class AppConfig:
    """Application configuration."""
    app_title: str = "Liquiditeit Dashboard"
    app_icon: str = "💰"
    default_forecast_weeks: int = 13
    default_history_months: int = 12

    # Cashflow categorieën (standaard classificatie)
    cashflow_categories: dict = None

    def __post_init__(self):
        if self.cashflow_categories is None:
            self.cashflow_categories = {
                "inkomend": {
                    "debiteuren": "Openstaande debiteuren",
                    "orderintake": "Verwachte orderintake",
                    "overig_in": "Overige inkomsten",
                },
                "uitgaand": {
                    "crediteuren": "Openstaande crediteuren",
                    "salarissen": "Salarisbetalingen",
                    "btw": "BTW afdracht",
                    "overig_uit": "Overige uitgaven",
                },
                "saldo": {
                    "bank": "Banksaldo",
                    "kas": "Kassaldo",
                },
            }


# Default thresholds for alerts
LIQUIDITY_THRESHOLDS = {
    "current_ratio_warning": 1.5,
    "current_ratio_danger": 1.0,
    "quick_ratio_warning": 1.0,
    "quick_ratio_danger": 0.5,
    "min_cash_buffer_days": 30,
}

# Color scheme matching Notifica branding
COLORS = {
    "primary": "#1E3A5F",      # Dark blue
    "secondary": "#3498DB",    # Light blue
    "success": "#27AE60",      # Green
    "warning": "#F39C12",      # Orange
    "danger": "#E74C3C",       # Red
    "neutral": "#95A5A6",      # Gray
}
