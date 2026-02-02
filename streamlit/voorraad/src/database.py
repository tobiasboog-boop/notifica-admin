"""
Database connections and queries for Voorraad Dashboard.

Based on Syntess DWH tables:
- Voorraad magazijnen
- Voorraad locaties
- Voorraad posities
- Voorraad mutaties
"""

import pandas as pd
from datetime import date, timedelta
from typing import Optional
from dataclasses import dataclass
import os

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "10.3.152.9"
    port: int = 5432
    database: str = "dwh"
    username: str = "postgres"
    password: str = ""

    @classmethod
    def from_secrets(cls):
        """Load config from Streamlit secrets (for Streamlit Cloud deployment)."""
        if STREAMLIT_AVAILABLE and hasattr(st, 'secrets') and 'database' in st.secrets:
            db = st.secrets["database"]
            return cls(
                host=db.get("host", "10.3.152.9"),
                port=int(db.get("port", 5432)),
                database=db.get("database", "dwh"),
                username=db.get("user", "postgres"),
                password=db.get("password", ""),
            )
        return cls.from_env()

    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "10.3.152.9"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "dwh"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )


# =============================================================================
# SQL QUERIES
# =============================================================================

QUERY_MAGAZIJNEN = """
-- Voorraad magazijnen met aggregaties
SELECT
    m."MagazijnKey" as magazijn_key,
    m."Magazijn" as magazijn,
    m."Projectmagazijn" as projectmagazijn,
    m."MagazijnGeblokkeerd" as geblokkeerd,
    m."Status" as status,
    a."Administratie" as administratie
FROM voorraad."Magazijnen" m
LEFT JOIN notifica."SSM Administraties" a ON m."AdministratieKey" = a."AdministratieKey"
WHERE TRIM(m."Status") = 'Actueel'
ORDER BY m."Magazijn"
"""

QUERY_LOCATIES = """
-- Voorraad locaties met hiërarchie
SELECT
    l."VoorraadlocatieKey" as locatie_key,
    l."Voorraadlocatie" as locatie,
    l."Voorraadlocatie code" as locatie_code,
    l."ParentVoorraadlocatieKey" as parent_key,
    l."MagazijnKey" as magazijn_key,
    m."Magazijn" as magazijn,
    l."Status" as status
FROM voorraad."Locaties" l
LEFT JOIN voorraad."Magazijnen" m ON l."MagazijnKey" = m."MagazijnKey"
WHERE TRIM(l."Status") = 'Actueel'
ORDER BY m."Magazijn", l."Voorraadlocatie"
"""

QUERY_POSITIES = """
-- Voorraad posities met min/max en berekende stand
SELECT
    p."VoorraadpositieKey" as positie_key,
    p."TariefKey" as tarief_key,
    t."Tarief" as artikel,
    p."voorraadlocatieKey" as locatie_key,
    l."Voorraadlocatie" as locatie,
    m."Magazijn" as magazijn,
    m."Projectmagazijn" as projectmagazijn,
    p."MinimumVoorraad" as minimum,
    p."MaximumVoorraad" as maximum,
    COALESCE(ontvangst.aantal, 0) as ontvangen,
    COALESCE(uitgifte.aantal, 0) as uitgegeven,
    COALESCE(ontvangst.aantal, 0) - COALESCE(uitgifte.aantal, 0) as stand,
    COALESCE(ontvangst.waarde, 0) - COALESCE(uitgifte.waarde, 0) as waarde,
    p."Status" as status
FROM voorraad."Posities" p
LEFT JOIN voorraad."Locaties" l ON p."voorraadlocatieKey" = l."VoorraadlocatieKey"
LEFT JOIN voorraad."Magazijnen" m ON p."MagazijnKey" = m."MagazijnKey"
LEFT JOIN stam."Tarieven" t ON p."TariefKey" = t."TariefKey"
-- Ontvangsten (via actieve relatie)
LEFT JOIN (
    SELECT
        "OntvangstVoorraadpositieKey" as positie_key,
        SUM("Aantal") as aantal,
        SUM("Kostprijs") as waarde
    FROM voorraad."Mutaties"
    GROUP BY "OntvangstVoorraadpositieKey"
) ontvangst ON p."VoorraadpositieKey" = ontvangst.positie_key
-- Uitgiftes (via inactieve relatie)
LEFT JOIN (
    SELECT
        "UitgifteVoorraadpositieKey" as positie_key,
        SUM("Aantal") as aantal,
        SUM("Kostprijs") as waarde
    FROM voorraad."Mutaties"
    GROUP BY "UitgifteVoorraadpositieKey"
) uitgifte ON p."VoorraadpositieKey" = uitgifte.positie_key
WHERE TRIM(p."Status") = 'Actueel'
ORDER BY m."Magazijn", l."Voorraadlocatie", t."Tarief"
"""

QUERY_MUTATIES = """
-- Voorraad mutaties met details
SELECT
    mut."OntvangstVoorraadpositieKey" as ontvangst_positie_key,
    mut."UitgifteVoorraadpositieKey" as uitgifte_positie_key,
    mut."TariefKey" as tarief_key,
    mut."Aantal" as aantal,
    mut."Kostprijs" as kostprijs,
    mut."Verrekenprijs" as verrekenprijs,
    mut."Boekdatum" as boekdatum,
    mut."Dagboek" as dagboek,
    mut."Bestelnummer" as bestelnummer,
    CASE
        WHEN mut."OntvangstVoorraadpositieKey" IS NOT NULL
             AND mut."UitgifteVoorraadpositieKey" IS NULL THEN mut."Aantal"
        ELSE 0
    END as ontvangst_aantal,
    CASE
        WHEN mut."UitgifteVoorraadpositieKey" IS NOT NULL
             AND mut."OntvangstVoorraadpositieKey" IS NULL THEN mut."Aantal"
        ELSE 0
    END as uitgifte_aantal,
    CASE
        WHEN mut."OntvangstVoorraadpositieKey" IS NOT NULL
             AND mut."UitgifteVoorraadpositieKey" IS NULL THEN 'Ontvangst'
        WHEN mut."UitgifteVoorraadpositieKey" IS NOT NULL
             AND mut."OntvangstVoorraadpositieKey" IS NULL THEN 'Uitgifte'
        WHEN mut."OntvangstVoorraadpositieKey" IS NOT NULL
             AND mut."UitgifteVoorraadpositieKey" IS NOT NULL THEN 'Transfer'
        ELSE 'Onbekend'
    END as mutatiesoort
FROM voorraad."Mutaties" mut
WHERE mut."Boekdatum" >= CURRENT_DATE - INTERVAL '12 months'
ORDER BY mut."Boekdatum" DESC
"""

# Query voor tarieven/artikelen met prijzen
QUERY_TARIEVEN = """
-- Tarieven met prijsinformatie voor P x Q validatie
SELECT
    t."TariefKey" as tarief_key,
    t."Tarief Code" as tarief_code,
    t."Tarief" as tarief,
    t."Categorie" as categorie,
    t."Verrekenprijs" as verrekenprijs,
    t."Bruto Verkoopprijs" as bruto_verkoopprijs,
    t."Consumentenprijs" as consumentenprijs,
    t."PrijsAantal" as prijs_aantal,
    t."Status" as status,
    t."Handelsartikel" as handelsartikel
FROM stam."Tarieven" t
WHERE TRIM(t."Status") = 'Actueel'
ORDER BY t."Tarief"
"""

# Query voor balanswaarde voorraad (uit balans via Rubrieken)
# Alleen rubriek 3000 (fysieke voorraden) voor eerlijke vergelijking
# 3002 (Correctie) en 3210 (Showroom) zijn aparte boekhoudkundige posten
# Let op: waarde is gebaseerd op DWH refresh datum, niet real-time
QUERY_BALANS_VOORRAAD = """
-- Voorraadwaarde uit balans (alleen rubriek 3000 = fysieke voorraden)
-- Dit moet aansluiten op de berekende voorraadwaarde (stand × kostprijs)
WITH recent_year AS (
    SELECT MAX(EXTRACT(YEAR FROM jr2."Boekdatum")) as jaar
    FROM financieel."Journaalregels" jr2
    JOIN financieel."Rubrieken" rub2 ON jr2."RubriekKey" = rub2."RubriekKey"
    WHERE rub2."Type" = 'B'
      AND rub2."Rubriek Code" = '3000'
)
SELECT
    SUM(
        CASE WHEN jr."Debet/Credit" = 'D' THEN jr."Bedrag" ELSE 0 END
        - CASE WHEN jr."Debet/Credit" = 'C' THEN jr."Bedrag" ELSE 0 END
    ) as balans_waarde
FROM financieel."Journaalregels" jr
JOIN financieel."Rubrieken" rub ON jr."RubriekKey" = rub."RubriekKey"
CROSS JOIN recent_year ry
WHERE rub."Type" = 'B'
  AND rub."Rubriek Code" = '3000'  -- Alleen fysieke voorraden
  AND EXTRACT(YEAR FROM jr."Boekdatum") = ry.jaar
"""

# Query voor voorraadmutaties per artikel per periode (voor ligduur berekening)
QUERY_MUTATIES_DETAIL = """
-- Gedetailleerde mutaties voor analyse per artikel
SELECT
    p."TariefKey" as tarief_key,
    t."Tarief" as artikel,
    mut."Boekdatum" as boekdatum,
    mut."Aantal" as aantal,
    mut."Kostprijs" as kostprijs,
    CASE
        WHEN mut."OntvangstVoorraadpositieKey" = p."VoorraadpositieKey" THEN 'Ontvangst'
        WHEN mut."UitgifteVoorraadpositieKey" = p."VoorraadpositieKey" THEN 'Uitgifte'
        ELSE 'Anders'
    END as richting
FROM voorraad."Mutaties" mut
JOIN voorraad."Posities" p ON
    p."VoorraadpositieKey" = mut."OntvangstVoorraadpositieKey" OR
    p."VoorraadpositieKey" = mut."UitgifteVoorraadpositieKey"
LEFT JOIN stam."Tarieven" t ON p."TariefKey" = t."TariefKey"
WHERE mut."Boekdatum" >= CURRENT_DATE - INTERVAL '12 months'
ORDER BY t."Tarief", mut."Boekdatum"
"""


class SyntessDWHConnection:
    """Database connection for Syntess DWH."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 required")

        conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
        )
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def get_magazijnen(self, administratie: str = None) -> pd.DataFrame:
        """Get all warehouses."""
        try:
            return self.execute_query(QUERY_MAGAZIJNEN)
        except Exception as e:
            print(f"Error fetching magazijnen: {e}")
            return pd.DataFrame()

    def get_locaties(self, magazijn_key: int = None) -> pd.DataFrame:
        """Get all locations."""
        try:
            return self.execute_query(QUERY_LOCATIES)
        except Exception as e:
            print(f"Error fetching locaties: {e}")
            return pd.DataFrame()

    def get_posities(self, administratie: str = None) -> pd.DataFrame:
        """Get all stock positions with calculated stand."""
        try:
            df = self.execute_query(QUERY_POSITIES)
            # Calculate min-max status
            if not df.empty and "stand" in df.columns:
                # Ensure numeric columns
                df["stand"] = pd.to_numeric(df["stand"], errors="coerce").fillna(0)
                df["minimum"] = pd.to_numeric(df.get("minimum", 0), errors="coerce").fillna(0)
                df["maximum"] = pd.to_numeric(df.get("maximum", 0), errors="coerce").fillna(0)

                df["tekort"] = df.apply(
                    lambda r: max(0, r["minimum"] - r["stand"]) if r["minimum"] > 0 else 0,
                    axis=1
                )
                df["status"] = df.apply(self._calculate_status, axis=1)
            elif not df.empty:
                # Query returned data but without expected columns - add empty status
                df["tekort"] = 0
                df["status"] = "Onbekend"
            return df
        except Exception as e:
            print(f"Error fetching posities: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_mutaties(self, startdatum: date = None, einddatum: date = None) -> pd.DataFrame:
        """Get stock mutations."""
        try:
            return self.execute_query(QUERY_MUTATIES)
        except Exception as e:
            print(f"Error fetching mutaties: {e}")
            return pd.DataFrame()

    def get_tarieven(self) -> pd.DataFrame:
        """Get tariff/article price information."""
        try:
            return self.execute_query(QUERY_TARIEVEN)
        except Exception as e:
            print(f"Error fetching tarieven: {e}")
            return pd.DataFrame()

    def get_balans_voorraad(self) -> float:
        """Get inventory value from balance sheet (grootboek).

        Returns the total value of inventory accounts from the general ledger.
        This can be compared with calculated stock value for reconciliation.
        """
        try:
            df = self.execute_query(QUERY_BALANS_VOORRAAD)
            if not df.empty and "balans_waarde" in df.columns:
                return float(df["balans_waarde"].iloc[0] or 0)
            return 0.0
        except Exception as e:
            print(f"Error fetching balans voorraad: {e}")
            return 0.0

    def get_mutaties_detail(self) -> pd.DataFrame:
        """Get detailed mutations per article for turnover analysis."""
        try:
            return self.execute_query(QUERY_MUTATIES_DETAIL)
        except Exception as e:
            print(f"Error fetching mutaties detail: {e}")
            return pd.DataFrame()

    @staticmethod
    def _calculate_status(row) -> str:
        """Calculate min-max status for a position."""
        stand = row.get("stand", 0) or 0
        minimum = row.get("minimum", 0) or 0
        maximum = row.get("maximum", 0) or 0

        if stand <= 0:
            return "Geen voorraad"
        elif minimum > 0 and stand < minimum:
            return "Onder minimum"
        elif maximum > 0 and stand > maximum:
            return "Boven maximum"
        else:
            return "OK"


class MockDatabase:
    """Mock database for demo/development."""

    def __init__(self):
        import numpy as np
        self.np = np

    def get_magazijnen(self) -> pd.DataFrame:
        """Generate mock warehouse data."""
        return pd.DataFrame({
            "magazijn_key": [1, 2, 3, 4],
            "magazijn": ["Hoofdmagazijn", "Buitenmagazijn", "Project A", "Project B"],
            "magazijn_code": ["HM", "BM", "PA", "PB"],
            "projectmagazijn": ["Nee", "Nee", "Ja", "Ja"],
            "geblokkeerd": ["Nee", "Nee", "Nee", "Nee"],
            "status": ["Actueel", "Actueel", "Actueel", "Actueel"],
            "administratie": ["Demo BV", "Demo BV", "Demo BV", "Demo BV"],
            "waarde": [450000, 125000, 85000, 42000],
            "aantal": [8500, 2100, 1200, 650],
        })

    def get_locaties(self) -> pd.DataFrame:
        """Generate mock location data."""
        return pd.DataFrame({
            "locatie_key": [1, 2, 3, 4, 5],
            "locatie": ["Stelling A", "Stelling B", "Stelling C", "Buitenvak 1", "Buitenvak 2"],
            "locatie_code": ["A", "B", "C", "BV1", "BV2"],
            "parent_key": [None, None, None, None, None],
            "magazijn_key": [1, 1, 1, 2, 2],
            "magazijn": ["Hoofdmagazijn", "Hoofdmagazijn", "Hoofdmagazijn", "Buitenmagazijn", "Buitenmagazijn"],
            "status": ["Ja", "Ja", "Ja", "Ja", "Ja"],
        })

    def get_posities(self) -> pd.DataFrame:
        """Generate mock position data with min-max."""
        np = self.np
        n = 150

        # Generate realistic data
        artikelen = [f"Artikel {i:03d}" for i in range(1, n+1)]
        locaties = ["Stelling A", "Stelling B", "Stelling C", "Buitenvak 1", "Buitenvak 2"]
        magazijnen = ["Hoofdmagazijn", "Hoofdmagazijn", "Hoofdmagazijn", "Buitenmagazijn", "Buitenmagazijn"]

        df = pd.DataFrame({
            "positie_key": range(1, n+1),
            "tarief_key": range(1001, 1001+n),
            "artikel": artikelen,
            "locatie_key": np.random.randint(1, 6, n),
            "minimum": np.random.choice([0, 5, 10, 20, 50], n, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
            "maximum": np.random.choice([0, 50, 100, 200, 500], n, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
            "stand": np.random.randint(-5, 150, n),  # Some negative for testing
            "waarde": np.random.uniform(10, 5000, n).round(2),
            "administratie": ["Demo BV"] * n,
        })

        # Assign locatie/magazijn based on locatie_key
        df["locatie"] = df["locatie_key"].map(dict(enumerate(locaties, 1)))
        df["magazijn"] = df["locatie_key"].map(dict(enumerate(magazijnen, 1)))
        df["projectmagazijn"] = "Nee"

        # Calculate tekort and status
        df["tekort"] = df.apply(
            lambda r: max(0, r["minimum"] - r["stand"]) if r["minimum"] > 0 else 0,
            axis=1
        )
        df["status"] = df.apply(
            lambda r: "Geen voorraad" if r["stand"] <= 0
            else "Onder minimum" if r["minimum"] > 0 and r["stand"] < r["minimum"]
            else "Boven maximum" if r["maximum"] > 0 and r["stand"] > r["maximum"]
            else "OK",
            axis=1
        )

        return df

    def get_mutaties(self) -> pd.DataFrame:
        """Generate mock mutation data."""
        np = self.np
        n = 500

        dates = pd.date_range(end=date.today(), periods=365, freq="D")

        df = pd.DataFrame({
            "mutatie_key": range(1, n+1),
            "boekdatum": np.random.choice(dates, n),
            "tarief_key": np.random.randint(1001, 1151, n),
            "aantal": np.random.randint(1, 100, n),
            "kostprijs": np.random.uniform(5, 500, n).round(2),
            "mutatiesoort": np.random.choice(["Ontvangst", "Uitgifte"], n, p=[0.45, 0.55]),
        })

        df["ontvangst_aantal"] = df.apply(lambda r: r["aantal"] if r["mutatiesoort"] == "Ontvangst" else 0, axis=1)
        df["uitgifte_aantal"] = df.apply(lambda r: r["aantal"] if r["mutatiesoort"] == "Uitgifte" else 0, axis=1)

        return df.sort_values("boekdatum", ascending=False)

    def get_tarieven(self) -> pd.DataFrame:
        """Generate mock tariff/price data."""
        np = self.np
        n = 150

        return pd.DataFrame({
            "tarief_key": range(1001, 1001 + n),
            "tarief_code": [f"ART{i:04d}" for i in range(1, n + 1)],
            "tarief": [f"Artikel {i:03d}" for i in range(1, n + 1)],
            "categorie": np.random.choice(["Materiaal", "Arbeid", "Overig"], n, p=[0.7, 0.2, 0.1]),
            "verrekenprijs": np.random.uniform(5, 200, n).round(2),
            "bruto_verkoopprijs": np.random.uniform(10, 300, n).round(2),
            "consumentenprijs": np.random.uniform(15, 400, n).round(2),
            "prijs_aantal": [1] * n,
            "status": ["Actueel"] * n,
            "handelsartikel": [f"HA-{i:05d}" for i in range(1, n + 1)],
        })

    def get_balans_voorraad(self) -> float:
        """Return mock balance sheet inventory value."""
        # Should be close to but not exactly equal to calculated value
        # to demonstrate reconciliation
        return 695000.0  # Approximately sum of mock posities waarde

    def get_mutaties_detail(self) -> pd.DataFrame:
        """Generate detailed mock mutations for analysis."""
        return self.get_mutaties()  # Reuse standard mutaties for mock


class LocalDatabase:
    """Database that reads from local Parquet files (exported snapshots)."""

    def __init__(self, data_dir: str, customer_code: str = None):
        from pathlib import Path
        import json

        self.data_dir = Path(data_dir)
        if customer_code:
            self.data_dir = self.data_dir / customer_code

        # Load export info if available
        self.export_info = {}
        info_path = self.data_dir / "export_info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.export_info = json.load(f)

    def _load_parquet(self, name: str) -> pd.DataFrame:
        """Load a parquet file."""
        filepath = self.data_dir / f"{name}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return pd.DataFrame()

    def get_magazijnen(self) -> pd.DataFrame:
        return self._load_parquet("magazijnen")

    def get_locaties(self) -> pd.DataFrame:
        return self._load_parquet("locaties")

    def get_posities(self) -> pd.DataFrame:
        df = self._load_parquet("posities")
        if not df.empty and "stand" in df.columns:
            # Recalculate status if needed
            if "status" not in df.columns:
                df["stand"] = pd.to_numeric(df["stand"], errors="coerce").fillna(0)
                df["minimum"] = pd.to_numeric(df.get("minimum", 0), errors="coerce").fillna(0)
                df["maximum"] = pd.to_numeric(df.get("maximum", 0), errors="coerce").fillna(0)
                df["tekort"] = df.apply(
                    lambda r: max(0, r["minimum"] - r["stand"]) if r["minimum"] > 0 else 0,
                    axis=1
                )
                df["status"] = df.apply(self._calculate_status, axis=1)
        return df

    def get_mutaties(self) -> pd.DataFrame:
        return self._load_parquet("mutaties")

    def get_tarieven(self) -> pd.DataFrame:
        return self._load_parquet("tarieven")

    def get_balans_voorraad(self) -> float:
        return self.export_info.get("balans_waarde", 0.0)

    def get_mutaties_detail(self) -> pd.DataFrame:
        return self.get_mutaties()

    def get_export_date(self) -> str:
        """Return the date when data was exported."""
        return self.export_info.get("export_date", "Onbekend")

    @staticmethod
    def _calculate_status(row) -> str:
        stand = row.get("stand", 0) or 0
        minimum = row.get("minimum", 0) or 0
        maximum = row.get("maximum", 0) or 0
        if stand <= 0:
            return "Geen voorraad"
        elif minimum > 0 and stand < minimum:
            return "Onder minimum"
        elif maximum > 0 and stand > maximum:
            return "Boven maximum"
        else:
            return "OK"


def get_local_data_path() -> str:
    """Find path to local data directory."""
    from pathlib import Path

    # Check various locations
    possible_paths = [
        Path(__file__).parent.parent / "data",  # streamlit/voorraad/data
        Path.cwd() / "data",                     # current dir / data
        Path.cwd() / "streamlit" / "voorraad" / "data",  # from repo root
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob("*/*.parquet")):
            return str(path)
        if path.exists() and any(path.glob("*.parquet")):
            return str(path)

    return None


def get_local_customer_code() -> tuple:
    """Get customer code and name from local data folder.

    Returns (customer_code, customer_name) or (None, None) if not found.
    """
    from pathlib import Path
    import json

    local_path = get_local_data_path()
    if not local_path:
        return None, None

    local_path = Path(local_path)

    # Find customer folders (folders with parquet files or export_info.json)
    for folder in local_path.iterdir():
        if folder.is_dir():
            info_file = folder / "export_info.json"
            if info_file.exists():
                try:
                    with open(info_file) as f:
                        info = json.load(f)
                    customer_code = info.get("customer_code", folder.name)
                    # Try to get name from info, otherwise use code
                    return customer_code, None
                except:
                    return folder.name, None
            # Check if folder has parquet files
            if any(folder.glob("*.parquet")):
                return folder.name, None

    return None, None


def get_database(customer_code: str = None):
    """Get appropriate database connection.

    Tries to connect to data in this order:
    1. Local Parquet files (for offline/cloud deployment)
    2. Live database via Streamlit secrets
    3. Live database via environment variables
    4. Falls back to MockDatabase for demo
    """
    # 1. Check for local data first (preferred for cloud deployment)
    local_path = get_local_data_path()
    if local_path:
        try:
            local_db = LocalDatabase(local_path, customer_code)
            # Verify data exists
            test_df = local_db.get_posities()
            if not test_df.empty:
                print(f"Using local data from: {local_path}")
                return local_db
        except Exception as e:
            print(f"Local data load failed: {e}")

    # 2. Try live database connection
    if PSYCOPG2_AVAILABLE:
        try:
            config = DatabaseConfig.from_secrets()
            if customer_code:
                config.database = customer_code
            # Test connection
            conn = SyntessDWHConnection(config)
            # Quick test query
            test_df = conn.execute_query("SELECT 1 as test")
            if not test_df.empty:
                return conn
        except Exception as e:
            print(f"Database connection failed: {e}")

    # 3. Fallback to mock data
    return MockDatabase()
