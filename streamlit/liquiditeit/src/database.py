"""
Liquiditeit Dashboard - Database Connection
============================================
Module voor database connecties naar Syntess DWH.
"""

import pandas as pd
from contextlib import contextmanager
from datetime import date
from typing import Optional, Generator, Union
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig

# Optional psycopg2 import (not needed for mock mode)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# =============================================================================
# SQL Queries for Syntess DWH
# =============================================================================
# GEBASEERD OP SEMANTISCH MODEL DAX FORMULES:
# - Debiteuren: SUM('Verkoopfactuur termijnen'[Bedrag]) WHERE Status = 'Openstaand' AND Alloc_datum <= standdatum
# - Crediteuren: SUM('Inkoopfactuur termijnen'[Bedrag]) WHERE Bankafschrift status = 'Openstaand' AND Alloc_datum <= standdatum
# - Bank: SUM(Journaalregels[Debetbedrag]) - SUM(Journaalregels[Creditbedrag]) WHERE Bankafschriftboeking type = 'Bank' AND Ja/Nee = 'Ja'
# =============================================================================

QUERY_BANKSALDO = """
-- Banksaldi: Haal actueel saldo per bankrekening via grootboekrekeningen
-- Dit omzeilt de 3-jaar filter in SSM Journaalregels door direct naar financieel schema te gaan
-- Filter: alleen boekingen OP de bankrekening zelf (RubriekKey = DagboekRubriekKey)
SELECT
    dag."Dagboek" as bank_naam,
    dag."Dagboek" as rekeningnummer,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as saldo,
    CURRENT_DATE as datum
FROM financieel."Journaalregels" j
JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
WHERE j."Boekdatum" <= %(standdatum)s
  AND d."StandaardEntiteitKey" = 10  -- Bank documenten
  AND j."RubriekKey" = dag."DagboekRubriekKey"  -- Alleen boekingen OP de bankrekening zelf
GROUP BY dag."Dagboek"
HAVING ABS(SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
            SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END)) > 0.01
ORDER BY saldo DESC
"""

QUERY_OPENSTAANDE_DEBITEUREN = """
-- Openstaande debiteuren per debiteur (netto saldo)
-- DAX 'Openstaande verkoopfacturen': SUM('Verkoopfactuur termijnen'[Bedrag]) met Standdatum filter
-- De tabel bevat zowel factuurregels (positief) als betalingsregels (negatief)
-- Door te groeperen per debiteur krijgen we het netto openstaande saldo
SELECT
    vft."Debiteur" as debiteur_code,
    vft."Debiteur" as debiteur_naam,
    'Diverse' as factuurnummer,
    MAX(vft."Alloc_datum") as factuurdatum,
    MAX(vft."Vervaldatum") as vervaldatum,
    SUM(vft."Bedrag") as bedrag_excl_btw,
    NULL::date as betaaldatum,
    0 as betaald,
    SUM(vft."Bedrag") as openstaand,
    30 as betaaltermijn_dagen,
    COALESCE(a."Administratie", 'Onbekend') as administratie,
    COALESCE(be."Bedrijfseenheid", 'Onbekend') as bedrijfseenheid
FROM notifica."SSM Verkoopfactuur termijnen" vft
LEFT JOIN notifica."SSM Documenten" d ON vft."VerkoopfactuurDocumentKey" = d."DocumentKey"
LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
WHERE vft."Alloc_datum" <= %(standdatum)s
GROUP BY vft."Debiteur", a."Administratie", be."Bedrijfseenheid"
HAVING ABS(SUM(vft."Bedrag")) > 0.01
ORDER BY SUM(vft."Bedrag") DESC
"""

QUERY_OPENSTAANDE_CREDITEUREN = """
-- Openstaande crediteuren via SSM Inkoopfactuur termijnen
-- DAX: SUM('Inkoopfactuur termijnen'[Bedrag]) WHERE Bankafschrift status = 'Openstaand' AND Alloc_datum <= standdatum
SELECT
    ift."Crediteur" as crediteur_code,
    ift."Crediteur" as crediteur_naam,
    COALESCE(d."Document code", CAST(ift."InkoopFactuurKey" AS TEXT)) as factuurnummer,
    ift."Alloc_datum" as factuurdatum,
    ift."Vervaldatum" as vervaldatum,
    ift."Bedrag" as bedrag_excl_btw,
    NULL::date as betaaldatum,
    0 as betaald,
    ift."Bedrag" as openstaand,
    COALESCE(a."Administratie", 'Onbekend') as administratie,
    COALESCE(be."Bedrijfseenheid", 'Onbekend') as bedrijfseenheid
FROM notifica."SSM Inkoopfactuur termijnen" ift
LEFT JOIN notifica."SSM Documenten" d ON ift."InkoopFactuurKey" = d."DocumentKey"
LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
WHERE ift."Alloc_datum" <= %(standdatum)s
  AND ift."Bankafschrift status" = 'Openstaand'
ORDER BY ift."Vervaldatum" ASC
"""

QUERY_HISTORISCH_BETALINGSGEDRAG = """
-- Historisch betalingsgedrag: maandelijkse bank mutaties (laatste 12 maanden)
-- Dit geeft inzicht in typische cashflow patronen
-- Direct naar financieel schema om 3-jaar filter te omzeilen
SELECT
    DATE_TRUNC('month', j."Boekdatum") as maand,
    30 as gem_betaaltermijn_debiteuren,
    30 as gem_betaaltermijn_crediteuren,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as inkomsten,
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as uitgaven
FROM financieel."Journaalregels" j
JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
WHERE j."Boekdatum" >= CURRENT_DATE - INTERVAL '12 months'
  AND d."StandaardEntiteitKey" = 10  -- Bank documenten
  AND j."RubriekKey" = dag."DagboekRubriekKey"  -- Alleen boekingen OP de bankrekening zelf
GROUP BY DATE_TRUNC('month', j."Boekdatum")
ORDER BY maand DESC
"""

QUERY_VOORZIENING_DEBITEUREN = """
-- Voorziening debiteuren: netto beweging op rubriek 1230 in het boekjaar van de standdatum
-- Power BI toont dit als aparte regel die van bruto debiteuren wordt afgetrokken
-- De voorziening is het verschil tussen Debet en Credit boekingen op rubriek 1230
-- INCLUSIEF jaarovergang boekingen aan het begin van het jaar (zoals Power BI doet)
-- Voor standdatum 2025-12-31 telt dit boekingen van 2024-01-01 t/m 2024-12-31
SELECT
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as voorziening
FROM financieel."Journaalregels" j
JOIN financieel."Rubrieken" rub ON j."RubriekKey" = rub."RubriekKey"
JOIN notifica."SSM Administraties" a ON j."AdministratieKey" = a."AdministratieKey"
WHERE j."Boekdatum" BETWEEN
      DATE_TRUNC('year', %(standdatum)s::date) - INTERVAL '1 year'
      AND DATE_TRUNC('year', %(standdatum)s::date) - INTERVAL '1 day'
  AND rub."Rubriek Code" = '1230'
  AND a."Administratie" = %(administratie)s
"""

QUERY_TERUGKERENDE_KOSTEN = """
-- Terugkerende kosten per maand (salarissen, huur, etc.)
-- Analyseert historische boekingen op specifieke kostenrubrieken om patronen te detecteren
-- Rubrieken 4xxx = personeelskosten (salaris), 6xxx = huisvestingskosten
SELECT
    DATE_TRUNC('month', j."Boekdatum") as maand,
    rub."Rubriek Code" as rubriek_code,
    rub."Rubriek" as rubriek_naam,
    CASE
        WHEN rub."Rubriek Code" LIKE '4%%' THEN 'Personeelskosten'
        WHEN rub."Rubriek Code" LIKE '61%%' THEN 'Huisvestingskosten'
        WHEN rub."Rubriek Code" LIKE '62%%' THEN 'Machinekosten'
        WHEN rub."Rubriek Code" LIKE '65%%' THEN 'Autokosten'
        ELSE 'Overige vaste kosten'
    END as kostensoort,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as bedrag
FROM financieel."Journaalregels" j
JOIN financieel."Rubrieken" rub ON j."RubriekKey" = rub."RubriekKey"
JOIN notifica."SSM Administraties" a ON j."AdministratieKey" = a."AdministratieKey"
WHERE j."Boekdatum" >= %(startdatum)s
  AND j."Boekdatum" < %(einddatum)s
  AND (rub."Rubriek Code" LIKE '4%%'   -- Personeelskosten
       OR rub."Rubriek Code" LIKE '61%%'  -- Huisvestingskosten
       OR rub."Rubriek Code" LIKE '62%%'  -- Machinekosten
       OR rub."Rubriek Code" LIKE '65%%') -- Autokosten
  AND a."Administratie" = %(administratie)s
GROUP BY DATE_TRUNC('month', j."Boekdatum"), rub."Rubriek Code", rub."Rubriek"
ORDER BY maand, rubriek_code
"""

QUERY_HISTORISCHE_CASHFLOW_PER_WEEK = """
-- Historische cashflow per week voor seasonality analyse
-- Gebruikt bankmutaties om het echte in/uit patroon te zien
SELECT
    DATE_TRUNC('week', j."Boekdatum") as week_start,
    EXTRACT(WEEK FROM j."Boekdatum") as week_nummer,
    EXTRACT(MONTH FROM j."Boekdatum") as maand,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as inkomsten,
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as uitgaven,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as netto
FROM financieel."Journaalregels" j
JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
WHERE j."Boekdatum" >= %(startdatum)s
  AND j."Boekdatum" < %(einddatum)s
  AND d."StandaardEntiteitKey" = 10  -- Bank documenten
  AND j."RubriekKey" = dag."DagboekRubriekKey"  -- Alleen boekingen OP de bankrekening
  AND a."Administratie" = %(administratie)s
GROUP BY DATE_TRUNC('week', j."Boekdatum"),
         EXTRACT(WEEK FROM j."Boekdatum"),
         EXTRACT(MONTH FROM j."Boekdatum")
ORDER BY week_start
"""

QUERY_HISTORISCHE_CASHFLOW_PER_WEEK_BY_KEY = """
-- Historische cashflow per week voor seasonality analyse (filter op AdministratieKey)
-- Gebruikt bankmutaties om het echte in/uit patroon te zien
SELECT
    DATE_TRUNC('week', j."Boekdatum") as week_start,
    EXTRACT(WEEK FROM j."Boekdatum") as week_nummer,
    EXTRACT(MONTH FROM j."Boekdatum") as maand,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as inkomsten,
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as uitgaven,
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as netto
FROM financieel."Journaalregels" j
JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
WHERE j."Boekdatum" >= %(startdatum)s
  AND j."Boekdatum" < %(einddatum)s
  AND d."StandaardEntiteitKey" = 10  -- Bank documenten
  AND j."RubriekKey" = dag."DagboekRubriekKey"  -- Alleen boekingen OP de bankrekening
  AND dag."AdministratieKey" = %(administratie_key)s
GROUP BY DATE_TRUNC('week', j."Boekdatum"),
         EXTRACT(WEEK FROM j."Boekdatum"),
         EXTRACT(MONTH FROM j."Boekdatum")
ORDER BY week_start
"""


class SyntessDWHConnection:
    """
    Database connection specifically for Syntess DWH.
    Handles connections to customer-specific databases (dwh_XXXX format).
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self._connection = None

    @contextmanager
    def get_connection(self) -> Generator:
        """Context manager for database connections."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for database connections. Install with: pip install psycopg2-binary")

        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
            )
            yield conn
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def test_connection(self) -> bool:
        """Test if database connection works."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def get_banksaldo(self, standdatum: date = None, administratie: str = None) -> pd.DataFrame:
        """Get bank balances from DWH at a specific reference date.

        Args:
            standdatum: Reference date for balances
            administratie: Optional filter for specific administration (e.g., "Kronenburg Techniek B.V")
        """
        if standdatum is None:
            standdatum = date.today()

        # Use filtered query if administratie is specified
        if administratie:
            query = """
            SELECT
                dag."Dagboek" as bank_naam,
                dag."Dagboek" as rekeningnummer,
                SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
                SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as saldo,
                CURRENT_DATE as datum
            FROM financieel."Journaalregels" j
            JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
            JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
            JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
            WHERE j."Boekdatum" <= %(standdatum)s
              AND d."StandaardEntiteitKey" = 10
              AND j."RubriekKey" = dag."DagboekRubriekKey"
              AND a."Administratie" = %(administratie)s
            GROUP BY dag."Dagboek"
            HAVING ABS(SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
                        SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END)) > 0.01
            ORDER BY saldo DESC
            """
            params = {"standdatum": standdatum, "administratie": administratie}
        else:
            query = QUERY_BANKSALDO
            params = {"standdatum": standdatum}

        try:
            df = self.execute_query(query, params)
            if df.empty:
                # Fallback: return empty dataframe with correct columns
                return pd.DataFrame({
                    "bank_naam": [],
                    "rekeningnummer": [],
                    "saldo": [],
                    "datum": [],
                })
            return df
        except Exception as e:
            print(f"Error fetching bank balances: {e}")
            return pd.DataFrame({
                "bank_naam": [],
                "rekeningnummer": [],
                "saldo": [],
                "datum": [],
            })

    def get_voorziening_debiteuren(self, standdatum: date = None, administratie: str = None) -> float:
        """Get provision for doubtful debts from DWH at a specific reference date.

        Args:
            standdatum: Reference date for the provision
            administratie: Administration name (required for this query)

        Returns:
            The provision amount (negative value that should be added to receivables)
        """
        if standdatum is None:
            standdatum = date.today()
        if not administratie:
            return 0.0

        try:
            df = self.execute_query(
                QUERY_VOORZIENING_DEBITEUREN,
                {"standdatum": standdatum, "administratie": administratie}
            )
            if df.empty or df['voorziening'].iloc[0] is None:
                return 0.0
            return float(df['voorziening'].iloc[0])
        except Exception as e:
            print(f"Error fetching provision: {e}")
            return 0.0

    def get_openstaande_debiteuren(self, standdatum: date = None, administratie: str = None) -> pd.DataFrame:
        """Get outstanding receivables from DWH at a specific reference date.

        Args:
            standdatum: Reference date for receivables
            administratie: Optional filter for specific administration
        """
        if standdatum is None:
            standdatum = date.today()

        try:
            # Use filtered query if administratie is specified
            if administratie:
                query = """
                SELECT
                    vft."Debiteur" as debiteur_code,
                    vft."Debiteur" as debiteur_naam,
                    'Diverse' as factuurnummer,
                    MAX(vft."Alloc_datum") as factuurdatum,
                    MAX(vft."Vervaldatum") as vervaldatum,
                    SUM(vft."Bedrag") as bedrag_excl_btw,
                    NULL::date as betaaldatum,
                    0 as betaald,
                    SUM(vft."Bedrag") as openstaand,
                    30 as betaaltermijn_dagen,
                    COALESCE(a."Administratie", 'Onbekend') as administratie,
                    COALESCE(be."Bedrijfseenheid", 'Onbekend') as bedrijfseenheid
                FROM notifica."SSM Verkoopfactuur termijnen" vft
                LEFT JOIN notifica."SSM Documenten" d ON vft."VerkoopfactuurDocumentKey" = d."DocumentKey"
                LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
                LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
                WHERE vft."Alloc_datum" <= %(standdatum)s
                  AND a."Administratie" = %(administratie)s
                GROUP BY vft."Debiteur", a."Administratie", be."Bedrijfseenheid"
                HAVING ABS(SUM(vft."Bedrag")) > 0.01
                ORDER BY SUM(vft."Bedrag") DESC
                """
                df = self.execute_query(query, {"standdatum": standdatum, "administratie": administratie})
            else:
                df = self.execute_query(QUERY_OPENSTAANDE_DEBITEUREN, {"standdatum": standdatum})
            return df
        except Exception as e:
            print(f"Error fetching receivables: {e}")
            return pd.DataFrame({
                "debiteur_code": [],
                "debiteur_naam": [],
                "factuurnummer": [],
                "factuurdatum": [],
                "vervaldatum": [],
                "bedrag_excl_btw": [],
                "betaaldatum": [],
                "betaald": [],
                "openstaand": [],
                "betaaltermijn_dagen": [],
                "administratie": [],
                "bedrijfseenheid": [],
            })

    def get_openstaande_crediteuren(self, standdatum: date = None) -> pd.DataFrame:
        """Get outstanding payables from DWH at a specific reference date."""
        if standdatum is None:
            standdatum = date.today()
        try:
            df = self.execute_query(QUERY_OPENSTAANDE_CREDITEUREN, {"standdatum": standdatum})
            return df
        except Exception as e:
            print(f"Error fetching payables: {e}")
            return pd.DataFrame({
                "crediteur_code": [],
                "crediteur_naam": [],
                "factuurnummer": [],
                "factuurdatum": [],
                "vervaldatum": [],
                "bedrag_excl_btw": [],
                "betaaldatum": [],
                "betaald": [],
                "openstaand": [],
                "administratie": [],
                "bedrijfseenheid": [],
            })

    def get_geplande_salarissen(self) -> pd.DataFrame:
        """
        Get planned salary payments.
        Note: Syntess doesn't have a specific salary planning table,
        so we estimate based on historical patterns or return empty.
        """
        # TODO: This would need custom implementation per customer
        # For now, return empty - customers can add manual entries
        return pd.DataFrame({
            "betaaldatum": [],
            "omschrijving": [],
            "bedrag": [],
        })

    def get_historisch_betalingsgedrag(self) -> pd.DataFrame:
        """Get historical payment behavior from DWH."""
        try:
            df = self.execute_query(QUERY_HISTORISCH_BETALINGSGEDRAG)
            return df
        except Exception as e:
            print(f"Error fetching payment history: {e}")
            return pd.DataFrame({
                "maand": [],
                "gem_betaaltermijn_debiteuren": [],
                "gem_betaaltermijn_crediteuren": [],
                "inkomsten": [],
                "uitgaven": [],
            })

    def get_terugkerende_kosten(self, startdatum: date = None, einddatum: date = None, administratie: str = None) -> pd.DataFrame:
        """
        Get recurring costs (salaries, rent, etc.) from historical bookings.

        Args:
            startdatum: Start date for analysis (default: 12 months ago)
            einddatum: End date for analysis (default: today)
            administratie: Administration name (required)

        Returns:
            DataFrame with monthly recurring costs by category
        """
        if einddatum is None:
            einddatum = date.today()
        if startdatum is None:
            startdatum = date(einddatum.year - 1, einddatum.month, 1)
        if not administratie:
            return pd.DataFrame({
                "maand": [],
                "kostensoort": [],
                "bedrag": [],
            })

        try:
            df = self.execute_query(
                QUERY_TERUGKERENDE_KOSTEN,
                {"startdatum": startdatum, "einddatum": einddatum, "administratie": administratie}
            )
            return df
        except Exception as e:
            print(f"Error fetching recurring costs: {e}")
            return pd.DataFrame({
                "maand": [],
                "kostensoort": [],
                "bedrag": [],
            })

    def get_historische_cashflow_per_week(
        self,
        startdatum: date = None,
        einddatum: date = None,
        administratie: str = None,
        administratie_key: int = None
    ) -> pd.DataFrame:
        """
        Get historical weekly cashflow for seasonality analysis.

        Args:
            startdatum: Start date for analysis (default: 12 months ago)
            einddatum: End date for analysis (default: today)
            administratie: Administration name (filter by name)
            administratie_key: Administration key (filter by key, takes precedence over name)

        Returns:
            DataFrame with weekly cashflow (inkomsten, uitgaven, netto)
        """
        if einddatum is None:
            einddatum = date.today()
        if startdatum is None:
            startdatum = date(einddatum.year - 1, einddatum.month, 1)

        # Return empty if no filter specified
        if not administratie and not administratie_key:
            return pd.DataFrame({
                "week_start": [],
                "week_nummer": [],
                "maand": [],
                "inkomsten": [],
                "uitgaven": [],
                "netto": [],
            })

        try:
            # Use administratie_key if provided, otherwise use administratie name
            if administratie_key:
                df = self.execute_query(
                    QUERY_HISTORISCHE_CASHFLOW_PER_WEEK_BY_KEY,
                    {"startdatum": startdatum, "einddatum": einddatum, "administratie_key": administratie_key}
                )
            else:
                df = self.execute_query(
                    QUERY_HISTORISCHE_CASHFLOW_PER_WEEK,
                    {"startdatum": startdatum, "einddatum": einddatum, "administratie": administratie}
                )
            return df
        except Exception as e:
            print(f"Error fetching historical weekly cashflow: {e}")
            return pd.DataFrame({
                "week_start": [],
                "week_nummer": [],
                "maand": [],
                "inkomsten": [],
                "uitgaven": [],
                "netto": [],
            })


class MockDatabase:
    """
    Mock database for development/demo purposes.
    Generates realistic sample data without requiring actual database.
    """

    def __init__(self):
        import numpy as np
        from datetime import datetime, timedelta

        self.np = np
        self.today = datetime.now().date()

    def get_banksaldo(self, standdatum: date = None) -> pd.DataFrame:
        """Generate mock bank balance data."""
        return pd.DataFrame({
            "bank_naam": ["ING Zakelijk", "Rabobank", "ABN AMRO Deposito"],
            "rekeningnummer": ["NL91INGB0001234567", "NL02RABO9876543210", "NL45ABNA1122334455"],
            "saldo": [125000.0, 45000.0, 100000.0],
            "datum": [self.today] * 3,
        })

    def get_openstaande_debiteuren(self, standdatum: date = None) -> pd.DataFrame:
        """Generate mock accounts receivable data."""
        self.np.random.seed(42)
        n_records = 25

        # Generate realistic due dates (some overdue, some upcoming)
        base_dates = pd.date_range(
            start=self.today - pd.Timedelta(days=30),
            end=self.today + pd.Timedelta(days=90),
            periods=n_records
        )

        companies = [
            "Bouwbedrijf De Vries BV", "Installatiebedrijf Jansen", "Techniek Plus BV",
            "Van der Berg Constructie", "Klimaat Totaal BV", "Electro Services NL",
            "Warmte Comfort BV", "Sanitair Express", "Dak & Gevel BV", "Schildersbedrijf Pietersen"
        ]

        return pd.DataFrame({
            "debiteur_code": [f"DEB{i:04d}" for i in range(n_records)],
            "debiteur_naam": self.np.random.choice(companies, n_records),
            "factuurnummer": [f"VF2024{i:05d}" for i in range(n_records)],
            "factuurdatum": base_dates - pd.Timedelta(days=30),
            "vervaldatum": base_dates,
            "bedrag_excl_btw": self.np.random.uniform(1500, 45000, n_records).round(2),
            "betaald": self.np.zeros(n_records),
            "openstaand": self.np.random.uniform(1500, 45000, n_records).round(2),
            "betaaltermijn_dagen": self.np.random.choice([14, 30, 45, 60], n_records),
            "administratie": ["Demo Administratie"] * n_records,
            "bedrijfseenheid": ["Demo Bedrijfseenheid"] * n_records,
        })

    def get_openstaande_crediteuren(self, standdatum: date = None) -> pd.DataFrame:
        """Generate mock accounts payable data."""
        self.np.random.seed(43)
        n_records = 20

        base_dates = pd.date_range(
            start=self.today - pd.Timedelta(days=14),
            end=self.today + pd.Timedelta(days=60),
            periods=n_records
        )

        suppliers = [
            "Groothandel Technische Artikelen", "Sanitair Groothandel BV", "Elektro Supplies NL",
            "Bouwmaterialen Direct", "CV Onderdelen Centrum", "Gereedschap Totaal",
            "Isolatiematerialen BV", "Verwarming Groothandel", "Loodgieter Supplies"
        ]

        return pd.DataFrame({
            "crediteur_code": [f"CRED{i:04d}" for i in range(n_records)],
            "crediteur_naam": self.np.random.choice(suppliers, n_records),
            "factuurnummer": [f"INK2024{i:05d}" for i in range(n_records)],
            "factuurdatum": base_dates - pd.Timedelta(days=14),
            "vervaldatum": base_dates,
            "bedrag_excl_btw": self.np.random.uniform(500, 25000, n_records).round(2),
            "betaald": self.np.zeros(n_records),
            "openstaand": self.np.random.uniform(500, 25000, n_records).round(2),
            "administratie": ["Demo Administratie"] * n_records,
            "bedrijfseenheid": ["Demo Bedrijfseenheid"] * n_records,
        })

    def get_geplande_salarissen(self) -> pd.DataFrame:
        """Generate mock salary payment data."""
        from datetime import date
        # Monthly salary run, typically around 25th of month
        dates = []
        amounts = []
        current = self.today.replace(day=25)

        for i in range(4):  # Next 4 months
            try:
                if current.month + i <= 12:
                    pay_date = current.replace(month=current.month + i)
                else:
                    pay_date = current.replace(year=current.year + 1, month=(current.month + i - 12))
                dates.append(pay_date)
                amounts.append(85000 + self.np.random.uniform(-5000, 5000))
            except ValueError:
                # Handle edge cases like day 31 in months with fewer days
                continue

        return pd.DataFrame({
            "betaaldatum": dates,
            "omschrijving": ["Salarisrun " + d.strftime("%B %Y") for d in dates],
            "bedrag": amounts,
        })

    def get_historisch_betalingsgedrag(self) -> pd.DataFrame:
        """Generate mock historical payment behavior data."""
        self.np.random.seed(44)
        n_months = 12

        dates = pd.date_range(
            end=self.today,
            periods=n_months,
            freq='ME'  # Month End - fixed deprecation warning
        )

        return pd.DataFrame({
            "maand": dates,
            "gem_betaaltermijn_debiteuren": self.np.random.uniform(28, 45, n_months).round(1),
            "gem_betaaltermijn_crediteuren": self.np.random.uniform(25, 35, n_months).round(1),
            "inkomsten": self.np.random.uniform(150000, 250000, n_months).round(2),
            "uitgaven": self.np.random.uniform(120000, 200000, n_months).round(2),
        })

    def get_terugkerende_kosten(self, startdatum: date = None, einddatum: date = None, administratie: str = None) -> pd.DataFrame:
        """Generate mock recurring costs data."""
        self.np.random.seed(45)
        n_months = 12

        dates = pd.date_range(
            end=self.today,
            periods=n_months,
            freq='ME'
        )

        rows = []
        kostensoorten = ["Personeelskosten", "Huisvestingskosten", "Autokosten", "Machinekosten"]
        base_amounts = [85000, 12000, 8000, 5000]

        for d in dates:
            for soort, base in zip(kostensoorten, base_amounts):
                rows.append({
                    "maand": d,
                    "kostensoort": soort,
                    "bedrag": base + self.np.random.uniform(-base * 0.1, base * 0.1)
                })

        return pd.DataFrame(rows)

    def get_historische_cashflow_per_week(self, startdatum: date = None, einddatum: date = None, administratie: str = None) -> pd.DataFrame:
        """Generate mock weekly historical cashflow data for ML model training."""
        self.np.random.seed(46)

        # Genereer 52 weken aan data (1 jaar)
        n_weeks = 52

        week_starts = pd.date_range(
            end=self.today,
            periods=n_weeks,
            freq='W-MON'  # Week start op maandag
        )

        # Basis inkomsten/uitgaven met seizoenspatroon
        base_inkomsten = 180000
        base_uitgaven = 150000

        rows = []
        for i, week_start in enumerate(week_starts):
            maand = week_start.month

            # Seizoenspatroon: zomer (juni-aug) lager, winter hoger
            if maand in [6, 7, 8]:
                seizoen_factor = 0.85  # Zomer: 15% lager
            elif maand in [11, 12, 1]:
                seizoen_factor = 1.15  # Winter: 15% hoger
            else:
                seizoen_factor = 1.0

            # Voeg wat random variatie toe
            inkomsten = base_inkomsten * seizoen_factor * self.np.random.uniform(0.8, 1.2)
            uitgaven = base_uitgaven * seizoen_factor * self.np.random.uniform(0.85, 1.15)

            rows.append({
                "week_start": week_start,
                "week_nummer": week_start.isocalendar()[1],
                "maand": maand,
                "inkomsten": round(inkomsten, 2),
                "uitgaven": round(uitgaven, 2),
                "netto": round(inkomsten - uitgaven, 2),
            })

        return pd.DataFrame(rows)


def get_database(use_mock: bool = True, customer_code: Optional[str] = None) -> Union[SyntessDWHConnection, MockDatabase, "FailedConnectionDatabase"]:
    """
    Factory function to get database connection.

    Args:
        use_mock: If True, returns mock database for demo purposes.
        customer_code: Optional 4-digit customer code (e.g., "1241")

    Returns:
        Database connection or mock database instance.
    """
    if use_mock:
        return MockDatabase()

    if not customer_code:
        print("[WARNING] No customer code provided, returning mock database")
        return MockDatabase()

    # Build config met customer_code
    config = DatabaseConfig.from_env(customer_code=customer_code)

    # Debug output - useful for troubleshooting
    print(f"[INFO] Attempting connection to database: {config.database} on {config.host}:{config.port}")

    db = SyntessDWHConnection(config)

    if db.test_connection():
        print(f"[SUCCESS] Connected to database: {config.database}")
        return db
    else:
        print(f"[ERROR] Database connection to {config.database} failed!")
        # Return a "failed" mock that shows the error, not silent fallback
        return FailedConnectionDatabase(customer_code, config.host)


class FailedConnectionDatabase:
    """
    Placeholder database that returns empty data with error indication.
    Used when real database connection fails, to avoid silently showing mock data.
    """

    def __init__(self, customer_code: str, host: str):
        self.customer_code = customer_code
        self.host = host
        self.error_msg = f"Kon niet verbinden met database {customer_code} op {host}"

    def get_banksaldo(self, standdatum: date = None) -> pd.DataFrame:
        return pd.DataFrame({
            "bank_naam": [f"[FOUT: {self.error_msg}]"],
            "rekeningnummer": [""],
            "saldo": [0.0],
            "datum": [pd.Timestamp.now()],
        })

    def get_openstaande_debiteuren(self, standdatum: date = None) -> pd.DataFrame:
        return pd.DataFrame({
            "debiteur_code": [],
            "debiteur_naam": [],
            "factuurnummer": [],
            "factuurdatum": [],
            "vervaldatum": [],
            "bedrag_excl_btw": [],
            "betaald": [],
            "openstaand": [],
            "betaaltermijn_dagen": [],
            "administratie": [],
            "bedrijfseenheid": [],
        })

    def get_openstaande_crediteuren(self, standdatum: date = None) -> pd.DataFrame:
        return pd.DataFrame({
            "crediteur_code": [],
            "crediteur_naam": [],
            "factuurnummer": [],
            "factuurdatum": [],
            "vervaldatum": [],
            "bedrag_excl_btw": [],
            "betaald": [],
            "openstaand": [],
            "administratie": [],
            "bedrijfseenheid": [],
        })

    def get_geplande_salarissen(self) -> pd.DataFrame:
        return pd.DataFrame({
            "betaaldatum": [],
            "omschrijving": [],
            "bedrag": [],
        })

    def get_historisch_betalingsgedrag(self) -> pd.DataFrame:
        return pd.DataFrame({
            "maand": [],
            "gem_betaaltermijn_debiteuren": [],
            "gem_betaaltermijn_crediteuren": [],
            "inkomsten": [],
            "uitgaven": [],
        })


# Backwards compatibility alias
DatabaseConnection = SyntessDWHConnection
