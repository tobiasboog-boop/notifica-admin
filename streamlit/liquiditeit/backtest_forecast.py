"""
Backtest Cashflow Forecast
==========================
Vergelijk forecast met realisatie om de voorspellingskwaliteit te meten.

Methodiek:
1. Kies een historische startdatum (bijv. 1 januari 2024)
2. Maak een prognose voor 13 weken vooruit
3. Haal de werkelijke bankmutaties op voor diezelfde 13 weken
4. Vergelijk en meet de afwijking
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple

conn = psycopg2.connect(
    host="10.3.152.9",
    port=5432,
    database="1273",
    user="postgres",
    password="TQwSTtLM9bSaLD"
)

admin_filter = "Kronenburg Techniek B.V"

print("=" * 100)
print("BACKTEST CASHFLOW FORECAST")
print("=" * 100)


def get_openstaande_debiteuren_op_datum(conn, standdatum: date, administratie: str) -> pd.DataFrame:
    """Haal openstaande debiteuren op zoals ze waren op een specifieke datum."""
    query = """
    SELECT
        vft."Debiteur" as debiteur,
        vft."Vervaldatum" as vervaldatum,
        SUM(vft."Bedrag") as bedrag
    FROM notifica."SSM Verkoopfactuur termijnen" vft
    LEFT JOIN notifica."SSM Documenten" d ON vft."VerkoopfactuurDocumentKey" = d."DocumentKey"
    LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
    LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
    WHERE vft."Alloc_datum" <= %(standdatum)s
      AND a."Administratie" = %(administratie)s
    GROUP BY vft."Debiteur", vft."Vervaldatum"
    HAVING ABS(SUM(vft."Bedrag")) > 0.01
    """
    return pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})


def get_openstaande_crediteuren_op_datum(conn, standdatum: date, administratie: str) -> pd.DataFrame:
    """Haal openstaande crediteuren op zoals ze waren op een specifieke datum."""
    query = """
    SELECT
        ift."Crediteur" as crediteur,
        ift."Vervaldatum" as vervaldatum,
        ift."Bedrag" as bedrag
    FROM notifica."SSM Inkoopfactuur termijnen" ift
    LEFT JOIN notifica."SSM Documenten" d ON ift."InkoopFactuurKey" = d."DocumentKey"
    LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
    LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
    WHERE ift."Alloc_datum" <= %(standdatum)s
      AND ift."Bankafschrift status" = 'Openstaand'
      AND a."Administratie" = %(administratie)s
    """
    return pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})


def get_werkelijke_bankmutaties(conn, startdatum: date, einddatum: date, administratie: str) -> pd.DataFrame:
    """Haal werkelijke bankmutaties op voor een periode."""
    query = """
    SELECT
        DATE_TRUNC('week', j."Boekdatum") as week_start,
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
      AND d."StandaardEntiteitKey" = 10
      AND j."RubriekKey" = dag."DagboekRubriekKey"
      AND a."Administratie" = %(administratie)s
    GROUP BY DATE_TRUNC('week', j."Boekdatum")
    ORDER BY week_start
    """
    return pd.read_sql_query(query, conn, params={
        "startdatum": startdatum,
        "einddatum": einddatum,
        "administratie": administratie
    })


def get_banksaldo_op_datum(conn, standdatum: date, administratie: str) -> float:
    """Haal banksaldo op een specifieke datum."""
    query = """
    SELECT
        SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
        SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as saldo
    FROM financieel."Journaalregels" j
    JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
    JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
    JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
    WHERE j."Boekdatum" <= %(standdatum)s
      AND d."StandaardEntiteitKey" = 10
      AND j."RubriekKey" = dag."DagboekRubriekKey"
      AND a."Administratie" = %(administratie)s
    """
    df = pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})
    return df['saldo'].iloc[0] if not df.empty else 0


def maak_simpele_forecast(debiteuren: pd.DataFrame, crediteuren: pd.DataFrame,
                          start_saldo: float, startdatum: date, weeks: int = 13) -> pd.DataFrame:
    """
    Maak een simpele forecast gebaseerd op vervaldatums.
    Dit is de huidige methodiek.
    """
    week_starts = [startdatum + timedelta(weeks=i) for i in range(weeks + 1)]

    forecast_rows = []
    for i in range(weeks):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]

        # Inkomsten: debiteuren met vervaldatum in deze week
        deb_income = 0.0
        if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
            deb = debiteuren.copy()
            deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
            mask = (deb["vervaldatum"] >= week_start) & (deb["vervaldatum"] < week_end)
            deb_income = deb.loc[mask, "bedrag"].sum()

        # Uitgaven: crediteuren met vervaldatum in deze week
        cred_expense = 0.0
        if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
            cred = crediteuren.copy()
            cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
            mask = (cred["vervaldatum"] >= week_start) & (cred["vervaldatum"] < week_end)
            cred_expense = cred.loc[mask, "bedrag"].sum()

        forecast_rows.append({
            "week_start": week_start,
            "week_nummer": i + 1,
            "forecast_inkomsten": deb_income,
            "forecast_uitgaven": cred_expense,
            "forecast_netto": deb_income - cred_expense,
        })

    df = pd.DataFrame(forecast_rows)
    df["forecast_cumulatief"] = start_saldo + df["forecast_netto"].cumsum()
    return df


def bereken_forecast_kwaliteit(forecast: pd.DataFrame, realisatie: pd.DataFrame) -> dict:
    """Bereken forecast kwaliteit metrics."""
    # Merge op week
    forecast["week_start"] = pd.to_datetime(forecast["week_start"])
    realisatie["week_start"] = pd.to_datetime(realisatie["week_start"])

    merged = forecast.merge(realisatie, on="week_start", how="left", suffixes=("_fc", "_re"))
    merged = merged.fillna(0)

    # Bereken afwijkingen
    merged["afwijking_netto"] = merged["forecast_netto"] - merged["netto"]
    merged["afwijking_abs"] = abs(merged["afwijking_netto"])

    # Metrics
    mae = merged["afwijking_abs"].mean()  # Mean Absolute Error

    # MAPE alleen als realisatie != 0
    mape_rows = merged[merged["netto"] != 0]
    if not mape_rows.empty:
        mape = (abs(mape_rows["afwijking_netto"]) / abs(mape_rows["netto"])).mean() * 100
    else:
        mape = float('inf')

    return {
        "mae": mae,
        "mape": mape,
        "total_forecast": merged["forecast_netto"].sum(),
        "total_realisatie": merged["netto"].sum(),
        "merged_data": merged,
    }


# ==================================================================================
# BACKTEST 1: Q1 2024 (1 jan - 31 maart)
# ==================================================================================
print("\n\n" + "=" * 80)
print("BACKTEST 1: Q1 2024")
print("=" * 80)

backtest_start = date(2024, 1, 1)
backtest_weeks = 13
backtest_end = backtest_start + timedelta(weeks=backtest_weeks)

print(f"Forecast startdatum: {backtest_start}")
print(f"Forecast periode: {backtest_weeks} weken tot {backtest_end}")

# Haal data op zoals het was op 1 jan 2024
print("\nData ophalen...")
debiteuren_q1 = get_openstaande_debiteuren_op_datum(conn, backtest_start, admin_filter)
crediteuren_q1 = get_openstaande_crediteuren_op_datum(conn, backtest_start, admin_filter)
saldo_q1 = get_banksaldo_op_datum(conn, backtest_start, admin_filter)
print(f"  Openstaande debiteuren: {len(debiteuren_q1)} records, EUR {debiteuren_q1['bedrag'].sum():,.2f}")
print(f"  Openstaande crediteuren: {len(crediteuren_q1)} records, EUR {crediteuren_q1['bedrag'].sum():,.2f}")
print(f"  Banksaldo: EUR {saldo_q1:,.2f}")

# Maak forecast
print("\nForecast maken...")
forecast_q1 = maak_simpele_forecast(debiteuren_q1, crediteuren_q1, saldo_q1, backtest_start, backtest_weeks)

# Haal werkelijke realisatie op
print("Realisatie ophalen...")
realisatie_q1 = get_werkelijke_bankmutaties(conn, backtest_start, backtest_end, admin_filter)
print(f"  Weken met data: {len(realisatie_q1)}")

# Vergelijk
print("\nVergelijking forecast vs realisatie:")
quality_q1 = bereken_forecast_kwaliteit(forecast_q1, realisatie_q1)
print(f"  Mean Absolute Error (MAE): EUR {quality_q1['mae']:,.0f} per week")
print(f"  Mean Absolute % Error (MAPE): {quality_q1['mape']:.1f}%")
print(f"  Totaal forecast: EUR {quality_q1['total_forecast']:,.0f}")
print(f"  Totaal realisatie: EUR {quality_q1['total_realisatie']:,.0f}")

# Detail per week
print("\n--- Detail per week ---")
merged = quality_q1["merged_data"]
print(f"{'Week':<12} {'Forecast':>15} {'Realisatie':>15} {'Afwijking':>15}")
print("-" * 60)
for _, row in merged.iterrows():
    week = row["week_start"].strftime("%Y-%m-%d")
    print(f"{week:<12} {row['forecast_netto']:>15,.0f} {row['netto']:>15,.0f} {row['afwijking_netto']:>+15,.0f}")


# ==================================================================================
# BACKTEST 2: Q2 2024 (1 apr - 30 juni)
# ==================================================================================
print("\n\n" + "=" * 80)
print("BACKTEST 2: Q2 2024")
print("=" * 80)

backtest_start = date(2024, 4, 1)
backtest_end = backtest_start + timedelta(weeks=backtest_weeks)

print(f"Forecast startdatum: {backtest_start}")

debiteuren_q2 = get_openstaande_debiteuren_op_datum(conn, backtest_start, admin_filter)
crediteuren_q2 = get_openstaande_crediteuren_op_datum(conn, backtest_start, admin_filter)
saldo_q2 = get_banksaldo_op_datum(conn, backtest_start, admin_filter)
print(f"  Openstaande debiteuren: {len(debiteuren_q2)} records, EUR {debiteuren_q2['bedrag'].sum():,.2f}")
print(f"  Openstaande crediteuren: {len(crediteuren_q2)} records, EUR {crediteuren_q2['bedrag'].sum():,.2f}")
print(f"  Banksaldo: EUR {saldo_q2:,.2f}")

forecast_q2 = maak_simpele_forecast(debiteuren_q2, crediteuren_q2, saldo_q2, backtest_start, backtest_weeks)
realisatie_q2 = get_werkelijke_bankmutaties(conn, backtest_start, backtest_end, admin_filter)

quality_q2 = bereken_forecast_kwaliteit(forecast_q2, realisatie_q2)
print(f"\n  MAE: EUR {quality_q2['mae']:,.0f} per week")
print(f"  MAPE: {quality_q2['mape']:.1f}%")
print(f"  Totaal forecast: EUR {quality_q2['total_forecast']:,.0f}")
print(f"  Totaal realisatie: EUR {quality_q2['total_realisatie']:,.0f}")


# ==================================================================================
# CONCLUSIE EN ANALYSE
# ==================================================================================
print("\n\n" + "=" * 100)
print("CONCLUSIE BACKTEST")
print("=" * 100)

print(f"""
HUIDIGE METHODIEK: Forecast op basis van vervaldatums

RESULTATEN:
  Q1 2024: MAE EUR {quality_q1['mae']:,.0f}/week, MAPE {quality_q1['mape']:.1f}%
  Q2 2024: MAE EUR {quality_q2['mae']:,.0f}/week, MAPE {quality_q2['mape']:.1f}%

ANALYSE:
Het probleem met de huidige methodiek is dat we ALLEEN kijken naar:
- Openstaande debiteuren (wanneer verwachten we betaling)
- Openstaande crediteuren (wanneer moeten we betalen)

Wat we MISSEN:
1. Nieuwe facturen die in de forecast periode worden gemaakt
2. Nieuwe inkoopfacturen die in de forecast periode binnenkomen
3. Vaste lasten (salaris, huur) - deze zitten NIET in crediteuren!
4. Seizoenspatronen in omzet

AANBEVELING:
Voor een goede forecast hebben we TWEE componenten nodig:

A. BEKENDE CASHFLOW (hoog vertrouwen):
   - Openstaande debiteuren met vervaldatum
   - Openstaande crediteuren met vervaldatum
   - Geplande vaste lasten (salaris = rond 25e van de maand)

B. VERWACHTE CASHFLOW (gebaseerd op historie):
   - Gemiddelde nieuwe omzet per week (uit historische data)
   - Gemiddelde nieuwe inkoop per week (uit historische data)
   - Seizoenscorrectie per maand

De combinatie van A + B geeft een realistischere prognose.
""")

conn.close()
print("\nBacktest klaar!")
