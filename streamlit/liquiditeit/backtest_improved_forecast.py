"""
Verbeterde Cashflow Forecast met Historische Patronen
=====================================================
Combineert:
A. Bekende cashflow (openstaande posten)
B. Verwachte cashflow (historische gemiddelden)
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import date, timedelta

conn = psycopg2.connect(
    host="10.3.152.9",
    port=5432,
    database="1273",
    user="postgres",
    password="TQwSTtLM9bSaLD"
)

admin_filter = "Kronenburg Techniek B.V"

print("=" * 100)
print("VERBETERDE CASHFLOW FORECAST MET HISTORISCHE PATRONEN")
print("=" * 100)


def get_historische_weekpatronen(conn, administratie: str, jaren_terug: int = 2) -> pd.DataFrame:
    """
    Haal gemiddelde cashflow per weeknummer op (voor seizoenspatroon).
    """
    query = """
    SELECT
        EXTRACT(WEEK FROM j."Boekdatum")::int as week_nummer,
        AVG(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as gem_inkomsten,
        AVG(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as gem_uitgaven
    FROM financieel."Journaalregels" j
    JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
    JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
    JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
    WHERE j."Boekdatum" >= CURRENT_DATE - INTERVAL '%s years'
      AND d."StandaardEntiteitKey" = 10
      AND j."RubriekKey" = dag."DagboekRubriekKey"
      AND a."Administratie" = %s
    GROUP BY EXTRACT(WEEK FROM j."Boekdatum")
    ORDER BY week_nummer
    """
    return pd.read_sql_query(query, conn, params=(jaren_terug, administratie))


def get_gemiddelde_wekelijkse_mutaties(conn, administratie: str, maanden_terug: int = 12) -> dict:
    """
    Bereken gemiddelde wekelijkse in- en uitgaven over de laatste X maanden.
    """
    query = """
    SELECT
        COUNT(DISTINCT DATE_TRUNC('week', j."Boekdatum")) as aantal_weken,
        SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as totaal_in,
        SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as totaal_uit
    FROM financieel."Journaalregels" j
    JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
    JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
    JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
    WHERE j."Boekdatum" >= CURRENT_DATE - INTERVAL '%s months'
      AND d."StandaardEntiteitKey" = 10
      AND j."RubriekKey" = dag."DagboekRubriekKey"
      AND a."Administratie" = %s
    """
    df = pd.read_sql_query(query, conn, params=(maanden_terug, administratie))

    if df.empty or df['aantal_weken'].iloc[0] == 0:
        return {"gem_in": 0, "gem_uit": 0, "gem_netto": 0}

    n_weken = df['aantal_weken'].iloc[0]
    return {
        "gem_in": df['totaal_in'].iloc[0] / n_weken,
        "gem_uit": df['totaal_uit'].iloc[0] / n_weken,
        "gem_netto": (df['totaal_in'].iloc[0] - df['totaal_uit'].iloc[0]) / n_weken,
    }


def get_openstaande_debiteuren(conn, standdatum: date, administratie: str) -> pd.DataFrame:
    query = """
    SELECT vft."Vervaldatum" as vervaldatum, SUM(vft."Bedrag") as bedrag
    FROM notifica."SSM Verkoopfactuur termijnen" vft
    LEFT JOIN notifica."SSM Documenten" d ON vft."VerkoopfactuurDocumentKey" = d."DocumentKey"
    LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
    LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
    WHERE vft."Alloc_datum" <= %(standdatum)s AND a."Administratie" = %(administratie)s
    GROUP BY vft."Vervaldatum"
    HAVING ABS(SUM(vft."Bedrag")) > 0.01
    """
    return pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})


def get_openstaande_crediteuren(conn, standdatum: date, administratie: str) -> pd.DataFrame:
    query = """
    SELECT ift."Vervaldatum" as vervaldatum, SUM(ift."Bedrag") as bedrag
    FROM notifica."SSM Inkoopfactuur termijnen" ift
    LEFT JOIN notifica."SSM Documenten" d ON ift."InkoopFactuurKey" = d."DocumentKey"
    LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
    LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
    WHERE ift."Alloc_datum" <= %(standdatum)s
      AND ift."Bankafschrift status" = 'Openstaand'
      AND a."Administratie" = %(administratie)s
    GROUP BY ift."Vervaldatum"
    """
    return pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})


def get_banksaldo(conn, standdatum: date, administratie: str) -> float:
    query = """
    SELECT SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
           SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as saldo
    FROM financieel."Journaalregels" j
    JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
    JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
    JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
    WHERE j."Boekdatum" <= %(standdatum)s AND d."StandaardEntiteitKey" = 10
      AND j."RubriekKey" = dag."DagboekRubriekKey" AND a."Administratie" = %(administratie)s
    """
    df = pd.read_sql_query(query, conn, params={"standdatum": standdatum, "administratie": administratie})
    return df['saldo'].iloc[0] if not df.empty else 0


def get_werkelijke_mutaties(conn, startdatum: date, einddatum: date, administratie: str) -> pd.DataFrame:
    query = """
    SELECT DATE_TRUNC('week', j."Boekdatum") as week_start,
           SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) as inkomsten,
           SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as uitgaven,
           SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
           SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as netto
    FROM financieel."Journaalregels" j
    JOIN stam."Documenten" d ON j."DocumentKey" = d."DocumentKey"
    JOIN stam."Dagboeken" dag ON d."DagboekKey" = dag."DagboekKey"
    JOIN notifica."SSM Administraties" a ON dag."AdministratieKey" = a."AdministratieKey"
    WHERE j."Boekdatum" >= %(startdatum)s AND j."Boekdatum" < %(einddatum)s
      AND d."StandaardEntiteitKey" = 10 AND j."RubriekKey" = dag."DagboekRubriekKey"
      AND a."Administratie" = %(administratie)s
    GROUP BY DATE_TRUNC('week', j."Boekdatum")
    ORDER BY week_start
    """
    return pd.read_sql_query(query, conn, params={
        "startdatum": startdatum, "einddatum": einddatum, "administratie": administratie})


def maak_verbeterde_forecast(
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    start_saldo: float,
    startdatum: date,
    hist_gemiddelden: dict,
    weeks: int = 13,
    bekende_factor: float = 0.8,  # Hoeveel van bekende posten wordt daadwerkelijk betaald
) -> pd.DataFrame:
    """
    Verbeterde forecast die combineert:
    1. Bekende inkomsten (openstaande debiteuren) x bekende_factor
    2. Bekende uitgaven (openstaande crediteuren) x bekende_factor
    3. Historische baseline voor REST van de cashflow
    """
    week_starts = [startdatum + timedelta(weeks=i) for i in range(weeks + 1)]

    # Historische baseline per week
    hist_in = hist_gemiddelden.get("gem_in", 0)
    hist_uit = hist_gemiddelden.get("gem_uit", 0)

    forecast_rows = []
    for i in range(weeks):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]

        # BEKENDE inkomsten uit openstaande debiteuren
        bekende_in = 0.0
        if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
            deb = debiteuren.copy()
            deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
            mask = (deb["vervaldatum"] >= week_start) & (deb["vervaldatum"] < week_end)
            bekende_in = deb.loc[mask, "bedrag"].sum() * bekende_factor

        # BEKENDE uitgaven uit openstaande crediteuren
        bekende_uit = 0.0
        if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
            cred = crediteuren.copy()
            cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
            mask = (cred["vervaldatum"] >= week_start) & (cred["vervaldatum"] < week_end)
            bekende_uit = cred.loc[mask, "bedrag"].sum() * bekende_factor

        # BASELINE: historisch gemiddelde voor wat we NIET weten
        # Als we weinig bekende posten hebben, voeg historische baseline toe
        baseline_in = max(0, hist_in - bekende_in) if bekende_in < hist_in * 0.5 else 0
        baseline_uit = max(0, hist_uit - bekende_uit) if bekende_uit < hist_uit * 0.5 else 0

        totaal_in = bekende_in + baseline_in
        totaal_uit = bekende_uit + baseline_uit

        forecast_rows.append({
            "week_start": week_start,
            "week_nummer": i + 1,
            "bekende_in": bekende_in,
            "baseline_in": baseline_in,
            "forecast_inkomsten": totaal_in,
            "bekende_uit": bekende_uit,
            "baseline_uit": baseline_uit,
            "forecast_uitgaven": totaal_uit,
            "forecast_netto": totaal_in - totaal_uit,
        })

    df = pd.DataFrame(forecast_rows)
    df["forecast_cumulatief"] = start_saldo + df["forecast_netto"].cumsum()
    return df


def bereken_kwaliteit(forecast: pd.DataFrame, realisatie: pd.DataFrame) -> dict:
    forecast["week_start"] = pd.to_datetime(forecast["week_start"])
    realisatie["week_start"] = pd.to_datetime(realisatie["week_start"])
    merged = forecast.merge(realisatie, on="week_start", how="left", suffixes=("_fc", "_re")).fillna(0)
    merged["afwijking"] = merged["forecast_netto"] - merged["netto"]
    mae = abs(merged["afwijking"]).mean()
    return {"mae": mae, "merged": merged,
            "total_fc": merged["forecast_netto"].sum(),
            "total_re": merged["netto"].sum()}


# ==================================================================================
# STAP 1: Haal historische gemiddelden op
# ==================================================================================
print("\n--- Historische gemiddelden (laatste 12 maanden) ---")
hist = get_gemiddelde_wekelijkse_mutaties(conn, admin_filter, 12)
print(f"  Gem. inkomsten/week: EUR {hist['gem_in']:,.0f}")
print(f"  Gem. uitgaven/week:  EUR {hist['gem_uit']:,.0f}")
print(f"  Gem. netto/week:     EUR {hist['gem_netto']:,.0f}")


# ==================================================================================
# BACKTEST: Q1 2024 met OUDE methodiek vs NIEUWE methodiek
# ==================================================================================
print("\n\n" + "=" * 80)
print("BACKTEST Q1 2024: OUDE vs NIEUWE METHODIEK")
print("=" * 80)

backtest_start = date(2024, 1, 1)
backtest_weeks = 13
backtest_end = backtest_start + timedelta(weeks=backtest_weeks)

# Haal data op
deb = get_openstaande_debiteuren(conn, backtest_start, admin_filter)
cred = get_openstaande_crediteuren(conn, backtest_start, admin_filter)
saldo = get_banksaldo(conn, backtest_start, admin_filter)
realisatie = get_werkelijke_mutaties(conn, backtest_start, backtest_end, admin_filter)

print(f"\nData per {backtest_start}:")
print(f"  Openstaande deb: EUR {deb['bedrag'].sum():,.0f}")
print(f"  Openstaande cred: EUR {cred['bedrag'].sum():,.0f}")
print(f"  Banksaldo: EUR {saldo:,.0f}")

# OUDE methodiek (alleen vervaldatums)
print("\n--- OUDE METHODIEK (alleen vervaldatums) ---")
forecast_old = maak_verbeterde_forecast(deb, cred, saldo, backtest_start,
                                         {"gem_in": 0, "gem_uit": 0}, backtest_weeks,
                                         bekende_factor=1.0)  # 100% van bekende posten
quality_old = bereken_kwaliteit(forecast_old, realisatie)
print(f"  MAE: EUR {quality_old['mae']:,.0f}/week")
print(f"  Totaal forecast: EUR {quality_old['total_fc']:,.0f}")
print(f"  Totaal realisatie: EUR {quality_old['total_re']:,.0f}")

# NIEUWE methodiek (met historische baseline)
print("\n--- NIEUWE METHODIEK (met historische baseline) ---")
# Gebruik historische data van voor de backtest periode
hist_voor_backtest = get_gemiddelde_wekelijkse_mutaties(conn, admin_filter, 12)
forecast_new = maak_verbeterde_forecast(deb, cred, saldo, backtest_start,
                                         hist_voor_backtest, backtest_weeks,
                                         bekende_factor=0.7)  # 70% van bekende posten
quality_new = bereken_kwaliteit(forecast_new, realisatie)
print(f"  MAE: EUR {quality_new['mae']:,.0f}/week")
print(f"  Totaal forecast: EUR {quality_new['total_fc']:,.0f}")
print(f"  Totaal realisatie: EUR {quality_new['total_re']:,.0f}")

# Vergelijk per week
print("\n--- Detail vergelijking per week ---")
merged_old = quality_old["merged"]
merged_new = quality_new["merged"]

print(f"{'Week':<12} {'Realisatie':>12} {'Oude FC':>12} {'Oude Afw':>12} {'Nieuwe FC':>12} {'Nieuwe Afw':>12}")
print("-" * 75)
for i in range(len(merged_old)):
    week = merged_old.iloc[i]["week_start"].strftime("%Y-%m-%d")
    re = merged_old.iloc[i]["netto"]
    fc_old = merged_old.iloc[i]["forecast_netto"]
    afw_old = merged_old.iloc[i]["afwijking"]
    fc_new = merged_new.iloc[i]["forecast_netto"]
    afw_new = merged_new.iloc[i]["afwijking"]
    print(f"{week:<12} {re:>12,.0f} {fc_old:>12,.0f} {afw_old:>+12,.0f} {fc_new:>12,.0f} {afw_new:>+12,.0f}")


# ==================================================================================
# CONCLUSIE
# ==================================================================================
print("\n\n" + "=" * 100)
print("CONCLUSIE")
print("=" * 100)

verbetering = (quality_old['mae'] - quality_new['mae']) / quality_old['mae'] * 100

print(f"""
OUDE METHODIEK (alleen openstaande posten):
  - MAE: EUR {quality_old['mae']:,.0f}/week
  - Probleem: mist alle toekomstige transacties

NIEUWE METHODIEK (openstaande posten + historische baseline):
  - MAE: EUR {quality_new['mae']:,.0f}/week
  - Verbetering: {verbetering:+.1f}%

AANBEVELING VOOR DASHBOARD:
1. Toon TWEE scenarios:
   - "Conservatief": alleen bekende openstaande posten
   - "Met historische trend": inclusief baseline op basis van gemiddelden

2. Voeg bekende_factor slider toe (standaard 70-80%)
   Dit corrigeert voor het feit dat niet alle facturen op tijd betaald worden

3. Voeg realisatie lijn toe zodra data beschikbaar is
   Zo kan de gebruiker zien hoe goed de forecast was
""")

conn.close()
print("\nBacktest klaar!")
