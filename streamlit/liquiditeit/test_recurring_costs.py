"""
Test terugkerende kosten en seasonality queries
"""

import psycopg2
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

conn = psycopg2.connect(
    host="10.3.152.9",
    port=5432,
    database="1273",
    user="postgres",
    password="TQwSTtLM9bSaLD"
)

standdatum = date(2025, 12, 31)
admin_filter = "Kronenburg Techniek B.V"
startdatum = date(2024, 1, 1)
einddatum = date(2024, 12, 31)

print("=" * 100)
print("TEST TERUGKERENDE KOSTEN & SEASONALITY")
print(f"Periode: {startdatum} - {einddatum}")
print(f"Administratie: {admin_filter}")
print("=" * 100)

# === TEST 1: Terugkerende kosten per maand ===
print("\n\n=== TERUGKERENDE KOSTEN PER MAAND ===")
query_kosten = """
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
  AND (rub."Rubriek Code" LIKE '4%%'
       OR rub."Rubriek Code" LIKE '61%%'
       OR rub."Rubriek Code" LIKE '62%%'
       OR rub."Rubriek Code" LIKE '65%%')
  AND a."Administratie" = %(administratie)s
GROUP BY DATE_TRUNC('month', j."Boekdatum"), rub."Rubriek Code", rub."Rubriek"
ORDER BY maand, rubriek_code
"""
df_kosten = pd.read_sql_query(query_kosten, conn, params={
    "startdatum": startdatum,
    "einddatum": einddatum,
    "administratie": admin_filter
})
print(f"Aantal records: {len(df_kosten)}")

# Samenvattng per kostensoort
print("\n--- Samenvatting per kostensoort (heel 2024) ---")
summary = df_kosten.groupby('kostensoort')['bedrag'].sum().sort_values(ascending=False)
for soort, bedrag in summary.items():
    gem_maand = bedrag / 12
    print(f"  {soort:25s}: totaal EUR {bedrag:>12,.2f} | gem/maand EUR {gem_maand:>10,.2f}")

# Samenvatting per maand
print("\n--- Samenvatting per maand ---")
monthly = df_kosten.groupby('maand')['bedrag'].sum().reset_index()
monthly['maand'] = pd.to_datetime(monthly['maand']).dt.strftime('%Y-%m')
for _, row in monthly.iterrows():
    print(f"  {row['maand']}: EUR {row['bedrag']:>12,.2f}")

# === TEST 2: Historische cashflow per week ===
print("\n\n=== HISTORISCHE CASHFLOW PER WEEK ===")
query_weekly = """
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
  AND d."StandaardEntiteitKey" = 10
  AND j."RubriekKey" = dag."DagboekRubriekKey"
  AND a."Administratie" = %(administratie)s
GROUP BY DATE_TRUNC('week', j."Boekdatum"),
         EXTRACT(WEEK FROM j."Boekdatum"),
         EXTRACT(MONTH FROM j."Boekdatum")
ORDER BY week_start
"""
df_weekly = pd.read_sql_query(query_weekly, conn, params={
    "startdatum": startdatum,
    "einddatum": einddatum,
    "administratie": admin_filter
})
print(f"Aantal weken: {len(df_weekly)}")

if not df_weekly.empty:
    print("\n--- Eerste 10 weken ---")
    print(df_weekly.head(10).to_string())

    print("\n--- Statistieken per week ---")
    print(f"  Gem. inkomsten/week:  EUR {df_weekly['inkomsten'].mean():>12,.2f}")
    print(f"  Gem. uitgaven/week:   EUR {df_weekly['uitgaven'].mean():>12,.2f}")
    print(f"  Gem. netto/week:      EUR {df_weekly['netto'].mean():>12,.2f}")
    print(f"  Std. netto/week:      EUR {df_weekly['netto'].std():>12,.2f}")

    # Seizoenspatroon per maand
    print("\n--- Gemiddelde cashflow per maand (seizoenspatroon) ---")
    monthly_pattern = df_weekly.groupby('maand').agg({
        'inkomsten': 'mean',
        'uitgaven': 'mean',
        'netto': 'mean'
    }).round(2)
    monthly_pattern.index = monthly_pattern.index.astype(int)
    maand_namen = ['', 'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
    for maand, row in monthly_pattern.iterrows():
        print(f"  {maand_namen[maand]:3s}: in EUR {row['inkomsten']:>10,.0f} | uit EUR {row['uitgaven']:>10,.0f} | netto EUR {row['netto']:>+10,.0f}")

# === CONCLUSIE ===
print("\n\n" + "=" * 100)
print("CONCLUSIE VOOR CASHFLOW PROGNOSE")
print("=" * 100)

if not df_kosten.empty:
    gem_personeelskosten = summary.get('Personeelskosten', 0) / 12
    gem_huisvesting = summary.get('Huisvestingskosten', 0) / 12
    gem_auto = summary.get('Autokosten', 0) / 12

    print(f"""
VASTE LASTEN (gemiddeld per maand):
  Personeelskosten:   EUR {gem_personeelskosten:>10,.2f}
  Huisvestingskosten: EUR {gem_huisvesting:>10,.2f}
  Autokosten:         EUR {gem_auto:>10,.2f}

  Per week (approx):  EUR {(gem_personeelskosten + gem_huisvesting + gem_auto) / 4.33:>10,.2f}

AANBEVELING VOOR PROGNOSE:
1. Voeg gemiddelde vaste lasten toe aan elke week
2. Gebruik seizoenspatroon om inkomsten/uitgaven te schalen
3. Combineer met openstaande posten voor nauwkeurige prognose
""")

conn.close()
print("\nTest klaar!")
