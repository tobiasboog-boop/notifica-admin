"""
Test cashflow volatiliteit - check vervaldatum verdeling
"""

import psycopg2
import pandas as pd
from datetime import date, timedelta

conn = psycopg2.connect(
    host="10.3.152.9",
    port=5432,
    database="1273",
    user="postgres",
    password="TQwSTtLM9bSaLD"
)

standdatum = date(2025, 12, 31)
admin_filter = "Kronenburg Techniek B.V"

print("=" * 100)
print("TEST CASHFLOW VOLATILITEIT")
print(f"Standdatum: {standdatum}")
print("=" * 100)

# Haal openstaande debiteuren op met vervaldatum
print("\n\n=== DEBITEUREN MET VERVALDATUM ===")
query_deb = """
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
ORDER BY vft."Vervaldatum" ASC
"""
df_deb = pd.read_sql_query(query_deb, conn, params={"standdatum": standdatum, "administratie": admin_filter})
print(f"Aantal records met unieke debiteur/vervaldatum: {len(df_deb)}")
print(f"\nEerste 20 records:")
print(df_deb.head(20).to_string())

# Verdeling per week vanaf standdatum
print("\n\n=== VERDELING PER WEEK (PROGNOSE) ===")
df_deb['vervaldatum'] = pd.to_datetime(df_deb['vervaldatum']).dt.date

# Bereken week nummer relatief aan standdatum
def get_week_num(verval):
    if verval is None:
        return 99  # Onbekend
    days = (verval - standdatum).days
    if days < 0:
        return -1  # Verlopen
    return days // 7 + 1

df_deb['week_nummer'] = df_deb['vervaldatum'].apply(get_week_num)

# Groepeer per week
week_summary = df_deb.groupby('week_nummer').agg({
    'bedrag': ['sum', 'count']
}).reset_index()
week_summary.columns = ['week', 'totaal_bedrag', 'aantal_facturen']

print(week_summary.to_string())

# Check NULL vervaldatums
null_count = df_deb['vervaldatum'].isna().sum()
print(f"\nAantal records met NULL vervaldatum: {null_count}")

# CREDITEUREN
print("\n\n=== CREDITEUREN MET VERVALDATUM ===")
query_cred = """
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
ORDER BY ift."Vervaldatum" ASC
"""
df_cred = pd.read_sql_query(query_cred, conn, params={"standdatum": standdatum, "administratie": admin_filter})
print(f"Aantal openstaande crediteuren: {len(df_cred)}")

if not df_cred.empty:
    df_cred['vervaldatum'] = pd.to_datetime(df_cred['vervaldatum']).dt.date
    df_cred['week_nummer'] = df_cred['vervaldatum'].apply(get_week_num)

    week_cred = df_cred.groupby('week_nummer').agg({
        'bedrag': ['sum', 'count']
    }).reset_index()
    week_cred.columns = ['week', 'totaal_bedrag', 'aantal_facturen']
    print("\nVerdeling per week:")
    print(week_cred.to_string())

# SIMULEER CASHFLOW PROGNOSE
print("\n\n=== GESIMULEERDE CASHFLOW PROGNOSE (13 weken) ===")
start_balance = 1444904  # Bank saldo Power BI

print(f"Start saldo: {start_balance:,.2f}")
print(f"\n{'Week':<6} {'Inkomsten':>15} {'Uitgaven':>15} {'Netto':>15} {'Saldo':>15}")
print("-" * 72)

cumulative = start_balance
for week in range(1, 14):
    # Inkomsten: debiteuren met vervaldatum in deze week
    inkomsten = df_deb[df_deb['week_nummer'] == week]['bedrag'].sum()

    # Uitgaven: crediteuren met vervaldatum in deze week
    uitgaven = df_cred[df_cred['week_nummer'] == week]['bedrag'].sum() if not df_cred.empty else 0

    netto = inkomsten - uitgaven
    cumulative += netto

    print(f"Week {week:<2} {inkomsten:>15,.2f} {uitgaven:>15,.2f} {netto:>15,.2f} {cumulative:>15,.2f}")

# ANALYSE
print("\n\n" + "=" * 100)
print("ANALYSE CASHFLOW VOLATILITEIT")
print("=" * 100)

# Check of er genoeg variatie is in vervaldatums
verlopen = df_deb[df_deb['week_nummer'] == -1]['bedrag'].sum()
komende_4_weken = df_deb[(df_deb['week_nummer'] >= 1) & (df_deb['week_nummer'] <= 4)]['bedrag'].sum()
weken_5_13 = df_deb[(df_deb['week_nummer'] >= 5) & (df_deb['week_nummer'] <= 13)]['bedrag'].sum()
later = df_deb[df_deb['week_nummer'] > 13]['bedrag'].sum()

print(f"""
DEBITEUREN VERDELING:
  Verlopen (vervaldatum < standdatum):  {verlopen:>14,.2f}
  Komende 4 weken:                       {komende_4_weken:>14,.2f}
  Weken 5-13:                            {weken_5_13:>14,.2f}
  Later dan 13 weken:                    {later:>14,.2f}

Als veel bedrag in "Verlopen" zit, betekent dit dat de meeste facturen
al vervallen zijn. Dit geeft GEEN volatiliteit in de prognose omdat
we deze facturen moeten behandelen als "al verwacht" in week 1.

CONCLUSIE:
""")

if verlopen > komende_4_weken + weken_5_13:
    print("⚠️  MEESTE FACTUREN ZIJN VERLOPEN - dit veroorzaakt een 'platte' prognose")
    print("   De app toont alle verlopen facturen als inkomst in week 1")
else:
    print("✅  Facturen zijn goed verdeeld over tijd - prognose zou volatiel moeten zijn")

conn.close()
print("\n\nTest klaar!")
