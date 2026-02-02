"""
Finale validatie - vergelijk met Power BI
"""

import psycopg2
import pandas as pd
from datetime import date

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
print("FINALE VALIDATIE - VERGELIJK MET POWER BI")
print("=" * 100)

# Power BI waarden
pbi_1200 = 1702102      # Rubriek 1200 - Debiteuren
pbi_1230 = -15957       # Rubriek 1230 - Voorziening
pbi_totaal = 1686144    # Totaal Debiteuren

# BRUTO DEBITEUREN (via SSM Verkoopfactuur termijnen)
query_bruto = """
SELECT SUM(vft."Bedrag") as bruto
FROM notifica."SSM Verkoopfactuur termijnen" vft
LEFT JOIN notifica."SSM Documenten" d ON vft."VerkoopfactuurDocumentKey" = d."DocumentKey"
LEFT JOIN notifica."SSM Bedrijfseenheden" be ON d."BedrijfseenheidKey"::bigint = be."BedrijfseenheidKey"
LEFT JOIN notifica."SSM Administraties" a ON be."AdministratieKey" = a."AdministratieKey"
WHERE vft."Alloc_datum" <= %(standdatum)s
  AND a."Administratie" = %(administratie)s
"""
df_bruto = pd.read_sql_query(query_bruto, conn, params={"standdatum": standdatum, "administratie": admin_filter})
bruto = df_bruto['bruto'].iloc[0]

# VOORZIENING (via rubriek 1230 - boekjaar 2024)
query_voorziening = """
SELECT
    SUM(CASE WHEN j."Debet/Credit" = 'D' THEN j."Bedrag" ELSE 0 END) -
    SUM(CASE WHEN j."Debet/Credit" = 'C' THEN j."Bedrag" ELSE 0 END) as voorziening
FROM financieel."Journaalregels" j
JOIN financieel."Rubrieken" rub ON j."RubriekKey" = rub."RubriekKey"
JOIN notifica."SSM Administraties" a ON j."AdministratieKey" = a."AdministratieKey"
WHERE j."Boekdatum" BETWEEN '2024-01-01' AND '2024-12-31'
  AND rub."Rubriek Code" = '1230'
  AND a."Administratie" = %(administratie)s
"""
df_voorziening = pd.read_sql_query(query_voorziening, conn, params={"administratie": admin_filter})
voorziening = df_voorziening['voorziening'].iloc[0] if df_voorziening['voorziening'].iloc[0] is not None else 0

# NETTO DEBITEUREN
netto = bruto + voorziening  # voorziening is negatief

print(f"""
=== POWER BI ===
1200 - Debiteuren:      {pbi_1200:>14,}
1230 - Voorziening:     {pbi_1230:>14,}
Totaal:                 {pbi_totaal:>14,}

=== ONZE BEREKENING ===
Bruto (SSM):            {bruto:>14,.2f}
Voorziening (1230):     {voorziening:>14,.2f}
Netto:                  {netto:>14,.2f}

=== VERGELIJKING ===
                        Berekend       Power BI       Verschil   Match?
Bruto (1200):           {bruto:>12,.2f}   {pbi_1200:>12,}   {bruto - pbi_1200:>+12,.2f}   {'OK' if abs(bruto - pbi_1200) < 5000 else 'NOK'}
Voorziening (1230):     {voorziening:>12,.2f}   {pbi_1230:>12,}   {voorziening - pbi_1230:>+12,.2f}   {'OK' if abs(voorziening - pbi_1230) < 100 else 'NOK'}
Netto Totaal:           {netto:>12,.2f}   {pbi_totaal:>12,}   {netto - pbi_totaal:>+12,.2f}   {'OK' if abs(netto - pbi_totaal) < 5000 else 'NOK'}

=== ANALYSE ===
- Verschil bruto: {bruto - pbi_1200:+,.2f} ({(bruto - pbi_1200) / pbi_1200 * 100:+.2f}%)
- Verschil voorziening: {voorziening - pbi_1230:+,.2f}
- Verschil netto: {netto - pbi_totaal:+,.2f} ({(netto - pbi_totaal) / pbi_totaal * 100:+.2f}%)

CONCLUSIE: {'MATCH - Verschil binnen 1%!' if abs((netto - pbi_totaal) / pbi_totaal * 100) < 1 else 'Verschil groter dan 1%'}
""")

conn.close()
print("\nValidatie klaar!")
