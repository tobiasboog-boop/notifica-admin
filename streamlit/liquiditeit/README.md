# Liquiditeit Dashboard

Streamlit applicatie voor liquiditeitsanalyse en cashflow prognoses voor Syntess/Notifica klanten.

## Features

- **13-weeks Cashflow Prognose** - Vooruitkijkend overzicht van verwachte in- en uitgaande geldstromen
- **Liquiditeitsratio's** - Current ratio, Quick ratio en andere KPI's
- **Ouderdomsanalyse** - Aging buckets voor debiteuren en crediteuren
- **Scenario Analyse** - What-if simulaties voor betalingsgedrag
- **Alerts** - Automatische waarschuwingen bij verwachte liquiditeitstekorten

## Installatie

```bash
# Maak virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installeer dependencies
pip install -r requirements.txt

# Kopieer en configureer environment
cp .env.example .env
# Bewerk .env met je database credentials
```

## Starten

```bash
# Start de applicatie
streamlit run app.py

# Of met specifieke poort
streamlit run app.py --server.port 8501
```

De applicatie opent automatisch in je browser op http://localhost:8501

## Demo Modus

De app start standaard in demo modus met fictieve data. Dit is handig voor:
- Testen van de interface zonder database connectie
- Presentaties en demonstraties
- Ontwikkeling van nieuwe features

Schakel demo modus uit in de sidebar om echte data te gebruiken.

## Datamodel

De app verwacht de volgende data uit het Syntess DWH:

| Tabel | Beschrijving |
|-------|--------------|
| `banksaldo` | Actuele banksaldi per rekening |
| `debiteuren` | Openstaande vorderingen met vervaldata |
| `crediteuren` | Openstaande schulden met vervaldata |
| `salarissen` | Geplande salarisbetalingen |

## Project Structuur

```
liquiditeit/
├── app.py              # Hoofdapplicatie
├── config.py           # Configuratie en constanten
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── README.md           # Deze file
└── src/
    ├── __init__.py
    ├── database.py     # Database connectie module
    └── calculations.py # Business logic en berekeningen
```

## Roadmap

- [ ] Echte database queries voor Syntess DWH
- [ ] Export naar PDF/Excel
- [ ] Historische trend analyse
- [ ] ML-gebaseerde betalingsvoorspelling
- [ ] Multi-tenant support (meerdere klanten)

---

*Notifica - Business Intelligence voor installatiebedrijven*
