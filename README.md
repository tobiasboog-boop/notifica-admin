# Notifica Customer Health Analytics

Een interactief dashboard voor het analyseren van Power BI rapportgebruik per klant.

## Features

- 🚦 **Stoplicht systeem** - Groen/Oranje/Rood per klant
- 📊 **Customer Health Score** - Gewogen score op basis van:
  - Views (40%)
  - Aantal functionarissen (30%)
  - Spreiding gebruik (20%)
  - Breedte productgroepen (10%)
- 🔍 **Drill-down** - Per klant, productgroep, functionaris
- 👤 **Eigenaar view** - Overzicht per accountmanager
- 📤 **Export** - CSV en Excel exports

## Installatie

```bash
# Maak virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Installeer dependencies
pip install -r requirements.txt

# Start de app
streamlit run app.py
```

## Gebruik

1. Start de applicatie met `streamlit run app.py`
2. Upload de Power BI Activity export (Excel)
3. Upload de Pipedrive Organizations export (Excel)
4. Bekijk de scorecard en drill-down per klant

## Data Formaat

### Power BI Export
Verwachte kolommen:
- `Workspace name` - Format: "1234 - Productie"
- `DisplayName` - Naam van de gebruiker
- `Report name` - Naam van het rapport
- `Aantal activity reportviews` - Aantal views
- `Jaar`, `Maand` - Periode

### Pipedrive Export
Verwachte kolommen:
- `Organisatie - Klantnummer` - 4-cijferige klantcode
- `Organisatie - Naam` - Klantnaam
- `Organisatie - Eigenaar` - Account eigenaar

## Toekomstige Uitbreidingen

- [ ] Power BI API directe integratie
- [ ] Pipedrive API koppeling
- [ ] Automatische alerts (email/Teams)
- [ ] Trend analyse (vergelijking met vorige periode)
- [ ] Power BI embedded dashboard

## Configuratie

In `app.py` kun je aanpassen:
- `INTERNE_MEDEWERKERS` - Lijst van Notifica medewerkers om uit te sluiten
- `SCORE_WEIGHTS` - Gewichten voor scoreberekening
- `SCORE_THRESHOLDS` - Drempelwaarden voor Groen/Oranje/Rood
