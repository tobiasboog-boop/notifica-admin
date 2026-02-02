# Streamlit Launcher Service

Automatische launcher voor Streamlit dashboards die lokaal draaien.

## Wat doet dit?

Deze launcher is een kleine Flask server die op de achtergrond draait en Streamlit apps automatisch start wanneer je ze opent via de admin portal.

## Setup (eenmalig)

1. **Installeer dependencies:**
   ```bash
   pip install -r streamlit/launcher/requirements.txt
   ```

2. **Start de launcher:**
   - Dubbelklik op `START_LAUNCHER.bat`
   - Of run: `python streamlit/launcher/launcher.py`

3. **Optioneel - Start bij Windows opstarten:**
   - Druk `Win + R`
   - Type: `shell:startup`
   - Maak een shortcut naar `START_LAUNCHER.bat` in deze map

## Hoe werkt het?

1. Launcher draait op poort `8500`
2. Admin portal pagina's roepen automatisch de launcher aan via:
   - `http://localhost:8500/start/liquiditeit` → start liquiditeit dashboard op poort 8501
   - `http://localhost:8500/start/voorraad` → start voorraad dashboard op poort 8503
3. Als de app al draait, doet de launcher niks
4. Als de launcher niet draait, valt de pagina terug op de oude detectie methode

## Endpoints

- `GET /health` - Check of launcher draait
- `GET /start/liquiditeit` - Start liquiditeit dashboard
- `GET /start/voorraad` - Start voorraad dashboard
- `GET /status/<app_name>` - Check of een app draait

## Voor collega's

Als je niet Tobias bent en deze pagina's opent, krijg je een melding dat de dashboards alleen op Tobias' computer werken (vanwege DWH toegang).
