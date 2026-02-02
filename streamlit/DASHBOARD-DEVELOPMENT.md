# Notifica Dashboard Development Guide

Dit document beschrijft de standaard patronen en best practices voor het ontwikkelen en deployen van Streamlit dashboards bij Notifica.

## Inhoudsopgave

1. [Branding & Styling](#branding--styling)
2. [Architectuur: DWH vs Lokale Data](#architectuur-dwh-vs-lokale-data)
3. [Klant-specifieke Deployments](#klant-specifieke-deployments)
4. [Authenticatie](#authenticatie)
5. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
6. [Checklist Nieuwe Dashboard](#checklist-nieuwe-dashboard)

---

## Branding & Styling

### Notifica Logo

**STANDAARD VOOR ALLE DASHBOARDS**: Het Notifica logo moet linksboven in de sidebar worden getoond.

```python
from pathlib import Path
import streamlit as st

# In de sidebar render functie:
logo_path = Path(__file__).parent / "assets" / "notifica_logo.jpg"
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)
```

**Bestandslocatie**: `streamlit/<app>/assets/notifica_logo.jpg`

Het logo bestand moet in elke dashboard folder aanwezig zijn onder `assets/`.

### Klantnaam in Titel

Bij klant-specifieke dashboards, toon de klantnaam prominent:

```python
st.sidebar.markdown(f"### 🏢 {customer_name}")
st.sidebar.markdown("*Dashboard exclusief voor deze klant*")
```

---

## Architectuur: DWH vs Lokale Data

### Ontwikkelomgeving (Intern)

Intern werken we met een directe verbinding naar het Data Warehouse:

- **Host**: `10.3.152.9` (intern IP, niet bereikbaar van buitenaf)
- **Port**: `5432` (PostgreSQL)
- **Database**: Klantcode (bijv. `1256`)

```python
# .streamlit/secrets.toml (lokaal)
[database]
host = "10.3.152.9"
port = 5432
database = "1256"
user = "postgres"
password = "..."
```

### Cloud Deployment (Extern)

Voor Streamlit Cloud deployments is de interne database **niet bereikbaar**. Gebruik daarom lokale Parquet bestanden:

```
streamlit/<app>/
├── data/
│   └── 1256/           # Klantcode als folder
│       ├── voorraad.parquet
│       ├── inkopen.parquet
│       └── ...
```

### Data Export Script

Elk dashboard moet een `export_data.py` script hebben om data te exporteren:

```bash
python export_data.py --customer 1256
```

Dit script:
1. Verbindt met de DWH
2. Haalt alle benodigde data op
3. Slaat op als Parquet bestanden in `data/<klantcode>/`

### Automatische Detectie

Het dashboard detecteert automatisch of lokale data beschikbaar is:

```python
def get_local_data_path() -> Optional[Path]:
    """Zoek naar lokale data folder."""
    possible_paths = [
        Path(__file__).parent.parent / "data",
        Path(__file__).parent.parent.parent / "data",
    ]
    for path in possible_paths:
        if path.exists() and any(path.iterdir()):
            return path
    return None

def is_local_data_mode() -> bool:
    return get_local_data_path() is not None
```

---

## Klant-specifieke Deployments

### Geen Klant Selector

Bij klant-specifieke deployments:
- **GEEN** dropdown om klant te selecteren
- **GEEN** demo data optie
- **GEEN** database configuratie UI

```python
def render_customer_selector():
    local_mode = is_local_data_mode()

    if local_mode:
        # Toon logo
        logo_path = Path(__file__).parent / "assets" / "notifica_logo.jpg"
        if logo_path.exists():
            st.sidebar.image(str(logo_path), use_container_width=True)

        # Toon klantnaam (uit secrets of folder)
        customer_name = st.secrets.get("customer", {}).get("name", "Klant")
        st.sidebar.markdown(f"### 🏢 {customer_name}")

        # GEEN selector - direct data laden
        return False, get_local_customer_code()

    # Anders: normale selector voor intern gebruik
    # ...
```

### Secrets Configuratie

```toml
# .streamlit/secrets.toml
[customer]
code = "1256"
name = "Van den Buijs"
```

---

## Authenticatie

### Wachtwoord Beveiliging

Gebruik SHA256 hashing voor wachtwoorden:

```python
import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
```

### Secrets Configuratie

```toml
# .streamlit/secrets.toml
[auth]
password_hash = "9ed1ef61fd976015748c38a69bd6bf2a50c42bf4444c622ada199c73e6ba1630"
```

### Hash Genereren

```bash
python -c "import hashlib; print(hashlib.sha256('JOUW_WACHTWOORD'.encode()).hexdigest())"
```

### Wachtwoord Richtlijnen

- Gebruik **willekeurige** wachtwoorden (niet klantnaam + jaar)
- Formaat: `Dashboard` + 6 willekeurige alfanumerieke tekens
- Voorbeeld: `Dashboard6A8130`

### Login Flow Fix

**Belangrijk**: Na succesvolle login moet `st.rerun()` worden aangeroepen:

```python
if st.button("Inloggen", type="primary", use_container_width=True):
    password_entered()
    if st.session_state.get("password_correct", False):
        st.rerun()  # ESSENTIEEL - anders blijft login scherm hangen
```

---

## Streamlit Cloud Deployment

### Monorepo Setup

Onze site is een monorepo. Deploy specifieke apps met subdirectory:

```
notifica_site/
├── src/                    # Website
├── streamlit/
│   ├── voorraad/          # App 1
│   ├── liquiditeit/       # App 2
│   └── ...
```

Bij Streamlit Cloud deployment:
- **Main file path**: `streamlit/voorraad/app.py`

### Deployment Checklist

1. [ ] Lokale data geëxporteerd naar `data/<klantcode>/`
2. [ ] Logo aanwezig in `assets/notifica_logo.jpg`
3. [ ] `requirements.txt` up-to-date
4. [ ] Secrets geconfigureerd in Streamlit Cloud:
   - `[database]` (optioneel voor lokale mode)
   - `[customer]` code en name
   - `[auth]` password_hash

### Admin Portal Link

Voeg nieuwe klant-dashboards toe aan `/admin/` onder "Klant Specifieke Dashboards":

```html
<a href="https://xxx.streamlit.app/" class="page-link" target="_blank">
    Klantnaam - Dashboard Naam
    <span class="url">streamlit.app</span>
    <span class="status active">Pilot</span>
</a>
```

---

## Checklist Nieuwe Dashboard

### Development

- [ ] Basis app structuur aangemaakt
- [ ] `src/database.py` met DWH én lokale data support
- [ ] `src/auth.py` voor wachtwoordbeveiliging
- [ ] `export_data.py` voor data export
- [ ] Notifica logo in `assets/`
- [ ] `requirements.txt`

### Klant Deployment

- [ ] Data geëxporteerd: `python export_data.py --customer XXXX`
- [ ] Secrets voorbereid (customer code/name, password hash)
- [ ] Getest met lokale data (verwijder database secrets tijdelijk)
- [ ] Streamlit Cloud app aangemaakt
- [ ] Secrets ingesteld in Streamlit Cloud
- [ ] Login getest met wachtwoord
- [ ] Link toegevoegd aan Admin Portal

### Wachtwoord Documentatie

Bewaar wachtwoorden veilig! Noteer in Admin Portal:
```
Wachtwoord voor [Klantnaam]: [Wachtwoord]
```

---

## Financiële Data (Balanswaarden)

### Architectuur Overzicht

De financiële data flow in Notifica:

```
DWH (PostgreSQL)          →    Power BI Semantic Model    →    Rapporten
─────────────────               ─────────────────────          ────────
financieel.Journaalregels       Journaalregels                 Balans
financieel.Rubrieken            Rubrieken                      W&V
stam.Administraties             CoA Balans mapping (RAAS)      etc.
```

### Balanswaarde Query

Voor voorraadwaarde uit de balans:

```sql
-- Rubrieken met Type='B' (Balans), codes 30xx/32xx (Voorraden)
SELECT SUM(
    CASE WHEN jr."Debet/Credit" = 'D' THEN jr."Bedrag" ELSE 0 END
    - CASE WHEN jr."Debet/Credit" = 'C' THEN jr."Bedrag" ELSE 0 END
) as balans_waarde
FROM financieel."Journaalregels" jr
JOIN financieel."Rubrieken" rub ON jr."RubriekKey" = rub."RubriekKey"
WHERE rub."Type" = 'B'
  AND (rub."Rubriek Code" LIKE '30%' OR rub."Rubriek Code" LIKE '32%')
  AND rub."Rubriek Code" NOT LIKE '31%'  -- Exclude onderhanden projecten
  AND EXTRACT(YEAR FROM jr."Boekdatum") = <jaar>
```

### Rubriek Codes (Voorbeeld Van den Buijs)

| Code | Omschrijving            | Categorie                        |
| ---- | ----------------------- | -------------------------------- |
| 3000 | Voorraden               | Vlottende Activa                 |
| 3002 | Correctie voorraden     | Vlottende Activa                 |
| 3100 | Onderhanden projecten   | Vlottende Activa (NIET voorraad) |
| 3210 | Voorraad showroom       | Vlottende Activa                 |

**Let op**: Codes 31xx zijn onderhanden projecten, NIET voorraden!

### TypePeriode

Journaalregels hebben een `TypePeriode` kolom:

- **Openingsperiode**: Beginbalans (1 januari)
- **Periodiek**: Normale boekingen gedurende het jaar
- **Afsluitperiode**: Eindejaar correcties

De balansstand = Openingsperiode + Periodiek voor het lopende jaar.

### DWH vs Power BI Discrepantie

Waarom balanswaarden kunnen verschillen:

1. **Refresh datum**: DWH wordt periodiek ververst, Power BI kan actueler zijn
2. **Administratie filter**: Power BI filtert vaak op specifieke administraties
3. **RAAS mapping**: Power BI gebruikt CoA mapping uit RAAS appdata

---

## Referentie Implementatie

Zie `streamlit/voorraad/` voor een complete referentie-implementatie van:
- Dual-mode database (DWH + lokaal)
- Klant-specifieke deployment
- Wachtwoordbeveiliging
- Notifica branding

---

*Laatst bijgewerkt: Januari 2026*
