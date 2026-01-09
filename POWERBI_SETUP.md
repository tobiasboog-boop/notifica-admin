# Power BI API Koppeling Instellen

## Wat heb je nodig?

Om automatisch data uit Power BI op te halen heb je een van deze opties nodig:

### Optie 1: Service Principal (Aanbevolen voor productie)

1. **Azure AD App Registration maken**
   - Ga naar https://portal.azure.com
   - Azure Active Directory > App registrations > New registration
   - Naam: "Notifica Customer Health"
   - Kopieer: Application (client) ID en Directory (tenant) ID

2. **Client Secret aanmaken**
   - In de App Registration > Certificates & secrets
   - New client secret
   - Kopieer de secret value (alleen nu zichtbaar!)

3. **Power BI Admin rechten geven**
   - Ga naar https://app.powerbi.com
   - Settings (tandwiel) > Admin portal
   - Tenant settings > Developer settings
   - "Service principals can use Fabric APIs" > Enabled
   - Voeg je App toe aan een Security Group die toegang heeft

4. **API Permissions toevoegen**
   - In Azure AD App Registration > API permissions
   - Add permission > Power BI Service
   - Voeg toe:
     - Tenant.Read.All
     - Report.Read.All
     - Workspace.Read.All

### Optie 2: Power BI REST API met User Token (Snel testen)

Voor snel testen kun je ook je eigen user token gebruiken:

1. Ga naar https://docs.microsoft.com/en-us/rest/api/power-bi/
2. Klik "Try it" op een API endpoint
3. Log in en kopieer de Bearer token uit de request

---

## Credentials invullen

Maak een bestand `.env` aan in de `notifica_customer_health` folder:

```env
# Azure AD / Power BI credentials
POWERBI_TENANT_ID=jouw-tenant-id-hier
POWERBI_CLIENT_ID=jouw-client-id-hier
POWERBI_CLIENT_SECRET=jouw-client-secret-hier

# Pipedrive API (optioneel)
PIPEDRIVE_API_KEY=jouw-pipedrive-api-key
PIPEDRIVE_COMPANY_DOMAIN=notifica
```

---

## Snelle Test

Nadat je de credentials hebt ingevuld, test met:

```bash
python test_powerbi_connection.py
```

---

## Veelvoorkomende Problemen

### "Unauthorized" fout
- Check of de Service Principal toegang heeft in Power BI Admin portal
- Wacht 15-30 minuten na het instellen van permissions

### "Forbidden" fout
- De App heeft niet de juiste API permissions
- Admin consent is niet gegeven

### Geen data
- Check of je workspaces "XXXX - Productie" heten
- Activity data is alleen beschikbaar voor admins

---

## Handmatige Export (Alternatief)

Als de API koppeling niet lukt, kun je ook handmatig exporteren:

1. Ga naar Power BI Admin Portal
2. Usage metrics > Activity log
3. Export naar Excel
4. Upload in de Streamlit app
