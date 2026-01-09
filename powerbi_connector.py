"""
Power BI API Connector
======================
Haalt automatisch activity data op uit Power BI via de Admin API.

Vereisten:
- Azure AD App Registration met Power BI Admin rechten
- Service Principal of User credentials
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class PowerBIConnector:
    """Connector voor Power BI REST API"""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        """
        Initialize connector met Azure AD credentials.

        Args:
            tenant_id: Azure AD Tenant ID
            client_id: App Registration Client ID
            client_secret: App Registration Client Secret
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.base_url = "https://api.powerbi.com/v1.0/myorg"

    def authenticate(self) -> bool:
        """
        Authenticate met Azure AD en verkrijg access token.

        Returns:
            bool: True als authenticatie succesvol
        """
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'https://analysis.windows.net/powerbi/api/.default'
        }

        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            self.access_token = response.json()['access_token']
            return True
        except Exception as e:
            print(f"Authenticatie fout: {e}")
            return False

    def _get_headers(self) -> dict:
        """Return headers met access token"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def get_workspaces(self) -> pd.DataFrame:
        """
        Haal alle workspaces op.

        Returns:
            DataFrame met workspace info
        """
        url = f"{self.base_url}/admin/groups?$top=5000"

        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()

        data = response.json().get('value', [])
        return pd.DataFrame(data)

    def get_activity_events(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Haal activity events op voor een periode.

        Args:
            start_date: Start datum (YYYY-MM-DD)
            end_date: Eind datum (YYYY-MM-DD), default vandaag

        Returns:
            DataFrame met activity events
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Power BI API accepteert ISO 8601 format
        start_dt = f"{start_date}T00:00:00.000Z"
        end_dt = f"{end_date}T23:59:59.999Z"

        url = f"{self.base_url}/admin/activityevents"
        params = {
            'startDateTime': f"'{start_dt}'",
            'endDateTime': f"'{end_dt}'"
        }

        all_events = []
        continuation_token = None

        while True:
            if continuation_token:
                params['continuationToken'] = f"'{continuation_token}'"

            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()

            data = response.json()
            events = data.get('activityEventEntities', [])
            all_events.extend(events)

            continuation_token = data.get('continuationToken')
            if not continuation_token:
                break

        return pd.DataFrame(all_events)

    def get_report_views(self, days: int = 180) -> pd.DataFrame:
        """
        Haal report views op voor de afgelopen X dagen.

        Args:
            days: Aantal dagen terug

        Returns:
            DataFrame met report views per user/report/workspace
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        events = self.get_activity_events(start_date)

        # Filter op ViewReport activity
        if 'Activity' in events.columns:
            report_views = events[events['Activity'] == 'ViewReport'].copy()
        else:
            report_views = events.copy()

        return report_views

    def get_workspace_reports(self, workspace_id: str) -> pd.DataFrame:
        """
        Haal alle reports op voor een workspace.

        Args:
            workspace_id: ID van de workspace

        Returns:
            DataFrame met report info
        """
        url = f"{self.base_url}/admin/groups/{workspace_id}/reports"

        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()

        data = response.json().get('value', [])
        return pd.DataFrame(data)


def fetch_and_process_data(connector: PowerBIConnector, days: int = 180) -> pd.DataFrame:
    """
    Haal data op en verwerk naar zelfde format als Excel export.

    Args:
        connector: Geauthenticeerde PowerBIConnector
        days: Aantal dagen terug

    Returns:
        DataFrame in zelfde format als handmatige export
    """
    # Haal activity data op
    events = connector.get_report_views(days)

    if events.empty:
        return pd.DataFrame()

    # Map naar verwacht format
    result = pd.DataFrame({
        'Workspace name': events.get('WorkspaceName', events.get('workspaceName', '')),
        'DisplayName': events.get('UserKey', events.get('userKey', '')),
        'Report name': events.get('ReportName', events.get('reportName', '')),
        'Aantal activity reportviews': 1,  # Elke rij is 1 view
        'Datum': pd.to_datetime(events.get('CreationTime', events.get('creationTime', '')))
    })

    # Aggregeer per combinatie
    result = result.groupby(['Workspace name', 'DisplayName', 'Report name']).agg({
        'Aantal activity reportviews': 'sum',
        'Datum': 'max'
    }).reset_index()

    # Voeg jaar/maand toe
    result['Jaar'] = result['Datum'].dt.year
    result['Maand'] = result['Datum'].dt.month_name()

    return result


# === CLI INTERFACE ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Power BI Activity Data Fetcher')
    parser.add_argument('--tenant', required=True, help='Azure AD Tenant ID')
    parser.add_argument('--client-id', required=True, help='App Client ID')
    parser.add_argument('--client-secret', required=True, help='App Client Secret')
    parser.add_argument('--days', type=int, default=180, help='Dagen terug (default: 180)')
    parser.add_argument('--output', default='powerbi_activity.xlsx', help='Output bestand')

    args = parser.parse_args()

    # Connect en fetch
    connector = PowerBIConnector(args.tenant, args.client_id, args.client_secret)

    print("Authenticeren...")
    if not connector.authenticate():
        print("Authenticatie mislukt!")
        exit(1)

    print(f"Ophalen activity data ({args.days} dagen)...")
    data = fetch_and_process_data(connector, args.days)

    if data.empty:
        print("Geen data gevonden!")
        exit(1)

    print(f"Opslaan naar {args.output}...")
    data.to_excel(args.output, index=False)

    print(f"Klaar! {len(data)} records opgeslagen.")
