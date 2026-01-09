"""
Test Power BI API Connection
=============================
Test of de Power BI credentials werken.

Gebruik:
    python test_powerbi_connection.py
"""

import os
import sys

# Probeer .env te laden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Tip: pip install python-dotenv voor .env support")

def test_connection():
    """Test de Power BI API verbinding"""

    # Check credentials
    tenant_id = os.getenv('POWERBI_TENANT_ID')
    client_id = os.getenv('POWERBI_CLIENT_ID')
    client_secret = os.getenv('POWERBI_CLIENT_SECRET')

    print("=" * 60)
    print("Power BI API Connection Test")
    print("=" * 60)

    # Check of credentials aanwezig zijn
    print("\n1. Checking credentials...")

    if not tenant_id:
        print("   ❌ POWERBI_TENANT_ID niet gevonden")
        print("      Set via environment variable of .env bestand")
        return False
    else:
        print(f"   ✅ Tenant ID: {tenant_id[:8]}...")

    if not client_id:
        print("   ❌ POWERBI_CLIENT_ID niet gevonden")
        return False
    else:
        print(f"   ✅ Client ID: {client_id[:8]}...")

    if not client_secret:
        print("   ❌ POWERBI_CLIENT_SECRET niet gevonden")
        return False
    else:
        print(f"   ✅ Client Secret: {'*' * 20}")

    # Probeer te authenticeren
    print("\n2. Authenticating with Azure AD...")

    try:
        import requests
    except ImportError:
        print("   ❌ requests library niet geinstalleerd")
        print("      Run: pip install requests")
        return False

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://analysis.windows.net/powerbi/api/.default'
    }

    try:
        response = requests.post(token_url, data=data)

        if response.status_code == 200:
            token = response.json()['access_token']
            print(f"   ✅ Authentication successful!")
            print(f"      Token: {token[:50]}...")
        else:
            print(f"   ❌ Authentication failed: {response.status_code}")
            print(f"      Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

    # Test API call
    print("\n3. Testing Power BI API...")

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Probeer workspaces op te halen
    try:
        response = requests.get(
            "https://api.powerbi.com/v1.0/myorg/admin/groups?$top=5",
            headers=headers
        )

        if response.status_code == 200:
            workspaces = response.json().get('value', [])
            print(f"   ✅ API call successful!")
            print(f"      Found {len(workspaces)} workspaces")

            if workspaces:
                print("\n   Sample workspaces:")
                for ws in workspaces[:3]:
                    print(f"      - {ws.get('name', 'Unknown')}")
        else:
            print(f"   ❌ API call failed: {response.status_code}")
            print(f"      Error: {response.text}")

            if response.status_code == 403:
                print("\n   Tip: Je Service Principal heeft mogelijk geen Admin rechten.")
                print("   Ga naar Power BI Admin Portal > Tenant Settings > Developer settings")

            return False

    except Exception as e:
        print(f"   ❌ API error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed! Power BI connection is working.")
    print("=" * 60)

    return True


def interactive_setup():
    """Interactieve setup als credentials missen"""

    print("\n" + "=" * 60)
    print("Interactive Setup")
    print("=" * 60)

    print("\nGeen credentials gevonden. Wil je ze nu invoeren? (y/n)")

    if input().lower() != 'y':
        print("\nZie POWERBI_SETUP.md voor instructies.")
        return

    print("\nVoer je Azure AD credentials in:")

    tenant_id = input("Tenant ID: ").strip()
    client_id = input("Client ID: ").strip()
    client_secret = input("Client Secret: ").strip()

    # Maak .env bestand
    env_content = f"""# Power BI API Credentials
POWERBI_TENANT_ID={tenant_id}
POWERBI_CLIENT_ID={client_id}
POWERBI_CLIENT_SECRET={client_secret}
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("\n✅ .env bestand aangemaakt!")
    print("Run dit script opnieuw om te testen.")


if __name__ == "__main__":
    # Check of credentials bestaan
    if not any([
        os.getenv('POWERBI_TENANT_ID'),
        os.getenv('POWERBI_CLIENT_ID'),
        os.getenv('POWERBI_CLIENT_SECRET')
    ]):
        # Kijk of .env bestaat
        if not os.path.exists('.env'):
            interactive_setup()
        else:
            print("❌ .env bestand gevonden maar credentials ontbreken")
            print("   Check het bestand en vul de waardes in.")
    else:
        success = test_connection()
        sys.exit(0 if success else 1)
