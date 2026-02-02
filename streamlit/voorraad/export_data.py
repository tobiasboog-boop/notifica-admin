"""
Data Export Script voor Voorraad Dashboard
==========================================
Exporteert data van een klant naar lokale Parquet bestanden.
Deze bestanden kunnen dan worden meegeleverd met de app voor offline gebruik.

Gebruik:
    python export_data.py --customer 1256
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.database import SyntessDWHConnection, DatabaseConfig


def export_customer_data(customer_code: str, output_dir: str = "data"):
    """Export all relevant data for a customer to Parquet files."""

    print(f"Exporteren data voor klant {customer_code}...")

    # Create output directory
    output_path = Path(output_dir) / customer_code
    output_path.mkdir(parents=True, exist_ok=True)

    # Connect to database with explicit credentials
    config = DatabaseConfig(
        host="10.3.152.9",
        port=5432,
        database=customer_code,
        username="postgres",
        password="TQwSTtLM9bSaLD"
    )
    db = SyntessDWHConnection(config)

    # Test connection
    try:
        test = db.execute_query("SELECT 1 as test")
        print(f"[OK] Verbonden met database {customer_code}")
    except Exception as e:
        print(f"[FOUT] Kan niet verbinden: {e}")
        return False

    # Export each dataset
    datasets = {
        "magazijnen": db.get_magazijnen,
        "locaties": db.get_locaties,
        "posities": db.get_posities,
        "mutaties": db.get_mutaties,
        "tarieven": db.get_tarieven,
    }

    export_info = {
        "customer_code": customer_code,
        "export_date": datetime.now().isoformat(),
        "files": {},
    }

    for name, getter in datasets.items():
        try:
            print(f"  Exporteren {name}...", end=" ")
            df = getter()

            if df.empty:
                print(f"(leeg)")
                export_info["files"][name] = {"rows": 0, "status": "empty"}
            else:
                filepath = output_path / f"{name}.parquet"
                df.to_parquet(filepath, index=False)
                print(f"[OK] {len(df)} rijen")
                export_info["files"][name] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "status": "ok"
                }
        except Exception as e:
            print(f"[FOUT] {e}")
            export_info["files"][name] = {"status": "error", "error": str(e)}

    # Export balans waarde separately (single value)
    try:
        balans = db.get_balans_voorraad()
        export_info["balans_waarde"] = balans
        print(f"  Balanswaarde: € {balans:,.0f}")
    except Exception as e:
        export_info["balans_waarde"] = 0
        print(f"  Balanswaarde: niet beschikbaar ({e})")

    # Save export info
    import json
    info_path = output_path / "export_info.json"
    with open(info_path, "w") as f:
        json.dump(export_info, f, indent=2, default=str)

    print(f"\n[OK] Export voltooid naar: {output_path}")
    print(f"  Totale grootte: {sum(f.stat().st_size for f in output_path.glob('*')) / 1024:.1f} KB")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export klantdata voor offline gebruik")
    parser.add_argument("--customer", "-c", default="1256", help="Klantcode (default: 1256)")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")

    args = parser.parse_args()

    success = export_customer_data(args.customer, args.output)
    exit(0 if success else 1)
