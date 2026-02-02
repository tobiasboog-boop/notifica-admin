"""
Calculations for Voorraad Dashboard.

Includes metrics calculation, min-max analysis, and turnover calculations.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class VoorraadMetrics:
    """Container for voorraad KPIs."""
    totale_waarde: float
    totaal_aantal: int
    onder_minimum: int
    boven_maximum: int
    omloopsnelheid: float
    dagen_voorraad: float
    aantal_artikelen: int
    aantal_magazijnen: int


def calculate_voorraad_metrics(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame = None,
) -> VoorraadMetrics:
    """
    Calculate main voorraad KPIs.

    Args:
        posities: DataFrame with stock positions
        mutaties: DataFrame with stock mutations (for turnover calculation)

    Returns:
        VoorraadMetrics with calculated values
    """
    if posities.empty:
        return VoorraadMetrics(
            totale_waarde=0,
            totaal_aantal=0,
            onder_minimum=0,
            boven_maximum=0,
            omloopsnelheid=0,
            dagen_voorraad=0,
            aantal_artikelen=0,
            aantal_magazijnen=0,
        )

    # Basic metrics
    totale_waarde = posities["waarde"].sum() if "waarde" in posities.columns else 0
    totaal_aantal = posities["stand"].sum() if "stand" in posities.columns else 0

    # Status counts
    status_col = "status" if "status" in posities.columns else None
    if status_col:
        onder_minimum = (posities[status_col] == "Onder minimum").sum()
        boven_maximum = (posities[status_col] == "Boven maximum").sum()
    else:
        onder_minimum = 0
        boven_maximum = 0

    # Count unique items and warehouses
    aantal_artikelen = posities["tarief_key"].nunique() if "tarief_key" in posities.columns else 0
    aantal_magazijnen = posities["magazijn"].nunique() if "magazijn" in posities.columns else 0

    # Calculate turnover
    omloopsnelheid, dagen_voorraad = calculate_omloopsnelheid(posities, mutaties)

    return VoorraadMetrics(
        totale_waarde=totale_waarde,
        totaal_aantal=int(totaal_aantal),
        onder_minimum=int(onder_minimum),
        boven_maximum=int(boven_maximum),
        omloopsnelheid=omloopsnelheid,
        dagen_voorraad=dagen_voorraad,
        aantal_artikelen=int(aantal_artikelen),
        aantal_magazijnen=int(aantal_magazijnen),
    )


def calculate_min_max_status(row: pd.Series) -> str:
    """
    Calculate min-max status for a single position.

    Args:
        row: DataFrame row with stand, minimum, maximum columns

    Returns:
        Status string: "Geen voorraad", "Onder minimum", "OK", or "Boven maximum"
    """
    stand = row.get("stand", 0) or 0
    minimum = row.get("minimum", 0) or 0
    maximum = row.get("maximum", 0) or 0

    if stand <= 0:
        return "Geen voorraad"
    elif minimum > 0 and stand < minimum:
        return "Onder minimum"
    elif maximum > 0 and stand > maximum:
        return "Boven maximum"
    else:
        return "OK"


def calculate_omloopsnelheid(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame = None,
) -> tuple[float, float]:
    """
    Calculate inventory turnover and days of inventory.

    Omloopsnelheid = Yearly cost of goods sold / Average inventory value
    Dagen voorraad = 365 / Omloopsnelheid

    Args:
        posities: DataFrame with current stock positions
        mutaties: DataFrame with mutations (for COGS calculation)

    Returns:
        Tuple of (omloopsnelheid, dagen_voorraad)
    """
    if posities.empty:
        return 0.0, 0.0

    # Current inventory value
    current_value = posities["waarde"].sum() if "waarde" in posities.columns else 0

    if current_value <= 0:
        return 0.0, 0.0

    # Calculate yearly cost of goods sold from mutations
    if mutaties is not None and not mutaties.empty and "kostprijs" in mutaties.columns:
        # Filter to uitgiftes only
        if "mutatiesoort" in mutaties.columns:
            uitgiftes = mutaties[mutaties["mutatiesoort"] == "Uitgifte"]
        elif "uitgifte_aantal" in mutaties.columns:
            uitgiftes = mutaties[mutaties["uitgifte_aantal"] > 0]
        else:
            uitgiftes = mutaties

        yearly_cogs = uitgiftes["kostprijs"].sum()
    else:
        # Estimate based on typical turnover
        yearly_cogs = current_value * 4  # Assume 4x turnover as default

    # Calculate turnover
    omloopsnelheid = yearly_cogs / current_value if current_value > 0 else 0

    # Calculate days of inventory
    dagen_voorraad = 365 / omloopsnelheid if omloopsnelheid > 0 else 0

    return round(omloopsnelheid, 2), round(dagen_voorraad, 1)


def calculate_shortage(posities: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate shortage amounts for positions under minimum.

    Args:
        posities: DataFrame with positions

    Returns:
        DataFrame with tekort column added
    """
    if posities.empty:
        return posities

    df = posities.copy()

    if "minimum" in df.columns and "stand" in df.columns:
        df["tekort"] = df.apply(
            lambda r: max(0, (r["minimum"] or 0) - (r["stand"] or 0))
            if (r["minimum"] or 0) > 0 else 0,
            axis=1
        )
    else:
        df["tekort"] = 0

    return df


def aggregate_by_magazijn(posities: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stock positions by warehouse.

    Args:
        posities: DataFrame with positions

    Returns:
        DataFrame aggregated by magazijn
    """
    if posities.empty or "magazijn" not in posities.columns:
        return pd.DataFrame()

    return posities.groupby("magazijn").agg({
        "stand": "sum",
        "waarde": "sum",
        "positie_key": "count",
    }).rename(columns={"positie_key": "aantal_posities"}).reset_index()


def aggregate_by_status(posities: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stock positions by min-max status.

    Args:
        posities: DataFrame with positions and status column

    Returns:
        DataFrame with counts and values per status
    """
    if posities.empty or "status" not in posities.columns:
        return pd.DataFrame()

    return posities.groupby("status").agg({
        "stand": "sum",
        "waarde": "sum",
        "positie_key": "count",
    }).rename(columns={"positie_key": "aantal"}).reset_index()


def calculate_artikel_omloopsnelheid(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate turnover rate per article for top 10 slow/fast moving analysis.

    Omloopsnelheid per artikel = Jaaromzet uitgifte / Gemiddelde voorraadwaarde

    Args:
        posities: DataFrame with current stock positions
        mutaties: DataFrame with mutations

    Returns:
        DataFrame with artikel, omloopsnelheid, gemiddelde_ligduur, waarde
    """
    if posities.empty:
        return pd.DataFrame(columns=[
            "artikel", "tarief_key", "stand", "waarde",
            "uitgifte_jaar", "omloopsnelheid", "gemiddelde_ligduur_dagen"
        ])

    # Get current stock value per article
    artikel_voorraad = posities.groupby(["tarief_key", "artikel"]).agg({
        "stand": "sum",
        "waarde": "sum",
    }).reset_index()

    if mutaties is None or mutaties.empty:
        # No mutations - can't calculate turnover, set defaults
        artikel_voorraad["uitgifte_jaar"] = 0
        artikel_voorraad["omloopsnelheid"] = 0.0
        artikel_voorraad["gemiddelde_ligduur_dagen"] = 999  # Very slow
        return artikel_voorraad

    # Calculate yearly outgoing per article from mutations
    # First need to join mutations to posities to get tarief_key
    if "tarief_key" not in mutaties.columns:
        # Need to join through posities
        uitgifte_per_positie = mutaties.groupby("uitgifte_positie_key").agg({
            "kostprijs": "sum",
            "aantal": "sum",
        }).reset_index()
        uitgifte_per_positie.columns = ["positie_key", "uitgifte_waarde", "uitgifte_aantal"]

        # Join with posities to get tarief_key
        pos_tarief = posities[["positie_key", "tarief_key", "artikel"]].drop_duplicates()
        uitgifte_per_artikel = uitgifte_per_positie.merge(
            pos_tarief, on="positie_key", how="left"
        ).dropna(subset=["tarief_key"])

        uitgifte_per_artikel = uitgifte_per_artikel.groupby(["tarief_key"]).agg({
            "uitgifte_waarde": "sum",
        }).reset_index()
    else:
        # Direct calculation from mutaties with tarief_key
        if "mutatiesoort" in mutaties.columns:
            uitgiftes = mutaties[mutaties["mutatiesoort"] == "Uitgifte"]
        elif "uitgifte_aantal" in mutaties.columns:
            uitgiftes = mutaties[mutaties["uitgifte_aantal"] > 0]
        else:
            uitgiftes = mutaties

        uitgifte_per_artikel = uitgiftes.groupby("tarief_key").agg({
            "kostprijs": "sum",
        }).reset_index()
        uitgifte_per_artikel.columns = ["tarief_key", "uitgifte_waarde"]

    # Merge with voorraad
    artikel_voorraad = artikel_voorraad.merge(
        uitgifte_per_artikel, on="tarief_key", how="left"
    )
    artikel_voorraad["uitgifte_waarde"] = artikel_voorraad["uitgifte_waarde"].fillna(0)
    artikel_voorraad["uitgifte_jaar"] = artikel_voorraad["uitgifte_waarde"]

    # Calculate omloopsnelheid (turnover rate)
    # Omloopsnelheid = Kostprijs verkocht / Gemiddelde voorraadwaarde
    artikel_voorraad["omloopsnelheid"] = artikel_voorraad.apply(
        lambda r: round(r["uitgifte_jaar"] / r["waarde"], 2)
        if r["waarde"] > 0 else 0.0,
        axis=1
    )

    # Calculate gemiddelde ligduur (days of inventory)
    # Ligduur = 365 / Omloopsnelheid
    artikel_voorraad["gemiddelde_ligduur_dagen"] = artikel_voorraad.apply(
        lambda r: round(365 / r["omloopsnelheid"], 0)
        if r["omloopsnelheid"] > 0 else 999,  # 999 = effectively no movement
        axis=1
    )

    return artikel_voorraad


def get_top_slow_moving(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame,
    top_n: int = 10,
    min_waarde: float = 100,
) -> pd.DataFrame:
    """
    Get top N slowest moving articles (highest ligduur, lowest turnover).

    Args:
        posities: Stock positions
        mutaties: Stock mutations
        top_n: Number of articles to return
        min_waarde: Minimum stock value to consider (filter out trivial items)

    Returns:
        DataFrame with slowest moving articles
    """
    df = calculate_artikel_omloopsnelheid(posities, mutaties)

    if df.empty:
        return df

    # Filter for items with meaningful stock value
    df = df[df["waarde"] >= min_waarde]

    # Sort by ligduur descending (slowest first), then by waarde (highest value first)
    df = df.sort_values(
        ["gemiddelde_ligduur_dagen", "waarde"],
        ascending=[False, False]
    )

    return df.head(top_n)


def get_top_fast_moving(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame,
    top_n: int = 10,
    min_waarde: float = 100,
) -> pd.DataFrame:
    """
    Get top N fastest moving articles (lowest ligduur, highest turnover).

    Args:
        posities: Stock positions
        mutaties: Stock mutations
        top_n: Number of articles to return
        min_waarde: Minimum stock value to consider

    Returns:
        DataFrame with fastest moving articles
    """
    df = calculate_artikel_omloopsnelheid(posities, mutaties)

    if df.empty:
        return df

    # Filter for items with meaningful stock value AND some turnover
    df = df[(df["waarde"] >= min_waarde) & (df["omloopsnelheid"] > 0)]

    # Sort by ligduur ascending (fastest first)
    df = df.sort_values(["gemiddelde_ligduur_dagen", "waarde"], ascending=[True, False])

    return df.head(top_n)


def calculate_gemiddelde_ligduur(
    posities: pd.DataFrame,
    mutaties: pd.DataFrame,
) -> float:
    """
    Calculate overall average days of inventory (gemiddelde ligduur).

    This is a weighted average based on stock value.

    Args:
        posities: Stock positions
        mutaties: Stock mutations

    Returns:
        Average days of inventory
    """
    if posities.empty:
        return 0.0

    # Use overall turnover calculation
    omloopsnelheid, dagen = calculate_omloopsnelheid(posities, mutaties)
    return dagen


@dataclass
class BalansAansluiting:
    """Container for balance sheet reconciliation."""
    voorraad_berekend: float  # Calculated from mutations (SUM ontvangst - SUM uitgifte)
    voorraad_balans: float    # From balance sheet (if available)
    verschil: float           # Difference
    verschil_percentage: float
    status: str               # "OK", "Afwijking", "Geen balansdata"
    artikelen_met_negatief: int  # Items with negative stock (data issue indicator)


def check_balans_aansluiting(
    posities: pd.DataFrame,
    balans_waarde: float = None,
    tolerantie_percentage: float = 1.0,
) -> BalansAansluiting:
    """
    Check reconciliation between calculated stock value and balance sheet.

    The stock value is calculated as SUM(Aantal * Kostprijs) from positions,
    which should match the balance sheet asset for inventory.

    Args:
        posities: Stock positions with waarde column
        balans_waarde: Balance sheet inventory value (if available)
        tolerantie_percentage: Acceptable difference percentage

    Returns:
        BalansAansluiting with comparison results
    """
    if posities.empty:
        return BalansAansluiting(
            voorraad_berekend=0,
            voorraad_balans=balans_waarde or 0,
            verschil=0,
            verschil_percentage=0,
            status="Geen data",
            artikelen_met_negatief=0,
        )

    # Calculated stock value
    voorraad_berekend = posities["waarde"].sum() if "waarde" in posities.columns else 0

    # Count items with negative stock (indicates data issues)
    artikelen_met_negatief = 0
    if "stand" in posities.columns:
        artikelen_met_negatief = (posities["stand"] < 0).sum()

    if balans_waarde is None:
        return BalansAansluiting(
            voorraad_berekend=voorraad_berekend,
            voorraad_balans=0,
            verschil=0,
            verschil_percentage=0,
            status="Geen balansdata",
            artikelen_met_negatief=artikelen_met_negatief,
        )

    # Calculate difference
    verschil = voorraad_berekend - balans_waarde
    verschil_percentage = (verschil / balans_waarde * 100) if balans_waarde != 0 else 0

    # Determine status
    if abs(verschil_percentage) <= tolerantie_percentage:
        status = "OK"
    elif abs(verschil_percentage) <= 5.0:
        status = "Kleine afwijking"
    else:
        status = "Significante afwijking"

    return BalansAansluiting(
        voorraad_berekend=voorraad_berekend,
        voorraad_balans=balans_waarde,
        verschil=verschil,
        verschil_percentage=verschil_percentage,
        status=status,
        artikelen_met_negatief=artikelen_met_negatief,
    )


def validate_p_times_q(
    posities: pd.DataFrame,
    tarieven: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Validate P x Q calculation (Price x Quantity = Value).

    Checks if waarde == stand * kostprijs_per_stuk

    Args:
        posities: Stock positions
        tarieven: Tariff/pricing data (optional, for reference price check)

    Returns:
        DataFrame with validation results for items with discrepancies
    """
    if posities.empty:
        return pd.DataFrame()

    df = posities.copy()

    # Calculate implied kostprijs per stuk
    df["kostprijs_per_stuk"] = df.apply(
        lambda r: r["waarde"] / r["stand"] if r["stand"] != 0 else 0,
        axis=1
    )

    # If tarieven provided, add reference price
    if tarieven is not None and not tarieven.empty:
        # Check for tarief_key column (can be either case)
        tarief_col = "tarief_key" if "tarief_key" in tarieven.columns else "TariefKey"
        prijs_col = "verrekenprijs" if "verrekenprijs" in tarieven.columns else "Verrekenprijs"

        if "tarief_key" in df.columns and tarief_col in tarieven.columns:
            prijs_df = tarieven[[tarief_col, prijs_col]].rename(
                columns={tarief_col: "tarief_key", prijs_col: "referentie_prijs"}
            )
            df = df.merge(prijs_df, on="tarief_key", how="left")

            # Calculate deviation from reference price
            df["prijs_afwijking_pct"] = df.apply(
                lambda r: abs(r["kostprijs_per_stuk"] - (r["referentie_prijs"] or 0))
                / r["referentie_prijs"] * 100
                if r.get("referentie_prijs", 0) > 0 else None,
                axis=1
            )

    # Filter to items with issues (negative stock or high price deviation)
    issues = df[
        (df["stand"] < 0) |
        (df.get("prijs_afwijking_pct", pd.Series([0] * len(df))) > 50)
    ].copy()

    return issues
