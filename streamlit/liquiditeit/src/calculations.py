"""
Liquiditeit Dashboard - Calculations
====================================
Business logic voor liquiditeitsberekeningen en cashflow prognoses.

Methoden:
- DSO per debiteur: Adjusted Due Date logica op basis van historisch betaalgedrag
- Fading Weight Ensemble: Glijdende schaal tussen ERP data en statistische forecast
- ML Forecast: Seizoenspatronen, trend en weighted moving average
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LiquidityMetrics:
    """Container for liquidity KPIs."""
    current_ratio: float
    quick_ratio: float
    cash_position: float
    total_receivables: float
    total_payables: float
    net_working_capital: float
    days_cash_on_hand: float


def calculate_liquidity_metrics(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    avg_daily_expenses: float = 5000.0
) -> LiquidityMetrics:
    """
    Calculate key liquidity metrics.

    Args:
        banksaldo: DataFrame with bank balances
        debiteuren: DataFrame with accounts receivable
        crediteuren: DataFrame with accounts payable
        avg_daily_expenses: Average daily operating expenses

    Returns:
        LiquidityMetrics with calculated KPIs
    """
    cash = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    receivables = debiteuren["openstaand"].sum() if not debiteuren.empty else 0
    payables = crediteuren["openstaand"].sum() if not crediteuren.empty else 0

    # Current ratio = Current Assets / Current Liabilities
    current_assets = cash + receivables
    current_liabilities = payables if payables > 0 else 1  # Avoid division by zero

    current_ratio = current_assets / current_liabilities

    # Quick ratio = (Current Assets - Inventory) / Current Liabilities
    # For service companies, quick ratio ≈ current ratio (no inventory)
    quick_ratio = (cash + receivables) / current_liabilities

    # Net working capital
    net_working_capital = current_assets - payables

    # Days cash on hand
    days_cash = cash / avg_daily_expenses if avg_daily_expenses > 0 else 0

    return LiquidityMetrics(
        current_ratio=round(current_ratio, 2),
        quick_ratio=round(quick_ratio, 2),
        cash_position=cash,
        total_receivables=receivables,
        total_payables=payables,
        net_working_capital=net_working_capital,
        days_cash_on_hand=round(days_cash, 1),
    )


# =============================================================================
# FASE 1: DSO PER DEBITEUR - ADJUSTED DUE DATE LOGICA
# =============================================================================

def calculate_dso_adjustment(
    betaalgedrag: pd.DataFrame,
    standaard_betaaltermijn: int = 30,
    fallback_dagen: float = None
) -> Dict[str, float]:
    """
    Bereken de DSO correctie per debiteur.

    Voor elke debiteur berekenen we hoeveel dagen EXTRA (of minder)
    we moeten optellen bij de vervaldatum om de verwachte betaaldatum te krijgen.

    Args:
        betaalgedrag: DataFrame met kolommen:
            - debiteur_code
            - gem_dagen_tot_betaling
            - betrouwbaarheid (0-1)
        standaard_betaaltermijn: Standaard betaaltermijn in dagen (bijv. 30)
        fallback_dagen: Fallback voor onbekende debiteuren (default: gemiddelde)

    Returns:
        Dict met debiteur_code -> extra_dagen (kan negatief zijn voor snelle betalers)
    """
    if betaalgedrag.empty:
        return {"_fallback": 0}

    # Bereken gemiddelde als fallback
    gem_alle = betaalgedrag["gem_dagen_tot_betaling"].mean()
    if fallback_dagen is None:
        fallback_dagen = gem_alle - standaard_betaaltermijn

    adjustments = {"_fallback": round(fallback_dagen, 1)}

    for _, row in betaalgedrag.iterrows():
        debiteur = row["debiteur_code"]
        gem_dagen = row["gem_dagen_tot_betaling"]
        betrouwbaarheid = row.get("betrouwbaarheid", 0.5)

        # Extra dagen = werkelijke betaaltijd - standaard termijn
        # Positief = betaalt later dan termijn
        # Negatief = betaalt eerder dan termijn
        extra_dagen = gem_dagen - standaard_betaaltermijn

        # Weeg de correctie met betrouwbaarheid
        # Lage betrouwbaarheid -> correctie richting gemiddelde
        gewogen_extra = (
            betrouwbaarheid * extra_dagen +
            (1 - betrouwbaarheid) * (gem_alle - standaard_betaaltermijn)
        )

        adjustments[debiteur] = round(gewogen_extra, 1)

    return adjustments


def adjust_receivables_due_dates(
    debiteuren: pd.DataFrame,
    dso_adjustments: Dict[str, float],
    date_column: str = "vervaldatum"
) -> pd.DataFrame:
    """
    Pas de vervaldatums aan op basis van historisch betaalgedrag per debiteur.

    Dit is de kern van de "Adjusted Due Date" methode:
    Verwachte betaaldatum = Vervaldatum + DSO correctie per klant

    Args:
        debiteuren: DataFrame met openstaande debiteuren
        dso_adjustments: Dict van debiteur_code -> extra_dagen
        date_column: Naam van de datum kolom

    Returns:
        DataFrame met extra kolom 'verwachte_betaling'
    """
    if debiteuren.empty:
        return debiteuren

    df = debiteuren.copy()

    # Zorg dat vervaldatum een date is
    df[date_column] = pd.to_datetime(df[date_column]).dt.date

    # Fallback voor onbekende debiteuren
    fallback = dso_adjustments.get("_fallback", 0)

    def get_expected_date(row):
        debiteur = row.get("debiteur_code", row.get("debiteur_naam", ""))
        vervaldatum = row[date_column]

        if pd.isna(vervaldatum):
            return None

        # Zoek DSO correctie voor deze debiteur
        extra_dagen = dso_adjustments.get(debiteur, fallback)

        # Bereken verwachte betaaldatum
        if isinstance(vervaldatum, datetime):
            return (vervaldatum + timedelta(days=extra_dagen)).date()
        else:
            return vervaldatum + timedelta(days=extra_dagen)

    df["verwachte_betaling"] = df.apply(get_expected_date, axis=1)
    df["dso_correctie_dagen"] = df.apply(
        lambda row: dso_adjustments.get(
            row.get("debiteur_code", row.get("debiteur_naam", "")),
            fallback
        ),
        axis=1
    )

    return df


# =============================================================================
# FASE 2: FADING WEIGHT ENSEMBLE MODEL
# =============================================================================

def sigmoid_fading_weight(week: int, midpoint: int = 5, steepness: float = 1.5) -> Tuple[float, float]:
    """
    Bereken fading weights met sigmoid functie voor natuurlijke overgang.

    Week 1-2: ~90-95% ERP data (bekende posten)
    Week 5-6: ~50% ERP, 50% statistiek
    Week 10+: ~10-20% ERP, 80-90% statistiek

    Args:
        week: Week nummer (1 = eerste forecast week)
        midpoint: Week waar 50/50 split is (default: 5)
        steepness: Hoe snel de overgang is (hoger = steilere curve)

    Returns:
        Tuple van (weight_erp, weight_statistiek)
    """
    # Sigmoid: 1 / (1 + e^((week - midpoint) / steepness))
    weight_erp = 1 / (1 + math.exp((week - midpoint) / steepness))

    # Begrens tussen 0.05 en 0.95 (nooit 100% één bron)
    weight_erp = max(0.05, min(0.95, weight_erp))
    weight_stat = 1 - weight_erp

    return round(weight_erp, 3), round(weight_stat, 3)


def calculate_ensemble_forecast_week(
    week_num: int,
    bekende_inkomsten: float,
    bekende_uitgaven: float,
    stat_inkomsten: float,
    stat_uitgaven: float,
    midpoint: int = 5
) -> Tuple[float, float, float, float, str]:
    """
    Bereken de ensemble forecast voor één week.

    Combineert bekende openstaande posten (ERP) met statistische forecast
    met een glijdende schaal gebaseerd op de voorspelhorizon.

    Args:
        week_num: Week nummer (1 = eerste forecast week)
        bekende_inkomsten: Som van openstaande debiteuren met verwachte betaling in deze week
        bekende_uitgaven: Som van openstaande crediteuren met verwachte betaling in deze week
        stat_inkomsten: Statistische forecast inkomsten (ML model output)
        stat_uitgaven: Statistische forecast uitgaven (ML model output)
        midpoint: Week waar 50/50 split is

    Returns:
        Tuple van (forecast_in, forecast_uit, weight_erp, weight_stat, methode_beschrijving)
    """
    w_erp, w_stat = sigmoid_fading_weight(week_num, midpoint)

    # Ensemble berekening
    # Als er bekende posten zijn, gebruik die met w_erp gewicht
    # Anders valt de w_erp component weg en gebruiken we alleen stat

    if bekende_inkomsten > 0 or bekende_uitgaven > 0:
        # We hebben ERP data
        forecast_in = (w_erp * bekende_inkomsten) + (w_stat * stat_inkomsten)
        forecast_uit = (w_erp * bekende_uitgaven) + (w_stat * stat_uitgaven)
        methode = f"ensemble: ERP {w_erp*100:.0f}% + stat {w_stat*100:.0f}%"
    else:
        # Geen ERP data voor deze week - gebruik alleen statistiek
        # maar met lagere confidence (zie create_fading_weight_forecast)
        forecast_in = stat_inkomsten
        forecast_uit = stat_uitgaven
        methode = f"statistisch (geen bekende posten)"

    return forecast_in, forecast_uit, w_erp, w_stat, methode


def create_fading_weight_forecast(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    historische_cashflow: pd.DataFrame,
    betaalgedrag_debiteuren: pd.DataFrame,
    betaalgedrag_crediteuren: pd.DataFrame = None,
    weeks: int = 13,
    weeks_history: int = 4,
    reference_date=None,
    standaard_betaaltermijn: int = 30,
    ensemble_midpoint: int = 5,
) -> Tuple[pd.DataFrame, int, Dict]:
    """
    Creëer cashflow forecast met Fading Weight Ensemble methode.

    Deze methode combineert:
    1. DSO-gecorrigeerde openstaande posten (korte termijn, hoge zekerheid)
    2. Statistische forecast uit ML model (lange termijn, lagere zekerheid)
    3. Sigmoid-based fading weights voor natuurlijke overgang

    Args:
        banksaldo: Huidige banksaldi
        debiteuren: Openstaande debiteuren
        crediteuren: Openstaande crediteuren
        historische_cashflow: Historische wekelijkse cashflow
        betaalgedrag_debiteuren: DSO data per debiteur
        betaalgedrag_crediteuren: DPO data per crediteur (optioneel)
        weeks: Aantal weken forecast
        weeks_history: Aantal weken realisatie data
        reference_date: Standdatum
        standaard_betaaltermijn: Standaard betaaltermijn voor DSO berekening
        ensemble_midpoint: Week waar 50/50 split is

    Returns:
        Tuple van (forecast DataFrame, forecast_start_idx, metadata dict)
    """
    if reference_date is None:
        reference_date = datetime.now().date()
    elif isinstance(reference_date, datetime):
        reference_date = reference_date.date()

    # =========================================================================
    # STAP 1: Bereken DSO correcties per debiteur
    # =========================================================================
    dso_adjustments = calculate_dso_adjustment(
        betaalgedrag_debiteuren,
        standaard_betaaltermijn=standaard_betaaltermijn
    )

    # Pas vervaldatums aan op basis van DSO
    debiteuren_adjusted = adjust_receivables_due_dates(debiteuren, dso_adjustments)

    # Crediteur DPO (optioneel - vaak betalen we zelf op tijd)
    if betaalgedrag_crediteuren is not None and not betaalgedrag_crediteuren.empty:
        dpo_adjustments = calculate_dso_adjustment(
            betaalgedrag_crediteuren.rename(columns={"crediteur_code": "debiteur_code"}),
            standaard_betaaltermijn=standaard_betaaltermijn
        )
        crediteuren_adjusted = adjust_receivables_due_dates(
            crediteuren.rename(columns={"crediteur_code": "debiteur_code"}),
            dpo_adjustments
        ).rename(columns={"debiteur_code": "crediteur_code"})
    else:
        # Geen DPO data - gebruik vervaldatum direct
        crediteuren_adjusted = crediteuren.copy()
        if not crediteuren_adjusted.empty and "vervaldatum" in crediteuren_adjusted.columns:
            crediteuren_adjusted["verwachte_betaling"] = pd.to_datetime(
                crediteuren_adjusted["vervaldatum"]
            ).dt.date

    # =========================================================================
    # STAP 2: Leer statistisch model uit historische data
    # =========================================================================
    pattern = learn_weekly_pattern(historische_cashflow)

    all_rows = []

    # =========================================================================
    # STAP 3: REALISATIE - Historische weken
    # =========================================================================
    if weeks_history > 0 and not historische_cashflow.empty:
        hist = historische_cashflow.copy()
        hist["week_start"] = pd.to_datetime(hist["week_start"]).dt.date
        hist = hist[hist["week_start"] < reference_date]
        hist = hist.sort_values("week_start", ascending=False).head(weeks_history)
        hist = hist.sort_values("week_start", ascending=True)

        for i, row in enumerate(hist.itertuples()):
            week_num = -(weeks_history - i)
            inkomsten = getattr(row, "inkomsten", 0)
            uitgaven = getattr(row, "uitgaven", 0)
            netto = inkomsten - uitgaven

            all_rows.append({
                "week_nummer": week_num,
                "week_label": f"Week {week_num}",
                "week_start": row.week_start,
                "week_eind": row.week_start + timedelta(days=7),
                "maand": row.week_start.month if hasattr(row.week_start, 'month') else 0,
                "inkomsten_debiteuren": round(inkomsten, 2),
                "uitgaven_crediteuren": round(uitgaven, 2),
                "netto_cashflow": round(netto, 2),
                "data_type": "Realisatie",
                "is_realisatie": True,
                "confidence": 1.0,
                "weight_erp": 1.0,
                "weight_stat": 0.0,
                "methode": "realisatie",
            })

    forecast_start_idx = len(all_rows)

    # =========================================================================
    # STAP 4: PROGNOSE - Fading Weight Ensemble
    # =========================================================================
    week_starts = [reference_date + timedelta(weeks=i) for i in range(weeks + 1)]

    for i in range(weeks):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]
        week_maand = week_start.month
        week_num = i + 1

        # --- Bekende inkomsten uit DSO-gecorrigeerde debiteuren ---
        bekende_in = 0.0
        if not debiteuren_adjusted.empty and "verwachte_betaling" in debiteuren_adjusted.columns:
            mask = (
                (debiteuren_adjusted["verwachte_betaling"] >= week_start) &
                (debiteuren_adjusted["verwachte_betaling"] < week_end)
            )
            bekende_in = debiteuren_adjusted.loc[mask, "openstaand"].sum()

        # --- Bekende uitgaven uit crediteuren ---
        bekende_uit = 0.0
        if not crediteuren_adjusted.empty and "verwachte_betaling" in crediteuren_adjusted.columns:
            mask = (
                (crediteuren_adjusted["verwachte_betaling"] >= week_start) &
                (crediteuren_adjusted["verwachte_betaling"] < week_end)
            )
            bekende_uit = crediteuren_adjusted.loc[mask, "openstaand"].sum()

        # --- Statistische forecast uit ML model ---
        stat_in, stat_uit, ml_conf, ml_methode = predict_week_ml(
            week_idx=i,
            week_maand=week_maand,
            pattern=pattern,
            bekende_in=0,  # Geef 0 mee, we doen zelf de blending
            bekende_uit=0,
            weeks_ahead=week_num
        )

        # --- Ensemble berekening ---
        forecast_in, forecast_uit, w_erp, w_stat, methode = calculate_ensemble_forecast_week(
            week_num=week_num,
            bekende_inkomsten=bekende_in,
            bekende_uitgaven=bekende_uit,
            stat_inkomsten=stat_in,
            stat_uitgaven=stat_uit,
            midpoint=ensemble_midpoint
        )

        # --- Confidence berekening ---
        # Basis: sigmoid weight (hoe meer ERP, hoe hoger confidence)
        base_confidence = w_erp * 0.95 + w_stat * ml_conf

        # Verlaag confidence als geen bekende posten
        if bekende_in == 0 and bekende_uit == 0:
            base_confidence *= 0.8

        confidence = round(min(0.95, max(0.2, base_confidence)), 2)

        netto = forecast_in - forecast_uit

        all_rows.append({
            "week_nummer": week_num,
            "week_label": f"Week {week_num}",
            "week_start": week_start,
            "week_eind": week_end,
            "maand": week_maand,
            "inkomsten_debiteuren": round(forecast_in, 2),
            "uitgaven_crediteuren": round(forecast_uit, 2),
            "inkomsten_bekend": round(bekende_in, 2),
            "uitgaven_bekend": round(bekende_uit, 2),
            "inkomsten_stat": round(stat_in, 2),
            "uitgaven_stat": round(stat_uit, 2),
            "netto_cashflow": round(netto, 2),
            "data_type": "Prognose",
            "is_realisatie": False,
            "confidence": confidence,
            "weight_erp": w_erp,
            "weight_stat": w_stat,
            "methode": methode,
        })

    df = pd.DataFrame(all_rows)

    # Bereken cumulatief saldo
    start_balance = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    df["cumulatief_saldo"] = start_balance + df["netto_cashflow"].cumsum()

    # Metadata voor debugging/rapportage
    metadata = {
        "methode": "fading_weight_ensemble",
        "dso_adjustments": dso_adjustments,
        "ensemble_midpoint": ensemble_midpoint,
        "standaard_betaaltermijn": standaard_betaaltermijn,
        "pattern_has_data": pattern.get("has_pattern", False),
        "n_debiteuren_met_dso": len([k for k in dso_adjustments.keys() if k != "_fallback"]),
    }

    return df, forecast_start_idx, metadata


def create_weekly_cashflow_forecast(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    salarissen: pd.DataFrame,
    weeks: int = 13,
    debiteur_delay_days: int = 0,
    crediteur_delay_days: int = 0,
) -> pd.DataFrame:
    """
    Create a weekly cashflow forecast for the specified number of weeks.

    Args:
        banksaldo: Current bank balances
        debiteuren: Outstanding receivables with due dates
        crediteuren: Outstanding payables with due dates
        salarissen: Planned salary payments
        weeks: Number of weeks to forecast
        debiteur_delay_days: Scenario: extra days before customers pay
        crediteur_delay_days: Scenario: extra days before we pay suppliers

    Returns:
        DataFrame with weekly cashflow forecast
    """
    today = datetime.now().date()

    # Create weekly buckets
    week_starts = [today + timedelta(weeks=i) for i in range(weeks + 1)]
    week_labels = [f"Week {i+1}" for i in range(weeks)]

    # Initialize result DataFrame
    forecast = pd.DataFrame({
        "week_nummer": range(1, weeks + 1),
        "week_label": week_labels,
        "week_start": week_starts[:-1],
        "week_eind": week_starts[1:],
        "inkomsten_debiteuren": 0.0,
        "uitgaven_crediteuren": 0.0,
        "uitgaven_salarissen": 0.0,
        "uitgaven_overig": 0.0,
        "netto_cashflow": 0.0,
        "cumulatief_saldo": 0.0,
    })

    # Aggregate debiteuren per week (with optional delay scenario)
    if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
        deb = debiteuren.copy()
        deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
        deb["verwachte_betaling"] = deb["vervaldatum"] + timedelta(days=debiteur_delay_days)

        for idx, row in forecast.iterrows():
            week_start = row["week_start"]
            week_end = row["week_eind"]
            mask = (deb["verwachte_betaling"] >= week_start) & (deb["verwachte_betaling"] < week_end)
            forecast.loc[idx, "inkomsten_debiteuren"] = deb.loc[mask, "openstaand"].sum()

    # Aggregate crediteuren per week (with optional delay scenario)
    if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
        cred = crediteuren.copy()
        cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
        cred["verwachte_betaling"] = cred["vervaldatum"] + timedelta(days=crediteur_delay_days)

        for idx, row in forecast.iterrows():
            week_start = row["week_start"]
            week_end = row["week_eind"]
            mask = (cred["verwachte_betaling"] >= week_start) & (cred["verwachte_betaling"] < week_end)
            forecast.loc[idx, "uitgaven_crediteuren"] = cred.loc[mask, "openstaand"].sum()

    # Add salary payments
    if not salarissen.empty and "betaaldatum" in salarissen.columns:
        sal = salarissen.copy()
        sal["betaaldatum"] = pd.to_datetime(sal["betaaldatum"]).dt.date

        for idx, row in forecast.iterrows():
            week_start = row["week_start"]
            week_end = row["week_eind"]
            mask = (sal["betaaldatum"] >= week_start) & (sal["betaaldatum"] < week_end)
            forecast.loc[idx, "uitgaven_salarissen"] = sal.loc[mask, "bedrag"].sum()

    # Calculate net cashflow per week
    forecast["netto_cashflow"] = (
        forecast["inkomsten_debiteuren"]
        - forecast["uitgaven_crediteuren"]
        - forecast["uitgaven_salarissen"]
        - forecast["uitgaven_overig"]
    )

    # Calculate cumulative balance starting from current bank balance
    start_balance = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    forecast["cumulatief_saldo"] = start_balance + forecast["netto_cashflow"].cumsum()

    return forecast


def calculate_aging_buckets(
    df: pd.DataFrame,
    date_column: str = "vervaldatum",
    amount_column: str = "openstaand",
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Calculate aging buckets (ouderdomsanalyse) for receivables or payables.

    Args:
        df: DataFrame with due dates and amounts
        date_column: Name of the due date column
        amount_column: Name of the amount column
        reference_date: Reference date for aging (default: today)

    Returns:
        DataFrame with aging bucket summary
    """
    if df.empty:
        return pd.DataFrame({
            "bucket": ["Niet vervallen", "1-30 dagen", "31-60 dagen", "61-90 dagen", "> 90 dagen"],
            "bedrag": [0.0] * 5,
            "percentage": [0.0] * 5,
        })

    if reference_date is None:
        reference_date = datetime.now().date()

    df = df.copy()
    # Convert to datetime first, then to date - keep NaT handling robust
    date_series = pd.to_datetime(df[date_column], errors='coerce')

    # Calculate days overdue - handle NaT values before .dt.date conversion
    def calc_days(dt_val):
        if pd.isna(dt_val):
            return 0
        try:
            return (reference_date - dt_val.date()).days
        except (AttributeError, TypeError):
            return 0

    df["dagen_vervallen"] = date_series.apply(calc_days)

    # Define buckets
    def assign_bucket(days):
        if days <= 0:
            return "Niet vervallen"
        elif days <= 30:
            return "1-30 dagen"
        elif days <= 60:
            return "31-60 dagen"
        elif days <= 90:
            return "61-90 dagen"
        else:
            return "> 90 dagen"

    df["bucket"] = df["dagen_vervallen"].apply(assign_bucket)

    # Aggregate by bucket
    bucket_order = ["Niet vervallen", "1-30 dagen", "31-60 dagen", "61-90 dagen", "> 90 dagen"]
    summary = df.groupby("bucket")[amount_column].sum().reset_index()
    summary.columns = ["bucket", "bedrag"]

    # Ensure all buckets are present
    all_buckets = pd.DataFrame({"bucket": bucket_order})
    summary = all_buckets.merge(summary, on="bucket", how="left").fillna(0)

    # Calculate percentages
    total = summary["bedrag"].sum()
    summary["percentage"] = (summary["bedrag"] / total * 100).round(1) if total > 0 else 0

    return summary


def predict_payment_date(
    historical_behavior: pd.DataFrame,
    invoice_due_date: datetime,
    customer_id: Optional[str] = None
) -> Tuple[datetime, float]:
    """
    Predict when a payment will actually be received based on historical behavior.

    Args:
        historical_behavior: Historical payment data
        invoice_due_date: The official due date
        customer_id: Optional customer ID for customer-specific prediction

    Returns:
        Tuple of (predicted_date, confidence_score)
    """
    if historical_behavior.empty:
        # Default: assume payment on due date
        return invoice_due_date, 0.5

    # Calculate average delay from historical data
    avg_delay = historical_behavior["gem_betaaltermijn_debiteuren"].mean()
    std_delay = historical_behavior["gem_betaaltermijn_debiteuren"].std()

    # Assume invoices are sent with 30-day terms
    standard_terms = 30
    extra_days = int(avg_delay - standard_terms)

    if isinstance(invoice_due_date, datetime):
        predicted_date = invoice_due_date + timedelta(days=max(0, extra_days))
    else:
        predicted_date = datetime.combine(invoice_due_date, datetime.min.time()) + timedelta(days=max(0, extra_days))

    # Confidence based on consistency (lower std = higher confidence)
    confidence = max(0.3, min(0.95, 1 - (std_delay / 30))) if std_delay else 0.7

    return predicted_date, round(confidence, 2)


def calculate_seasonality_factors(
    historische_cashflow: pd.DataFrame,
) -> dict:
    """
    Bereken seizoensgebonden factoren per maand op basis van historische cashflow.

    Args:
        historische_cashflow: DataFrame met week_start, maand, inkomsten, uitgaven, netto

    Returns:
        Dict met per maand (1-12) de gemiddelde inkomsten, uitgaven en netto
    """
    if historische_cashflow.empty or "maand" not in historische_cashflow.columns:
        return {}

    # Groepeer per maand en bereken gemiddelden
    monthly = historische_cashflow.groupby("maand").agg({
        "inkomsten": "mean",
        "uitgaven": "mean",
        "netto": "mean"
    }).to_dict(orient="index")

    return monthly


def calculate_recurring_costs_per_week(
    terugkerende_kosten: pd.DataFrame,
) -> dict:
    """
    Bereken gemiddelde terugkerende kosten per week, gegroepeerd per kostensoort.

    Args:
        terugkerende_kosten: DataFrame met maand, kostensoort, bedrag

    Returns:
        Dict met per kostensoort het gemiddelde bedrag per week
    """
    if terugkerende_kosten.empty or "kostensoort" not in terugkerende_kosten.columns:
        return {"totaal_per_week": 0.0}

    # Tel totaal per kostensoort over alle maanden
    by_soort = terugkerende_kosten.groupby("kostensoort")["bedrag"].sum()

    # Bepaal aantal maanden in de dataset
    n_maanden = terugkerende_kosten["maand"].nunique() if "maand" in terugkerende_kosten.columns else 12

    result = {}
    totaal = 0.0
    for soort, totaal_bedrag in by_soort.items():
        gem_per_maand = totaal_bedrag / n_maanden if n_maanden > 0 else 0
        gem_per_week = gem_per_maand / 4.33  # ~4.33 weken per maand
        result[soort] = round(gem_per_week, 2)
        totaal += gem_per_week

    result["totaal_per_week"] = round(totaal, 2)
    return result


def create_enhanced_cashflow_forecast(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    terugkerende_kosten: pd.DataFrame,
    historische_cashflow: pd.DataFrame,
    weeks: int = 13,
    weeks_history: int = 4,
    debiteur_delay_days: int = 0,
    crediteur_delay_days: int = 0,
    reference_date = None,
    include_recurring_costs: bool = True,
    use_seasonality: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Verbeterde cashflow prognose gebaseerd op historisch profiel + bekende posten.
    Inclusief realisatie data (historische weken voor de standdatum).

    Methodiek:
    1. Week 1-4: Primair gebaseerd op openstaande posten (hoge zekerheid)
    2. Week 5+: Geleidelijke overgang naar historisch profiel (afnemende zekerheid)
    3. Confidence indicator per week voor transparantie

    Returns:
        Tuple van (DataFrame, forecast_start_idx) - net als create_transparent_cashflow_forecast
    """
    if reference_date is None:
        reference_date = datetime.now().date()
    elif isinstance(reference_date, datetime):
        reference_date = reference_date.date()

    # Bereken seizoensfactoren (historisch gemiddelde per maand)
    seasonality = calculate_seasonality_factors(historische_cashflow)

    # Bereken overall gemiddelden als fallback
    if not historische_cashflow.empty:
        overall_gem_in = historische_cashflow["inkomsten"].mean()
        overall_gem_uit = historische_cashflow["uitgaven"].mean()
        overall_gem_netto = historische_cashflow["netto"].mean() if "netto" in historische_cashflow.columns else overall_gem_in - overall_gem_uit
    else:
        overall_gem_in = 0
        overall_gem_uit = 0
        overall_gem_netto = 0

    all_rows = []

    # =========================================================================
    # DEEL 1: REALISATIE - Historische weken uit de echte cashflow data
    # =========================================================================
    if weeks_history > 0 and not historische_cashflow.empty:
        hist = historische_cashflow.copy()
        hist["week_start"] = pd.to_datetime(hist["week_start"]).dt.date

        # Filter op weken voor de reference_date
        hist = hist[hist["week_start"] < reference_date]
        hist = hist.sort_values("week_start", ascending=False).head(weeks_history)
        hist = hist.sort_values("week_start", ascending=True)  # Terug naar chronologisch

        for i, row in enumerate(hist.itertuples()):
            week_num = -(weeks_history - i)
            inkomsten = getattr(row, "inkomsten", 0)
            uitgaven = getattr(row, "uitgaven", 0)
            netto = inkomsten - uitgaven

            all_rows.append({
                "week_nummer": week_num,
                "week_label": f"Week {week_num}",
                "week_start": row.week_start,
                "week_eind": row.week_start + timedelta(days=7),
                "maand": row.week_start.month if hasattr(row.week_start, 'month') else 0,
                "inkomsten_debiteuren": round(inkomsten, 2),
                "uitgaven_crediteuren": round(uitgaven, 2),
                "netto_cashflow": round(netto, 2),
                "data_type": "Realisatie",
                "is_realisatie": True,
                "confidence": 1.0,  # 100% zekerheid voor realisatie
                "bron_inkomsten": "realisatie",
                "bron_uitgaven": "realisatie",
            })

    forecast_start_idx = len(all_rows)

    # =========================================================================
    # DEEL 2: PROGNOSE - Toekomstige weken met slimme combinatie
    # =========================================================================
    week_starts = [reference_date + timedelta(weeks=i) for i in range(weeks + 1)]

    # Bereken totaal bekende openstaande posten voor confidence berekening
    totaal_bekende_deb = debiteuren["openstaand"].sum() if not debiteuren.empty else 0
    totaal_bekende_cred = crediteuren["openstaand"].sum() if not crediteuren.empty else 0

    for i in range(weeks):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]
        week_maand = week_start.month

        # === STAP 1: Haal historisch profiel voor deze maand ===
        if use_seasonality and week_maand in seasonality:
            hist_in = seasonality[week_maand].get("inkomsten", overall_gem_in)
            hist_uit = seasonality[week_maand].get("uitgaven", overall_gem_uit)
        else:
            hist_in = overall_gem_in
            hist_uit = overall_gem_uit

        # === STAP 2: Bekende inkomsten uit openstaande debiteuren ===
        bekende_in = 0.0
        if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
            deb = debiteuren.copy()
            deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
            deb["verwachte_betaling"] = deb["vervaldatum"].apply(
                lambda x: x + timedelta(days=debiteur_delay_days) if pd.notna(x) else None
            )
            mask = (deb["verwachte_betaling"] >= week_start) & (deb["verwachte_betaling"] < week_end)
            bekende_in = deb.loc[mask, "openstaand"].sum()

        # === STAP 3: Bekende uitgaven uit openstaande crediteuren ===
        bekende_uit = 0.0
        if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
            cred = crediteuren.copy()
            cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
            cred["verwachte_betaling"] = cred["vervaldatum"].apply(
                lambda x: x + timedelta(days=crediteur_delay_days) if pd.notna(x) else None
            )
            mask = (cred["verwachte_betaling"] >= week_start) & (cred["verwachte_betaling"] < week_end)
            bekende_uit = cred.loc[mask, "openstaand"].sum()

        # === STAP 4: Bereken blend factor (0-1) ===
        # Week 1-4: veel gewicht op bekende posten
        # Week 5+: geleidelijk meer gewicht op historisch profiel
        if i < 4:
            # Eerste 4 weken: 90% bekende posten, 10% historisch (als aanvulling)
            blend_factor = 0.9
        else:
            # Week 5+: lineair aflopend naar 30% bekende posten
            # Week 5: 70%, Week 6: 60%, ... Week 13: 30%
            blend_factor = max(0.3, 0.9 - (i - 3) * 0.1)

        # === STAP 5: Combineer bronnen met blend factor ===
        # Als er bekende posten zijn, gebruik die met blend_factor gewicht
        # Vul aan met historisch profiel voor het resterende deel

        if bekende_in > 0:
            # Er zijn bekende inkomsten - combineer met historisch
            forecast_in = (blend_factor * bekende_in) + ((1 - blend_factor) * hist_in)
            bron_in = f"openstaand ({blend_factor*100:.0f}%) + historisch ({(1-blend_factor)*100:.0f}%)"
        elif hist_in > 0:
            # Geen bekende inkomsten - gebruik volledig historisch profiel
            forecast_in = hist_in
            bron_in = "historisch profiel"
        else:
            # Geen data beschikbaar
            forecast_in = 0
            bron_in = "geen data"

        if bekende_uit > 0:
            forecast_uit = (blend_factor * bekende_uit) + ((1 - blend_factor) * hist_uit)
            bron_uit = f"openstaand ({blend_factor*100:.0f}%) + historisch ({(1-blend_factor)*100:.0f}%)"
        elif hist_uit > 0:
            forecast_uit = hist_uit
            bron_uit = "historisch profiel"
        else:
            forecast_uit = 0
            bron_uit = "geen data"

        # === STAP 6: Bereken confidence score ===
        # Gebaseerd op: (1) hoeveel is bekend vs geschat, (2) hoe ver in de toekomst
        week_confidence = blend_factor  # Basis: blend factor
        if bekende_in == 0 and bekende_uit == 0:
            # Volledig gebaseerd op historie - lagere confidence
            week_confidence *= 0.6
        elif bekende_in == 0 or bekende_uit == 0:
            # Deels gebaseerd op historie
            week_confidence *= 0.8

        # Netto cashflow
        netto = forecast_in - forecast_uit

        all_rows.append({
            "week_nummer": i + 1,
            "week_label": f"Week {i + 1}",
            "week_start": week_start,
            "week_eind": week_end,
            "maand": week_maand,
            "inkomsten_debiteuren": round(forecast_in, 2),
            "uitgaven_crediteuren": round(forecast_uit, 2),
            "inkomsten_bekend": round(bekende_in, 2),
            "uitgaven_bekend": round(bekende_uit, 2),
            "inkomsten_historisch": round(hist_in, 2),
            "uitgaven_historisch": round(hist_uit, 2),
            "netto_cashflow": round(netto, 2),
            "data_type": "Prognose",
            "is_realisatie": False,
            "confidence": round(week_confidence, 2),
            "bron_inkomsten": bron_in,
            "bron_uitgaven": bron_uit,
        })

    df = pd.DataFrame(all_rows)

    # Bereken cumulatief saldo
    start_balance = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    df["cumulatief_saldo"] = start_balance + df["netto_cashflow"].cumsum()

    return df, forecast_start_idx


# =============================================================================
# ML-GEBASEERDE FORECAST MET BACKTESTING
# =============================================================================

@dataclass
class ForecastModelMetrics:
    """Container voor model performance metrics."""
    mape_inkomsten: float  # Mean Absolute Percentage Error voor inkomsten
    mape_uitgaven: float   # Mean Absolute Percentage Error voor uitgaven
    mape_netto: float      # Mean Absolute Percentage Error voor netto cashflow
    rmse_netto: float      # Root Mean Square Error voor netto cashflow
    bias: float            # Systematische over/onderschatting
    n_test_weeks: int      # Aantal weken in test set
    model_type: str        # Type model gebruikt


def learn_weekly_pattern(
    historische_cashflow: pd.DataFrame,
    min_weeks: int = 8
) -> dict:
    """
    Leer het wekelijkse cashflow patroon uit historische data.

    Berekent:
    - Gemiddelde per maand (seizoenspatroon)
    - Trend (stijgend/dalend)
    - Volatiliteit (standaarddeviatie)

    Args:
        historische_cashflow: DataFrame met week_start, maand, inkomsten, uitgaven, netto
        min_weeks: Minimum aantal weken data nodig

    Returns:
        Dict met geleerde parameters
    """
    if historische_cashflow.empty or len(historische_cashflow) < min_weeks:
        return {
            "has_pattern": False,
            "reason": f"Te weinig data ({len(historische_cashflow)} weken, minimum {min_weeks})"
        }

    df = historische_cashflow.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    # === 1. SEIZOENSPATROON PER MAAND ===
    monthly_pattern = df.groupby("maand").agg({
        "inkomsten": ["mean", "std", "count"],
        "uitgaven": ["mean", "std", "count"],
        "netto": ["mean", "std"]
    })

    # Flatten column names
    monthly_pattern.columns = [
        "inkomsten_mean", "inkomsten_std", "inkomsten_count",
        "uitgaven_mean", "uitgaven_std", "uitgaven_count",
        "netto_mean", "netto_std"
    ]

    # === 2. TREND DETECTIE (linear regression) ===
    df["week_idx"] = range(len(df))

    # Inkomsten trend
    if df["inkomsten"].std() > 0:
        x = df["week_idx"].values
        y = df["inkomsten"].values
        slope_in, intercept_in = np.polyfit(x, y, 1)
    else:
        slope_in, intercept_in = 0, df["inkomsten"].mean()

    # Uitgaven trend
    if df["uitgaven"].std() > 0:
        x = df["week_idx"].values
        y = df["uitgaven"].values
        slope_uit, intercept_uit = np.polyfit(x, y, 1)
    else:
        slope_uit, intercept_uit = 0, df["uitgaven"].mean()

    # === 3. VOLATILITEIT ===
    volatility_in = df["inkomsten"].std() / df["inkomsten"].mean() if df["inkomsten"].mean() > 0 else 0
    volatility_uit = df["uitgaven"].std() / df["uitgaven"].mean() if df["uitgaven"].mean() > 0 else 0

    # === 4. WEIGHTED MOVING AVERAGE (recente weken belangrijker) ===
    # Exponentiële weights: recentste week heeft hoogste weight
    n = len(df)
    alpha = 0.3  # Smoothing factor
    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()

    wma_inkomsten = np.average(df["inkomsten"].values, weights=weights)
    wma_uitgaven = np.average(df["uitgaven"].values, weights=weights)

    return {
        "has_pattern": True,
        "monthly_pattern": monthly_pattern.to_dict(orient="index"),
        "trend": {
            "inkomsten_slope": slope_in,
            "inkomsten_intercept": intercept_in,
            "uitgaven_slope": slope_uit,
            "uitgaven_intercept": intercept_uit,
        },
        "volatility": {
            "inkomsten": volatility_in,
            "uitgaven": volatility_uit,
        },
        "weighted_avg": {
            "inkomsten": wma_inkomsten,
            "uitgaven": wma_uitgaven,
        },
        "overall_avg": {
            "inkomsten": df["inkomsten"].mean(),
            "uitgaven": df["uitgaven"].mean(),
            "netto": df["netto"].mean() if "netto" in df.columns else df["inkomsten"].mean() - df["uitgaven"].mean(),
        },
        "n_weeks": len(df),
        "last_week_idx": n - 1,
    }


def predict_week_ml(
    week_idx: int,
    week_maand: int,
    pattern: dict,
    bekende_in: float = 0,
    bekende_uit: float = 0,
    weeks_ahead: int = 1,
) -> Tuple[float, float, float, str]:
    """
    Voorspel cashflow voor een specifieke week met ML-gebaseerd model.

    Combineert:
    1. Bekende openstaande posten (als beschikbaar)
    2. Seizoenspatroon voor de maand
    3. Trend extrapolatie
    4. Weighted moving average

    Args:
        week_idx: Index van de week (vanaf laatste bekende week)
        week_maand: Maand nummer (1-12)
        pattern: Geleerde patronen uit learn_weekly_pattern()
        bekende_in: Bekende inkomsten uit openstaande debiteuren
        bekende_uit: Bekende uitgaven uit openstaande crediteuren
        weeks_ahead: Hoeveel weken vooruit we voorspellen

    Returns:
        Tuple van (voorspelde_inkomsten, voorspelde_uitgaven, confidence, methode_beschrijving)
    """
    if not pattern.get("has_pattern", False):
        # Geen patroon geleerd - gebruik alleen bekende posten
        return bekende_in, bekende_uit, 0.3, "alleen bekende posten (geen historie)"

    # === Component 1: Seizoenspatroon ===
    monthly = pattern["monthly_pattern"]
    if week_maand in monthly:
        seasonal_in = monthly[week_maand].get("inkomsten_mean", 0)
        seasonal_uit = monthly[week_maand].get("uitgaven_mean", 0)
        seasonal_count = monthly[week_maand].get("inkomsten_count", 0)
    else:
        seasonal_in = pattern["overall_avg"]["inkomsten"]
        seasonal_uit = pattern["overall_avg"]["uitgaven"]
        seasonal_count = 0

    # === Component 2: Trend extrapolatie ===
    future_idx = pattern["last_week_idx"] + week_idx + 1
    trend_in = pattern["trend"]["inkomsten_slope"] * future_idx + pattern["trend"]["inkomsten_intercept"]
    trend_uit = pattern["trend"]["uitgaven_slope"] * future_idx + pattern["trend"]["uitgaven_intercept"]

    # Begrens trend om extreme extrapolaties te voorkomen
    avg_in = pattern["overall_avg"]["inkomsten"]
    avg_uit = pattern["overall_avg"]["uitgaven"]
    trend_in = np.clip(trend_in, avg_in * 0.5, avg_in * 1.5)
    trend_uit = np.clip(trend_uit, avg_uit * 0.5, avg_uit * 1.5)

    # === Component 3: Weighted Moving Average ===
    wma_in = pattern["weighted_avg"]["inkomsten"]
    wma_uit = pattern["weighted_avg"]["uitgaven"]

    # === Combineer componenten met gewichten ===
    # Gewichten afhankelijk van beschikbare data en voorspelhorizon

    if weeks_ahead <= 4:
        # Korte termijn: meer gewicht op bekende posten en WMA
        w_bekend = 0.5 if (bekende_in > 0 or bekende_uit > 0) else 0.0
        w_seasonal = 0.2
        w_trend = 0.1
        w_wma = 0.3 if w_bekend == 0 else 0.2
    else:
        # Lange termijn: meer gewicht op seizoen en trend
        w_bekend = 0.2 if (bekende_in > 0 or bekende_uit > 0) else 0.0
        w_seasonal = 0.4
        w_trend = 0.2
        w_wma = 0.4 if w_bekend == 0 else 0.2

    # Normaliseer gewichten
    total_w = w_bekend + w_seasonal + w_trend + w_wma
    w_bekend /= total_w
    w_seasonal /= total_w
    w_trend /= total_w
    w_wma /= total_w

    # Bereken gewogen voorspelling
    pred_in = (
        w_bekend * bekende_in +
        w_seasonal * seasonal_in +
        w_trend * trend_in +
        w_wma * wma_in
    )

    pred_uit = (
        w_bekend * bekende_uit +
        w_seasonal * seasonal_uit +
        w_trend * trend_uit +
        w_wma * wma_uit
    )

    # === Bereken confidence ===
    # Basis confidence aflopend met voorspelhorizon
    base_confidence = max(0.3, 1.0 - (weeks_ahead * 0.05))

    # Verhoog confidence als we veel seizoensdata hebben
    if seasonal_count >= 4:
        base_confidence *= 1.1

    # Verlaag confidence bij hoge volatiliteit
    vol = max(pattern["volatility"]["inkomsten"], pattern["volatility"]["uitgaven"])
    if vol > 0.5:
        base_confidence *= 0.8

    # Verhoog confidence als bekende posten overeenkomen met patroon
    if bekende_in > 0 and abs(bekende_in - seasonal_in) / max(seasonal_in, 1) < 0.3:
        base_confidence *= 1.05

    confidence = min(0.95, max(0.2, base_confidence))

    # Beschrijving
    methode = f"ML ensemble: bekend({w_bekend*100:.0f}%) + seizoen({w_seasonal*100:.0f}%) + trend({w_trend*100:.0f}%) + WMA({w_wma*100:.0f}%)"

    return pred_in, pred_uit, confidence, methode


def backtest_forecast_model(
    historische_cashflow: pd.DataFrame,
    test_weeks: int = 8,
    forecast_horizon: int = 4,
) -> Tuple[ForecastModelMetrics, pd.DataFrame]:
    """
    Voer walk-forward backtesting uit om model nauwkeurigheid te meten.

    Methodiek:
    1. Train op eerste N-test_weeks weken
    2. Voorspel de volgende forecast_horizon weken
    3. Vergelijk met werkelijke waarden
    4. Schuif 1 week op en herhaal

    Args:
        historische_cashflow: Volledige historische data
        test_weeks: Aantal weken om te testen
        forecast_horizon: Hoeveel weken vooruit voorspellen per test

    Returns:
        Tuple van (ForecastModelMetrics, DataFrame met alle voorspellingen vs actuals)
    """
    if historische_cashflow.empty or len(historische_cashflow) < test_weeks + 8:
        return ForecastModelMetrics(
            mape_inkomsten=999,
            mape_uitgaven=999,
            mape_netto=999,
            rmse_netto=999,
            bias=0,
            n_test_weeks=0,
            model_type="insufficient_data"
        ), pd.DataFrame()

    df = historische_cashflow.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    results = []

    # Walk-forward validation
    for test_start_idx in range(len(df) - test_weeks, len(df) - forecast_horizon + 1):
        # Training data: alles tot test_start_idx
        train_data = df.iloc[:test_start_idx].copy()

        # Leer patroon van training data
        pattern = learn_weekly_pattern(train_data)

        if not pattern.get("has_pattern", False):
            continue

        # Voorspel de volgende forecast_horizon weken
        for h in range(min(forecast_horizon, len(df) - test_start_idx)):
            actual_idx = test_start_idx + h
            actual_row = df.iloc[actual_idx]

            week_maand = int(actual_row["maand"])

            # Voorspel (zonder bekende posten - pure ML test)
            pred_in, pred_uit, conf, methode = predict_week_ml(
                week_idx=h,
                week_maand=week_maand,
                pattern=pattern,
                bekende_in=0,  # Geen bekende posten voor pure ML test
                bekende_uit=0,
                weeks_ahead=h + 1
            )

            actual_in = actual_row["inkomsten"]
            actual_uit = actual_row["uitgaven"]
            actual_netto = actual_in - actual_uit
            pred_netto = pred_in - pred_uit

            results.append({
                "test_date": actual_row["week_start"],
                "horizon": h + 1,
                "pred_inkomsten": pred_in,
                "actual_inkomsten": actual_in,
                "pred_uitgaven": pred_uit,
                "actual_uitgaven": actual_uit,
                "pred_netto": pred_netto,
                "actual_netto": actual_netto,
                "confidence": conf,
                "error_in": pred_in - actual_in,
                "error_uit": pred_uit - actual_uit,
                "error_netto": pred_netto - actual_netto,
            })

    if not results:
        return ForecastModelMetrics(
            mape_inkomsten=999,
            mape_uitgaven=999,
            mape_netto=999,
            rmse_netto=999,
            bias=0,
            n_test_weeks=0,
            model_type="no_results"
        ), pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Bereken metrics
    def safe_mape(pred, actual):
        """MAPE met bescherming tegen deling door 0."""
        mask = actual != 0
        if mask.sum() == 0:
            return 0
        return np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100

    mape_in = safe_mape(results_df["pred_inkomsten"].values, results_df["actual_inkomsten"].values)
    mape_uit = safe_mape(results_df["pred_uitgaven"].values, results_df["actual_uitgaven"].values)
    mape_netto = safe_mape(results_df["pred_netto"].values, results_df["actual_netto"].values)

    rmse_netto = np.sqrt(np.mean(results_df["error_netto"] ** 2))
    bias = results_df["error_netto"].mean()  # Positief = overschatting, negatief = onderschatting

    metrics = ForecastModelMetrics(
        mape_inkomsten=round(mape_in, 1),
        mape_uitgaven=round(mape_uit, 1),
        mape_netto=round(mape_netto, 1),
        rmse_netto=round(rmse_netto, 2),
        bias=round(bias, 2),
        n_test_weeks=len(results_df),
        model_type="ml_ensemble"
    )

    return metrics, results_df


def create_ml_forecast(
    banksaldo: pd.DataFrame,
    debiteuren: pd.DataFrame,
    crediteuren: pd.DataFrame,
    historische_cashflow: pd.DataFrame,
    weeks: int = 13,
    weeks_history: int = 4,
    debiteur_delay_days: int = 0,
    crediteur_delay_days: int = 0,
    reference_date=None,
) -> Tuple[pd.DataFrame, int, ForecastModelMetrics]:
    """
    ML-gebaseerde cashflow forecast met automatische backtesting.

    Combineert:
    1. Geleerde patronen uit historische data (seizoen, trend, WMA)
    2. Bekende openstaande posten met vervaldatums
    3. Automatische weging gebaseerd op voorspelhorizon

    Returns:
        Tuple van (forecast DataFrame, forecast_start_idx, model metrics)
    """
    if reference_date is None:
        reference_date = datetime.now().date()
    elif isinstance(reference_date, datetime):
        reference_date = reference_date.date()

    # === STAP 1: Leer patroon uit historische data ===
    pattern = learn_weekly_pattern(historische_cashflow)

    # === STAP 2: Backtest het model ===
    metrics, backtest_results = backtest_forecast_model(
        historische_cashflow,
        test_weeks=8,
        forecast_horizon=4
    )

    all_rows = []

    # === STAP 3: REALISATIE - Historische weken ===
    if weeks_history > 0 and not historische_cashflow.empty:
        hist = historische_cashflow.copy()
        hist["week_start"] = pd.to_datetime(hist["week_start"]).dt.date
        hist = hist[hist["week_start"] < reference_date]
        hist = hist.sort_values("week_start", ascending=False).head(weeks_history)
        hist = hist.sort_values("week_start", ascending=True)

        for i, row in enumerate(hist.itertuples()):
            week_num = -(weeks_history - i)
            inkomsten = getattr(row, "inkomsten", 0)
            uitgaven = getattr(row, "uitgaven", 0)
            netto = inkomsten - uitgaven

            all_rows.append({
                "week_nummer": week_num,
                "week_label": f"Week {week_num}",
                "week_start": row.week_start,
                "week_eind": row.week_start + timedelta(days=7),
                "maand": row.week_start.month if hasattr(row.week_start, 'month') else 0,
                "inkomsten_debiteuren": round(inkomsten, 2),
                "uitgaven_crediteuren": round(uitgaven, 2),
                "netto_cashflow": round(netto, 2),
                "data_type": "Realisatie",
                "is_realisatie": True,
                "confidence": 1.0,
                "methode": "realisatie",
            })

    forecast_start_idx = len(all_rows)

    # === STAP 4: PROGNOSE - ML-gebaseerd ===
    week_starts = [reference_date + timedelta(weeks=i) for i in range(weeks + 1)]

    for i in range(weeks):
        week_start = week_starts[i]
        week_end = week_starts[i + 1]
        week_maand = week_start.month

        # Bekende inkomsten uit openstaande debiteuren
        bekende_in = 0.0
        if not debiteuren.empty and "vervaldatum" in debiteuren.columns:
            deb = debiteuren.copy()
            deb["vervaldatum"] = pd.to_datetime(deb["vervaldatum"]).dt.date
            deb["verwachte_betaling"] = deb["vervaldatum"].apply(
                lambda x: x + timedelta(days=debiteur_delay_days) if pd.notna(x) else None
            )
            mask = (deb["verwachte_betaling"] >= week_start) & (deb["verwachte_betaling"] < week_end)
            bekende_in = deb.loc[mask, "openstaand"].sum()

        # Bekende uitgaven uit openstaande crediteuren
        bekende_uit = 0.0
        if not crediteuren.empty and "vervaldatum" in crediteuren.columns:
            cred = crediteuren.copy()
            cred["vervaldatum"] = pd.to_datetime(cred["vervaldatum"]).dt.date
            cred["verwachte_betaling"] = cred["vervaldatum"].apply(
                lambda x: x + timedelta(days=crediteur_delay_days) if pd.notna(x) else None
            )
            mask = (cred["verwachte_betaling"] >= week_start) & (cred["verwachte_betaling"] < week_end)
            bekende_uit = cred.loc[mask, "openstaand"].sum()

        # ML voorspelling
        pred_in, pred_uit, confidence, methode = predict_week_ml(
            week_idx=i,
            week_maand=week_maand,
            pattern=pattern,
            bekende_in=bekende_in,
            bekende_uit=bekende_uit,
            weeks_ahead=i + 1
        )

        netto = pred_in - pred_uit

        all_rows.append({
            "week_nummer": i + 1,
            "week_label": f"Week {i + 1}",
            "week_start": week_start,
            "week_eind": week_end,
            "maand": week_maand,
            "inkomsten_debiteuren": round(pred_in, 2),
            "uitgaven_crediteuren": round(pred_uit, 2),
            "inkomsten_bekend": round(bekende_in, 2),
            "uitgaven_bekend": round(bekende_uit, 2),
            "netto_cashflow": round(netto, 2),
            "data_type": "Prognose",
            "is_realisatie": False,
            "confidence": round(confidence, 2),
            "methode": methode,
        })

    df = pd.DataFrame(all_rows)

    # Bereken cumulatief saldo
    start_balance = banksaldo["saldo"].sum() if not banksaldo.empty else 0
    df["cumulatief_saldo"] = start_balance + df["netto_cashflow"].cumsum()

    return df, forecast_start_idx, metrics
