"""
Business calculations for Notifica Consultancy Forecast
"""
from datetime import datetime, date
from typing import Optional
from .models import (
    DataStore, Partner, Project, HourEntry, Invoice,
    ProjectType, ProjectStatus, get_data_store
)


# Constants
WORKDAYS_PER_MONTH = 22  # Approximate working days per month
DAILY_RATE = 1150.0
TARGET_BILLABLE_PERCENTAGE = 0.5


def get_months_range(start_month: str, end_month: str) -> list[str]:
    """Generate list of months between start and end (inclusive)"""
    months = []
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def get_current_month() -> str:
    """Get current month in YYYY-MM format"""
    return date.today().strftime("%Y-%m")


def get_month_display(month: str) -> str:
    """Convert 2025-01 to 'Januari 2025'"""
    month_names = {
        "01": "Januari", "02": "Februari", "03": "Maart",
        "04": "April", "05": "Mei", "06": "Juni",
        "07": "Juli", "08": "Augustus", "09": "September",
        "10": "Oktober", "11": "November", "12": "December"
    }
    year, m = month.split("-")
    return f"{month_names[m]} {year}"


class MonthlyMetrics:
    """Metrics for a specific month"""

    def __init__(self, month: str, store: DataStore):
        self.month = month
        self.store = store
        self._calculate()

    def _calculate(self):
        projects = self.store.get_projects()
        hours = self.store.get_hours_for_month(self.month)
        invoices = self.store.get_invoices_for_month(self.month)
        partners = self.store.get_partners()

        # Target revenue (all 4 partners at 50%)
        self.target_revenue = sum(p.monthly_target_revenue for p in partners)

        # Planned/Committed revenue (from active projects)
        self.committed_revenue = 0
        self.pipeline_revenue = 0

        for project in projects:
            monthly_value = project.get_monthly_value(self.month)
            if project.status == ProjectStatus.ACTIVE:
                self.committed_revenue += monthly_value
            elif project.status == ProjectStatus.PIPELINE:
                self.pipeline_revenue += monthly_value

        self.total_planned = self.committed_revenue + self.pipeline_revenue

        # Realized revenue (invoiced)
        self.realized_revenue = sum(inv.amount for inv in invoices)

        # Hours
        self.total_hours = sum(h.hours for h in hours)
        self.billable_hours = sum(h.hours for h in hours if h.billable)
        self.non_billable_hours = self.total_hours - self.billable_hours

        # Gap analysis
        self.gap_to_target = self.target_revenue - self.total_planned
        self.gap_to_realized = self.committed_revenue - self.realized_revenue

    @property
    def on_track(self) -> bool:
        return self.total_planned >= self.target_revenue

    @property
    def percentage_of_target(self) -> float:
        if self.target_revenue == 0:
            return 0
        return (self.total_planned / self.target_revenue) * 100


class PartnerMetrics:
    """Metrics for a specific partner in a month"""

    def __init__(self, partner: Partner, month: str, store: DataStore):
        self.partner = partner
        self.month = month
        self.store = store
        self._calculate()

    def _calculate(self):
        projects = self.store.get_projects()
        hours = [h for h in self.store.get_hours_for_month(self.month)
                 if h.partner_id == self.partner.id]

        # Target
        self.target_revenue = self.partner.monthly_target_revenue
        self.target_days = self.partner.monthly_target_days

        # Planned revenue from projects assigned to this partner
        self.planned_revenue = 0
        for project in projects:
            if self.partner.id in project.assigned_partners:
                # Split value among assigned partners
                partner_share = 1 / len(project.assigned_partners)
                self.planned_revenue += project.get_monthly_value(self.month) * partner_share

        # Hours
        self.total_hours = sum(h.hours for h in hours)
        self.billable_hours = sum(h.hours for h in hours if h.billable)
        self.non_billable_hours = self.total_hours - self.billable_hours

        # Days (8 hours = 1 day)
        self.billable_days = self.billable_hours / 8
        self.non_billable_days = self.non_billable_hours / 8

    @property
    def billable_percentage(self) -> float:
        """Percentage of target days that are billable"""
        if self.target_days == 0:
            return 0
        return (self.billable_days / self.target_days) * 100

    @property
    def revenue_percentage(self) -> float:
        """Percentage of target revenue that is planned"""
        if self.target_revenue == 0:
            return 0
        return (self.planned_revenue / self.target_revenue) * 100


def get_forecast_summary(store: DataStore, months: list[str]) -> dict:
    """Get forecast summary across multiple months"""
    summary = {
        "months": [],
        "total_target": 0,
        "total_committed": 0,
        "total_pipeline": 0,
        "total_realized": 0,
    }

    for month in months:
        metrics = MonthlyMetrics(month, store)
        summary["months"].append({
            "month": month,
            "display": get_month_display(month),
            "target": metrics.target_revenue,
            "committed": metrics.committed_revenue,
            "pipeline": metrics.pipeline_revenue,
            "realized": metrics.realized_revenue,
            "on_track": metrics.on_track,
        })
        summary["total_target"] += metrics.target_revenue
        summary["total_committed"] += metrics.committed_revenue
        summary["total_pipeline"] += metrics.pipeline_revenue
        summary["total_realized"] += metrics.realized_revenue

    summary["gap"] = summary["total_target"] - (summary["total_committed"] + summary["total_pipeline"])

    return summary


def get_partner_role_description(partner_id: str) -> str:
    """Get role description for partner"""
    roles = {
        "tobias": "Directie & Commercie",
        "arthur": "Klantrelaties & Regie",
        "dolf": "Backend & Realisatie",
        "mark": "Backend & Realisatie",
    }
    return roles.get(partner_id, "Partner")


def get_projects_by_partner(store: DataStore, partner_id: str) -> list[Project]:
    """Get all projects assigned to a partner"""
    return [p for p in store.get_projects() if partner_id in p.assigned_partners]


def get_projects_for_month(store: DataStore, month: str) -> list[Project]:
    """Get all projects active in a specific month"""
    projects = []
    for project in store.get_projects():
        if project.start_month <= month <= project.end_month:
            projects.append(project)
    return projects
