"""
Data models for Notifica Consultancy Forecast App
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import streamlit as st


class ProjectType(str, Enum):
    CLIENT_PROJECT = "Klantproject"
    PRODUCT_TARGET = "Product-target"
    REGIE_BUDGET = "Regie-budget"


class ProjectStatus(str, Enum):
    PIPELINE = "Pipeline"
    ACTIVE = "Actief"
    COMPLETED = "Afgerond"


@dataclass
class Partner:
    id: str
    name: str
    target_percentage: float = 0.5  # 50% facturabel
    daily_rate: float = 1150.0

    @property
    def monthly_target_days(self) -> float:
        """~11 commercial days per month at 50%"""
        return 22 * self.target_percentage  # 22 workdays/month * 50%

    @property
    def monthly_target_revenue(self) -> float:
        return self.monthly_target_days * self.daily_rate


@dataclass
class Client:
    id: str
    name: str
    active: bool = True


@dataclass
class Project:
    id: str
    name: str
    project_type: ProjectType
    client_id: Optional[str]  # None for product-targets without specific client
    total_value: float
    start_month: str  # Format: "2025-01"
    end_month: str    # Format: "2025-03"
    assigned_partners: list[str] = field(default_factory=list)  # Partner IDs
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str = ""
    monthly_distribution: dict[str, float] = field(default_factory=dict)  # {"2025-01": 5000, ...}

    def get_monthly_value(self, month: str) -> float:
        """Get the planned value for a specific month"""
        if self.monthly_distribution:
            return self.monthly_distribution.get(month, 0)
        # Default: distribute evenly
        months = self._get_months_in_range()
        if not months:
            return 0
        return self.total_value / len(months)

    def _get_months_in_range(self) -> list[str]:
        """Get all months between start and end"""
        months = []
        start = datetime.strptime(self.start_month, "%Y-%m")
        end = datetime.strptime(self.end_month, "%Y-%m")
        current = start
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months


@dataclass
class HourEntry:
    id: str
    date: str  # Format: "2025-01-21"
    partner_id: str
    project_id: str
    hours: float
    billable: bool = True
    description: str = ""

    @property
    def month(self) -> str:
        return self.date[:7]  # "2025-01"


@dataclass
class Invoice:
    id: str
    project_id: str
    month: str  # Format: "2025-01"
    amount: float
    invoiced_date: Optional[str] = None
    description: str = ""


class DataStore:
    """Simple JSON-based data store"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.partners_file = self.data_dir / "partners.json"
        self.clients_file = self.data_dir / "clients.json"
        self.projects_file = self.data_dir / "projects.json"
        self.hours_file = self.data_dir / "hours.json"
        self.invoices_file = self.data_dir / "invoices.json"

    def _load_json(self, filepath: Path) -> list[dict]:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_json(self, filepath: Path, data: list[dict]):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Partners
    def get_partners(self) -> list[Partner]:
        data = self._load_json(self.partners_file)
        return [Partner(**p) for p in data]

    def save_partners(self, partners: list[Partner]):
        self._save_json(self.partners_file, [asdict(p) for p in partners])

    def get_partner_by_id(self, partner_id: str) -> Optional[Partner]:
        for p in self.get_partners():
            if p.id == partner_id:
                return p
        return None

    # Clients
    def get_clients(self) -> list[Client]:
        data = self._load_json(self.clients_file)
        return [Client(**c) for c in data]

    def save_clients(self, clients: list[Client]):
        self._save_json(self.clients_file, [asdict(c) for c in clients])

    def get_client_by_id(self, client_id: str) -> Optional[Client]:
        for c in self.get_clients():
            if c.id == client_id:
                return c
        return None

    # Projects
    def get_projects(self) -> list[Project]:
        data = self._load_json(self.projects_file)
        projects = []
        for p in data:
            p['project_type'] = ProjectType(p['project_type'])
            p['status'] = ProjectStatus(p['status'])
            projects.append(Project(**p))
        return projects

    def save_projects(self, projects: list[Project]):
        data = []
        for p in projects:
            d = asdict(p)
            d['project_type'] = p.project_type.value
            d['status'] = p.status.value
            data.append(d)
        self._save_json(self.projects_file, data)

    def add_project(self, project: Project):
        projects = self.get_projects()
        projects.append(project)
        self.save_projects(projects)

    def update_project(self, project: Project):
        projects = self.get_projects()
        for i, p in enumerate(projects):
            if p.id == project.id:
                projects[i] = project
                break
        self.save_projects(projects)

    def delete_project(self, project_id: str):
        projects = [p for p in self.get_projects() if p.id != project_id]
        self.save_projects(projects)

    # Hours
    def get_hours(self) -> list[HourEntry]:
        data = self._load_json(self.hours_file)
        return [HourEntry(**h) for h in data]

    def save_hours(self, hours: list[HourEntry]):
        self._save_json(self.hours_file, [asdict(h) for h in hours])

    def add_hour_entry(self, entry: HourEntry):
        hours = self.get_hours()
        hours.append(entry)
        self.save_hours(hours)

    def delete_hour_entry(self, entry_id: str):
        hours = [h for h in self.get_hours() if h.id != entry_id]
        self.save_hours(hours)

    def get_hours_for_month(self, month: str) -> list[HourEntry]:
        return [h for h in self.get_hours() if h.month == month]

    def get_hours_for_partner(self, partner_id: str) -> list[HourEntry]:
        return [h for h in self.get_hours() if h.partner_id == partner_id]

    # Invoices
    def get_invoices(self) -> list[Invoice]:
        data = self._load_json(self.invoices_file)
        return [Invoice(**i) for i in data]

    def save_invoices(self, invoices: list[Invoice]):
        self._save_json(self.invoices_file, [asdict(i) for i in invoices])

    def add_invoice(self, invoice: Invoice):
        invoices = self.get_invoices()
        invoices.append(invoice)
        self.save_invoices(invoices)

    def get_invoices_for_month(self, month: str) -> list[Invoice]:
        return [i for i in self.get_invoices() if i.month == month]


def get_data_store() -> DataStore:
    """Get or create the data store singleton"""
    data_dir = Path(__file__).parent.parent / "data"
    return DataStore(data_dir)


def initialize_default_data(store: DataStore):
    """Initialize with Notifica partners and sample data"""

    # Check if already initialized
    if store.get_partners():
        return

    # Partners
    partners = [
        Partner(id="tobias", name="Tobias", target_percentage=0.5, daily_rate=1150.0),
        Partner(id="arthur", name="Arthur", target_percentage=0.5, daily_rate=1150.0),
        Partner(id="dolf", name="Dolf", target_percentage=0.5, daily_rate=1150.0),
        Partner(id="mark", name="Mark", target_percentage=0.5, daily_rate=1150.0),
    ]
    store.save_partners(partners)

    # Clients
    clients = [
        Client(id="unica", name="Unica", active=True),
        Client(id="barth", name="Barth Groep", active=True),
        Client(id="castellum", name="Castellum", active=True),
        Client(id="wvc", name="WVC", active=True),
        Client(id="diverse_arthur", name="Diverse (Arthur regie)", active=True),
        Client(id="target_liqui", name="Target: Liquiditeitsprognose", active=True),
        Client(id="target_voorraad", name="Target: Voorraad Analyse", active=True),
        Client(id="target_ai", name="Target: AI Contract Checker", active=True),
        Client(id="target_blop", name="Target: BLOP Analyse", active=True),
    ]
    store.save_clients(clients)

    # Projects
    projects = [
        # Unica - 30 dagen regie (Mark & Dolf) - Jan t/m Mrt
        Project(
            id="unica-regie-2025",
            name="Unica Regie",
            project_type=ProjectType.REGIE_BUDGET,
            client_id="unica",
            total_value=34500.0,  # 30 dagen × €1150
            start_month="2025-01",
            end_month="2025-03",
            assigned_partners=["mark", "dolf"],
            status=ProjectStatus.ACTIVE,
            description="30 dagen regie-uren",
            monthly_distribution={"2025-01": 11500, "2025-02": 11500, "2025-03": 11500}
        ),
        # Barth Groep migratie (Tobias & Dolf) - Mrt t/m Jun
        Project(
            id="barth-migratie-2025",
            name="Barth Groep Migratie",
            project_type=ProjectType.CLIENT_PROJECT,
            client_id="barth",
            total_value=15000.0,
            start_month="2025-03",
            end_month="2025-06",
            assigned_partners=["tobias", "dolf"],
            status=ProjectStatus.ACTIVE,
            description="Database migratie",
            monthly_distribution={"2025-03": 3750, "2025-04": 3750, "2025-05": 3750, "2025-06": 3750}
        ),
        # Castellum (Tobias & Mark) - Apr t/m Jun
        Project(
            id="castellum-2025",
            name="Castellum Oplevering",
            project_type=ProjectType.CLIENT_PROJECT,
            client_id="castellum",
            total_value=20000.0,
            start_month="2025-04",
            end_month="2025-06",
            assigned_partners=["tobias", "mark"],
            status=ProjectStatus.ACTIVE,
            description="Afronding, oplevering + aanvullende functionaliteiten (database stacking, PowerApps)",
            monthly_distribution={"2025-04": 6667, "2025-05": 6667, "2025-06": 6666}
        ),
        # WVC AI Contract Checker pilot (Tobias) - Feb
        Project(
            id="wvc-ai-2025",
            name="WVC AI Contract Checker (pilot)",
            project_type=ProjectType.CLIENT_PROJECT,
            client_id="wvc",
            total_value=5000.0,
            start_month="2025-02",
            end_month="2025-02",
            assigned_partners=["tobias"],
            status=ProjectStatus.ACTIVE,
            description="AI Contract Checker pilot",
            monthly_distribution={"2025-02": 5000}
        ),
        # Arthur doorlopend regie
        Project(
            id="arthur-regie-2025",
            name="Arthur Regie (diverse klanten)",
            project_type=ProjectType.REGIE_BUDGET,
            client_id="diverse_arthur",
            total_value=60000.0,  # €10k × 6 maanden
            start_month="2025-01",
            end_month="2025-06",
            assigned_partners=["arthur"],
            status=ProjectStatus.ACTIVE,
            description="Doorlopende regie-uren bij bestaande klanten",
            monthly_distribution={"2025-01": 10000, "2025-02": 10000, "2025-03": 10000,
                                "2025-04": 10000, "2025-05": 10000, "2025-06": 10000}
        ),
        # Product targets - Liquiditeitsprognose (3×)
        Project(
            id="target-liqui-2025",
            name="Liquiditeitsprognose (3× verkoop)",
            project_type=ProjectType.PRODUCT_TARGET,
            client_id="target_liqui",
            total_value=15000.0,  # 3 × €5000
            start_month="2025-02",
            end_month="2025-04",
            assigned_partners=["tobias"],
            status=ProjectStatus.PIPELINE,
            description="Target: 3 klanten × €5.000",
            monthly_distribution={"2025-02": 5000, "2025-03": 5000, "2025-04": 5000}
        ),
        # Product targets - Voorraad Analyse (3×)
        Project(
            id="target-voorraad-2025",
            name="Voorraad Analyse (3× verkoop)",
            project_type=ProjectType.PRODUCT_TARGET,
            client_id="target_voorraad",
            total_value=15000.0,  # 3 × €5000
            start_month="2025-02",
            end_month="2025-04",
            assigned_partners=["tobias"],
            status=ProjectStatus.PIPELINE,
            description="Target: 3 klanten × €5.000",
            monthly_distribution={"2025-02": 5000, "2025-03": 5000, "2025-04": 5000}
        ),
        # Product targets - AI Contract Checker (2× extra)
        Project(
            id="target-ai-2025",
            name="AI Contract Checker (2× extra verkoop)",
            project_type=ProjectType.PRODUCT_TARGET,
            client_id="target_ai",
            total_value=10000.0,  # 2 × €5000
            start_month="2025-03",
            end_month="2025-04",
            assigned_partners=["tobias"],
            status=ProjectStatus.PIPELINE,
            description="Target: 2 extra klanten × €5.000 (na WVC pilot)",
            monthly_distribution={"2025-03": 5000, "2025-04": 5000}
        ),
        # Product targets - BLOP Analyse (3×)
        Project(
            id="target-blop-2025",
            name="BLOP Analyse (3× verkoop)",
            project_type=ProjectType.PRODUCT_TARGET,
            client_id="target_blop",
            total_value=15000.0,  # 3 × €5000
            start_month="2025-02",
            end_month="2025-04",
            assigned_partners=["tobias"],
            status=ProjectStatus.PIPELINE,
            description="Target: 3 klanten × €5.000 - Open velden analyse Syntess",
            monthly_distribution={"2025-02": 5000, "2025-03": 5000, "2025-04": 5000}
        ),
    ]
    store.save_projects(projects)
