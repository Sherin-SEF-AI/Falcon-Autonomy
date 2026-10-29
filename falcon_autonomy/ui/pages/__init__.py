"""The pages of the main window, in navigator order and grouped as the navigator shows them. A page appears here
only when it does what it says: pages of later milestones are added as they are built (docs/roadmap.md)."""
from .data import DataPage
from .hardware import HardwarePage
from .help import HelpPage
from .home import HomePage
from .live import LivePage
from .loop import LoopPage
from .models import ModelsPage
from .rarecases import RareCasesPage
from .replay import ReplayPage
from .reports import ReportsPage
from .rigs import RigsPage
from .scenarios import ScenariosPage
from .settings import SettingsPage
from .stack import StackPage
from .training import TrainingPage
from .worlds import WorldsPage

GROUPS = [
    ('', [HomePage]),
    ('Build', [WorldsPage, RigsPage, ScenariosPage]),
    ('Drive', [LivePage, DataPage, ReplayPage]),
    ('Improve', [LoopPage, RareCasesPage, TrainingPage, ModelsPage]),
    ('Ship', [StackPage, HardwarePage, ReportsPage]),
    ('', [SettingsPage, HelpPage]),
]
PAGES = [cls for _g, classes in GROUPS for cls in classes]
