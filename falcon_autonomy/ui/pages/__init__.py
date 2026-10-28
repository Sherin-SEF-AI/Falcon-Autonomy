"""The pages of the main window, in navigator order. A page appears here only when it does what it says: pages of
later milestones are added as they are built (docs/roadmap.md)."""
from .data import DataPage
from .help import HelpPage
from .home import HomePage
from .live import LivePage
from .rarecases import RareCasesPage
from .rigs import RigsPage
from .scenarios import ScenariosPage
from .settings import SettingsPage
from .worlds import WorldsPage

PAGES = [HomePage, WorldsPage, LivePage, RigsPage, ScenariosPage, DataPage, RareCasesPage, SettingsPage, HelpPage]
