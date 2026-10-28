"""Starting the application: QApplication, fonts, the theme, the main window, the first-run wizard."""
from __future__ import annotations

import os
import sys

from .. import APP_ID, APP_NAME


def make_app():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ID)
    return app


def main(project_path: str | None = None, fake_carla: str = '') -> int:
    from PyQt6.QtGui import QIcon
    from ..core.project import Project, ProjectError
    from ..core.settings import Settings
    from .mainwindow import MainWindow
    from .theme import THEME, load_fonts
    app = make_app()
    load_fonts()
    st = Settings()
    THEME.apply(app, st.get('theme') or 'dark')
    icon = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'falcon.png')
    if os.path.isfile(icon):
        app.setWindowIcon(QIcon(icon))
    project = None
    path = project_path or next((p for p in st.get('recent_projects') or [] if os.path.isfile(p)), None)
    if path:
        try:
            project = Project.load(path)
        except (OSError, ProjectError) as e:
            print(f'could not open {path}: {e}', file=sys.stderr)
    win = MainWindow(st, project, fake_carla=fake_carla)
    win.show()
    if not st.get('first_run_done') and not fake_carla:
        win.show_wizard()
    return app.exec()
