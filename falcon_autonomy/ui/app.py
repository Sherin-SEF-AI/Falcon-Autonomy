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
    app.setDesktopFileName('falcon-autonomy')   # Wayland app id: the dock matches the window to falcon-autonomy.desktop
    return app


def main(project_path: str | None = None, fake_carla: str = '') -> int:
    from PyQt6.QtGui import QIcon
    from ..core.project import Project, ProjectError
    from ..core.settings import Settings
    from .mainwindow import MainWindow
    from .theme import THEME, load_fonts
    app = make_app()
    # PyQt ends the process on an exception raised in a slot when no hook is set (as in the bundle): print it to the
    # terminal and keep the session — one failing button must not take the user's unsaved work with it
    import traceback
    sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb)
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
    # Ctrl+C in the terminal closes the window as File ▸ Quit does (jobs asked to stop, the server stopped), instead
    # of a KeyboardInterrupt inside a Qt slot; the timer lets Python run its signal handler while Qt waits
    import signal
    from PyQt6.QtCore import QTimer
    def interrupt(*_):
        if win.close():                   # it may ask to save first; then every window goes (the wizard's too)
            app.closeAllWindows()
            app.quit()
    signal.signal(signal.SIGINT, interrupt)
    tick = QTimer(app)
    tick.timeout.connect(lambda: None)
    tick.start(250)
    if not st.get('first_run_done') and not fake_carla:
        win.show_wizard()
    return app.exec()
