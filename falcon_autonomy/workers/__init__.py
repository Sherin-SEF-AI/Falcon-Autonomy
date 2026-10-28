"""Workers: processes that run in CARLA's own Python (the venv with the carla wheel) and never import Qt.

    python -m falcon_autonomy.workers <name> [args]

Each speaks the JSON-lines protocol (falcon_autonomy.core.protocol), writes a run.json describing exactly what ran,
and exits with a documented code. This package's __init__ imports nothing: __main__ must block SIGINT and SIGTERM
before numpy (whose OpenBLAS starts its threads at import) is imported.
"""
WORKERS = {
    'probe': 'falcon_autonomy.workers.probe',
    'tidy': 'falcon_autonomy.workers.tidy',
    'drive': 'falcon_autonomy.workers.drive',
    'live': 'falcon_autonomy.workers.live',
}
