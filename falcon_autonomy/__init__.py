"""Falcon Autonomy: one application for the loop of building a self-driving AI on the CARLA simulator.

The package is split by process (docs/architecture.md):

* falcon_autonomy.core     standard library (+ numpy where noted): the project, provenance, the catalogue, jobs,
                           budgets. Imported by the GUI, the CLI and the workers alike; never imports Qt or carla.
* falcon_autonomy.sim      the GUI side of CARLA: finding it, starting and stopping servers, building worker commands.
* falcon_autonomy.workers  run in CARLA's own Python (the venv with the carla wheel); never import Qt.
* falcon_autonomy.ui       the PyQt6 application; never imports carla.
"""
__version__ = '0.3.0'
APP_NAME = 'Falcon Autonomy'
APP_ID = 'FalconAutonomy'
