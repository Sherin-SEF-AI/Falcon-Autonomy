"""The bundle's entry point (PyInstaller). One executable is the application, its CLI and — for the fake carla of
the tests and demos only — a worker host:

    falcon                                   the application (as `python -m falcon_autonomy`)
    falcon <command> ...                     the CLI (doctor, drive, evaluate, train, ...)
    falcon -m falcon_autonomy <command> ...  the same: jobs the app starts as `sys.executable -m falcon_autonomy …`
    falcon -m falcon_autonomy.workers ...    a worker in this process's Python: only meaningful with the fake carla,
                                             since real workers run in CARLA's own Python with the sources the bundle
                                             carries as data files (sim.server.falcon_source)

A frozen interpreter ignores PYTHONPATH; its entries are put on sys.path here so a worker started this way finds
the fake carla package its caller put there.
"""
import os
import sys


def _paths():
    for p in reversed([x for x in os.environ.get('PYTHONPATH', '').split(os.pathsep) if x]):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    argv = sys.argv[1:]
    if argv[:2] == ['-m', 'falcon_autonomy.workers']:
        _paths()
        sys.argv = [sys.argv[0]] + argv[2:]
        from falcon_autonomy.workers.__main__ import main as worker_main
        return worker_main(argv[2:])
    if argv[:2] == ['-m', 'falcon_autonomy']:
        argv = argv[2:]
    if argv[:1] == ['-u']:
        argv = argv[1:]
    from falcon_autonomy.cli import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()          # torch data-loader workers in a frozen app
    sys.exit(main())
