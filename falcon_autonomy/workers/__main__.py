"""Entry point of every worker: python -m falcon_autonomy.workers <name> [args].

SIGINT and SIGTERM are blocked here, first, before anything imports numpy: numpy's PyPI OpenBLAS starts about 23
threads at import, and a signal delivered to one of them would end the worker at once (SIGTERM) or raise a
KeyboardInterrupt the watchdog never sees (SIGINT) — CARLA left untidied in synchronous mode either way. Every thread
started after this inherits the block; only the watchdog thread (common.Watchdog) takes them, with sigtimedwait.
"""
import faulthandler
import signal
import sys

if hasattr(signal, 'pthread_sigmask'):
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
# libcarla ends the process with std::terminate (SIGABRT) on some errors in its own threads: print every Python thread's
# stack to stderr (the worker's log) so the call that led there is known
faulthandler.enable(all_threads=True)


def main(argv=None) -> int:
    import importlib
    from . import WORKERS
    from ..core import protocol as P
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in WORKERS:
        print(f'usage: python -m falcon_autonomy.workers {{{",".join(WORKERS)}}} [args]', file=sys.stderr)
        return P.FAILED
    mod = importlib.import_module(WORKERS[argv[0]])
    return mod.main(argv[1:])


if __name__ == '__main__':
    sys.exit(main())
