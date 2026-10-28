#!/bin/sh
# Build the Linux x86_64 bundle: dist/falcon/ and dist/falcon-<version>-linux-x86_64.tar.gz (with SHA256SUMS).
# PyInstaller comes from a throw-away venv beside the repository (.venv-build, gitignored) that sees the system
# packages (PyQt6, numpy, torch …), so nothing is installed into the system Python.
set -eu
cd "$(dirname "$0")/.."
[ -x .venv-build/bin/python ] || python3 -m venv --system-site-packages .venv-build
.venv-build/bin/python -c 'import PyInstaller' 2>/dev/null || .venv-build/bin/pip install -q 'pyinstaller>=6.16'
VERSION=$(.venv-build/bin/python -c 'import falcon_autonomy; print(falcon_autonomy.__version__)')
.venv-build/bin/python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller \
    packaging/falcon.spec
NAME="falcon-$VERSION-linux-x86_64"
tar -C dist -czf "dist/$NAME.tar.gz" --transform "s,^falcon,$NAME," falcon
(cd dist && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
dist/falcon/falcon --help > /dev/null
echo "built dist/$NAME.tar.gz"
