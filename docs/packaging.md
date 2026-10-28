# Installing and packaging

## The Linux bundle

`packaging/build.sh` builds a one-folder bundle with PyInstaller. It produces `dist/falcon/`, plus
`dist/falcon-<version>-linux-x86_64.tar.gz` and its `.sha256`. PyInstaller runs from a throw-away venv beside the
repository (`.venv-build`, which can see the system's packages), so nothing is installed into the system Python.

```sh
./packaging/build.sh
tar xzf dist/falcon-*-linux-x86_64.tar.gz -C ~/opt
~/opt/falcon-*/falcon              # the application
~/opt/falcon-*/falcon doctor       # the CLI: every command of docs/cli.md
```

The single executable is three things:

- **The application:** run with no arguments, or with a project file.
- **The command line:** run with a command, such as `falcon doctor` or `falcon drive`.
- **A worker host for the fake carla:** the bundle runs `falcon -m falcon_autonomy …` for the jobs the app starts.

Real workers run in **CARLA's own Python**, not in the bundle. The bundle carries Falcon's sources as data files, and
the workers get them on `PYTHONPATH`, so CARLA's venv needs nothing but `carla==0.9.16`, numpy and OpenCV.

**What it contains:** PyQt6, numpy, SciPy, OpenCV, onnxruntime (CPU), PyAV, pyarrow, and torch with torchvision and
its CUDA libraries, so the Training lab trains on the GPU straight from the bundle.

**Size, measured:** 6.9 GB unpacked and 4.3 GB compressed. Most of it, 4.3 GB, is torch's CUDA libraries. Packages
the system Python has but Falcon does not use are left out (`packaging/falcon.spec`, `EXCLUDES`): triton, polars,
pandas, transformers and similar.

**Checked from the built bundle (0.6.0, this machine):**

- `falcon --help` and `falcon doctor`;
- the GPU path: `falcon gap --features`, with ResNet-50 features on CUDA;
- a drive worker started through the bundle against the fake carla;
- the GUI, started and closed.

**Not bundled:**

- **CARLA:** the first-run wizard finds it.
- **GStreamer:** the hardware-in-the-loop streams use the system's GStreamer.
- **Docker and ROS 2:** stack testing uses them when present.
- **Model weights:** downloaded only when asked.

## A desktop entry

`packaging/falcon-autonomy.desktop` and `falcon_autonomy/resources/icons/falcon.png` give the bundle a launcher.
Copy the entry to `~/.local/share/applications/`, set its `Exec` to the unpacked `falcon`, and copy the icon to
`~/.local/share/icons/falcon-autonomy.png`.

## From source

```sh
pip install -e .                   # PyQt6, numpy, OpenCV, jsonschema, pyqtgraph, PyOpenGL
python3 -m falcon_autonomy         # torch, torchvision, onnxruntime, scipy, mcap: optional, per feature
```

## Windows

Not supported in this release. Falcon's server manager and workers rely on Linux:

- systemd user scopes cap memory with no swap;
- `fcntl` file locks implement the machine-wide CARLA and GPU locks;
- Unix sockets carry the live view, the environment and the bridges;
- the signal handling is POSIX.

A Windows port needs replacements for each of these: a Job Object for the memory cap, `msvcrt` locks, named pipes. It
needs a Windows CARLA 0.9.16 to test against, too. The PyInstaller spec itself would carry over.
