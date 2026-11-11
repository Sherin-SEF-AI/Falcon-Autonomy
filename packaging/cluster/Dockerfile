# The Falcon worker image: CARLA 0.9.16 with Falcon on top, for Kubernetes Jobs and cloud bursts (docs/cluster.md,
# docs/cloud.md). Build it from the repository root and push it where the cluster or the provider can pull it:
#
#   docker build -f packaging/cluster/Dockerfile -t <registry>/falcon-worker:4.0 .
#   docker push <registry>/falcon-worker:4.0
#
# Everything runs as the image's own user carla (CARLA refuses to run as root). The entry point, when PUBLIC_KEY is set
# (a cloud burst's own ephemeral key, never yours), installs it for carla and starts sshd with keys only; without it
# (Kubernetes) no sshd runs.
#
# Memory caps: on a workstation Falcon starts CARLA and each worker in a systemd scope with a cap. A container has no
# systemd; its own limit (the pod's resources, docker run --memory) is the cap, and the systemd-run below only runs
# the command (it is a stand-in that drops the scope options).
#
# Rendering: CARLA needs Vulkan, i.e. the NVIDIA container toolkit's graphics capability (set below). A provider whose
# containers lack it can still run a criticality study: Falcon starts CARLA there with -nullrhi (no_render) and a rig
# without sensors.

FROM carlasim/carla:0.9.16

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip openssh-server tar ca-certificates util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && python3 -m pip install --no-cache-dir uv \
 && UV_PYTHON_INSTALL_DIR=/opt/python uv python install 3.12

# CARLA's client in its own venv (the workers' Python), Falcon in another (the daemon's), as on a workstation
RUN UV_PYTHON_INSTALL_DIR=/opt/python uv venv --python 3.12 /opt/carla-venv \
 && uv pip install --python /opt/carla-venv/bin/python --no-cache carla==0.9.16 numpy opencv-python-headless \
 && UV_PYTHON_INSTALL_DIR=/opt/python uv venv --python 3.12 /opt/falcon
COPY . /opt/falcon-src
RUN uv pip install --python /opt/falcon/bin/python --no-cache --no-deps /opt/falcon-src \
 && uv pip install --python /opt/falcon/bin/python --no-cache numpy jsonschema opencv-python-headless \
 && ln -s /opt/falcon/bin/falcon-worker-daemon /usr/local/bin/falcon-worker-daemon \
 && install -m 755 /opt/falcon-src/packaging/cluster/systemd-run-shim.sh /usr/local/bin/systemd-run \
 && install -m 755 /opt/falcon-src/packaging/cluster/entrypoint.sh /usr/local/bin/falcon-entrypoint \
 && mkdir -p /home/carla/.config/FalconAutonomy /home/carla/.ssh \
 && printf '{"carla_root": "/home/carla", "client_python": "/opt/carla-venv/bin/python"}\n' \
      > /home/carla/.config/FalconAutonomy/settings.json \
 && chown -R carla /home/carla/.config /home/carla/.ssh && chmod 700 /home/carla/.ssh

WORKDIR /home/carla
ENTRYPOINT ["/usr/local/bin/falcon-entrypoint"]
CMD ["sleep", "infinity"]
