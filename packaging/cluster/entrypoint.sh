#!/bin/sh
# The Falcon worker image's entry point (packaging/cluster/Dockerfile). With PUBLIC_KEY (a cloud burst's own
# ephemeral key): that key alone in carla's authorized_keys, sshd with keys only. Then the command, as carla (CARLA
# refuses to run as root).
set -e
if [ -n "$PUBLIC_KEY" ] && [ "$(id -u)" = 0 ]; then
    install -d -m 700 -o carla /home/carla/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /home/carla/.ssh/authorized_keys
    chown carla /home/carla/.ssh/authorized_keys
    chmod 600 /home/carla/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=no
fi
if [ "$(id -u)" = 0 ]; then
    exec runuser -u carla -- "$@"
fi
exec "$@"
