#!/bin/sh
# A stand-in for systemd-run inside the Falcon worker image (packaging/cluster/Dockerfile): a container has no systemd,
# and its own memory limit is the cap. Falcon calls `systemd-run --user --scope -q -p MemoryMax=… [--unit U] CMD …`;
# this drops the options and runs CMD in place (exec: the same pid, as systemd-run --scope does).
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--property|--unit|-u|--description|--slice|--setenv|-E) shift 2 ;;
        --*=*|--user|--scope|-q|--quiet|--collect|-G|--same-dir) shift ;;
        --) shift; break ;;
        -*) shift ;;
        *) break ;;
    esac
done
exec "$@"
