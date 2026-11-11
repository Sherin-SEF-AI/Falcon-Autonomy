"""Falcon's distributed worker mesh and cloud burst (docs/cluster.md, docs/cloud.md).

    protocol    the daemon's wire format: framed JSON messages, artifacts packed as a compressed tar
    bundle      what a remote job carries: its config and the content of its inputs, never this machine's paths
    nodes       the user's cluster (cluster.json: nodes, slots, pools; key paths only)
    daemon      `falcon worker-daemon serve --slot N` (a node's slot) and `oneshot` (one bundle, for Slurm/K8s)
    transport   SSH tunnels with keys (paramiko, imported only when a node needs it)
    scheduler   slots, matching (the backends each node advertises), heartbeats, loss and re-queue, pools and quotas
    launcher    RemoteLauncher: the JobQueue's door to the scheduler
    slurm, k8s  batch backends: sbatch arrays, Kubernetes Jobs (kubectl)
    cloud       cost estimates, approved bursts under caps, the ledger, the independent watchdog
    runpod, lambda_   the providers' HTTP APIs (standard library)
    gc          the caches nothing refers to: the workspace's content store, the nodes' scratch, Slurm array folders
    cli, cloudcli     `falcon cluster …`, `falcon cloud …`
"""
