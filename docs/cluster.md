# The cluster: running a study's drives on other machines

A sweep, a rare-case search, a failure-rate estimate or a regression bisect can send its scenario drives to other
machines — your LAN workstations, a Slurm cluster, a Kubernetes cluster — while the study itself runs here and decides
what to drive next. Results come back into the study's own run folders, so everything downstream (the study's results,
analysis, reports, the catalogue) is the same as for a local run. Cloud bursts are a separate path:
[cloud.md](cloud.md).

```
falcon sweep PROJECT --scenario lead_brake --name s1 --samples 200 --on cluster        # every pool
falcon search PROJECT --scenario cut_in --name s2 --budget 400 --on cluster:gpu         # one pool
falcon failrate run PROJECT --scenario lead_brake --on cluster:gpu
falcon bisect run REPO --config ci.json --good v1 --bad v2 --on cluster
falcon cluster dispatch sweep PROJECT --pool gpu --scenario lead_brake --name s1 --samples 200   # the same
```

In the window: Settings ▸ Compute lists the nodes and pools; the Jobs dock gains a **Node** column as soon as a job
runs elsewhere (the node and slot, or the pool it waits for; its tooltip says how often it was queued again). The
assistant has `cluster_nodes`, `cluster_quotas` and `cluster_dispatch` ([assistant tools](assistant-tools.md)).

## What goes, and what never does

Only drives go (the `drive` worker: every scenario run of a study). A remote job is a **bundle**, never a path: the
drive's config with what only makes sense here removed (where it is written, the server's address and ports — the
node uses its own) and the *content* of every file the config names (an imported world's OpenDRIVE file, say), each
checked by its SHA-256 on arrival. The node builds the worker's command itself from an allow-list; a client cannot
make it run anything else. Refused, before anything leaves:

- **anything derived from your dashcam footage** — a drive along a dashcam route, a scenario made from a dashcam
  incident, a study of one (the same test as the assistant's privacy rules, `assistant/privacy.py`);
- interactive jobs (a live session, the gym, the adversary, HIL): they talk to you while they run;
- a config that names a folder (it cannot be carried) — except the ODD data monitor's weights folder, which is left
  out with a note: the node monitors the ODD's conditions without the feature monitor;
- a bundle over 5 MiB compressed (1 MiB for Kubernetes: its ConfigMap limit);
- every other worker (stack runs, training, …): those jobs run on this machine and their log says so.

## Nodes

Nodes live in the private `cluster` setting (Settings ▸ Compute, or `falcon cluster nodes add`). Nothing secret is
kept: an SSH key is the **path** to your key file — never the key, never a password.

| kind | what it is | how Falcon reaches it |
|---|---|---|
| `ssh` | a workstation running the worker daemon's slots | an SSH tunnel with your key to 127.0.0.1:2350+N on it |
| `local` | daemons on this machine (a second GPU here; the tests) | 127.0.0.1:2350+N directly |
| `slurm` | a Slurm cluster | `sbatch` arrays on its shared folder; the login host over SSH, or here |
| `k8s` | a Kubernetes cluster | `kubectl` with your context: a Job per run |

```
falcon cluster nodes add ws1 --kind ssh --host ws1.lan --user jo --key ~/.ssh/id_ed25519 --slots 0,1 --pool gpu
falcon cluster nodes add hpc --kind slurm --host login.hpc --user jo --key ~/.ssh/id_ed25519 --workdir /scratch/jo/falcon \
       --gres gpu:1 --partition gpu --time-limit 02:00:00 --max-jobs 16 --pool batch
falcon cluster nodes add kube --kind k8s --context lab --namespace falcon --image registry.lab/falcon-worker:4.0 --gpu 1
falcon cluster nodes test ws1          # each slot's daemon answers (or the batch system does)
falcon cluster nodes list | remove NAME
```

A node may say what it can run: `--backend` (default carla-0.9.16; one per CARLA backend it has) and `--vram-gb`; a
job that names a backend or video memory (`meta['backend']`, `meta['vram_gb']`) goes only to a node that has them.

### A workstation as a node

On the workstation: install Falcon and CARLA as usual (its Settings: CARLA's folder and its Python), put your public
key in `~/.ssh/authorized_keys` (`ssh-copy-id` from here), and start one daemon per slot as a systemd **user** unit —
Falcon never needs root ([packaging/cluster/falcon-worker-daemon@.service](../packaging/cluster/falcon-worker-daemon@.service)):

```
cp packaging/cluster/falcon-worker-daemon@.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now falcon-worker-daemon@0      # slot 0; @1 for slot 1 …
loginctl enable-linger "$USER"                            # optional: keep it running while logged out
```

Slot N listens on **127.0.0.1:2350+N only** — never on the network; this machine reaches it through an SSH tunnel —
and runs its own Falcon service in a scratch workspace (`~/.cache/FalconAutonomy/worker-N`): its CARLA server on
2310+10N (traffic manager 8310+N), started on demand under Falcon's memory cap, a GPU's CARLA lock shared with every
other Falcon process there. On start it writes a random token to `~/.config/FalconAutonomy/cluster/daemon-N.token`
(readable only by you); a client must present it, so another user of the workstation cannot use your slot. The unit's
MemoryMax caps the daemon and everything it starts.

SSH follows `falcon hil deploy`'s rules: keys only (no agent, no other key, no password), a host key accepted on first
contact and kept in Falcon's own `cluster/known_hosts`, a changed one refused; messages never name the address or the
user. SSH needs the `cluster` extra (`pip install "falcon-autonomy[cluster]"`: paramiko); `local` nodes need nothing.

### Slurm

Jobs dispatched together go as one `sbatch --array` in a folder of their own under the node's shared folder
(`workdir`): `bundles/<i>.bundle`, `logs/<i>.out`, `out/<i>/`. Each task runs
`falcon-worker-daemon oneshot --bundle … --out …` (the node's `--falcon` command; Falcon and CARLA must be installed
where the compute nodes see them) and takes its CARLA ports from `$SLURM_LOCALID`, so tasks sharing a machine do not
collide. Falcon polls `squeue`/`sacct` (every 5 s; `poll_s`), reads each task's new log lines for progress, and brings
`out/<i>` back when the task ends; the array's folder is removed after. `NODE_FAIL`, `PREEMPTED` and `BOOT_FAIL` lose
the task (below); `scancel` cancels one.

### Kubernetes

Each run is a Job of one pod running the worker image ([packaging/cluster/Dockerfile](../packaging/cluster/Dockerfile):
`carlasim/carla:0.9.16` with Falcon; [k8s-job.yaml](../packaging/cluster/k8s-job.yaml) shows the object): the bundle in
a ConfigMap of its own, `nvidia.com/gpu` from the node's `--gpu`, `activeDeadlineSeconds` from `--deadline-s`,
`backoffLimit: 0` (Falcon decides about retries), `ttlSecondsAfterFinished`. Falcon follows `kubectl logs -f`; when
the worker is done the pod holds (`--hold-s`, 900 s) while Falcon copies its run out (`kubectl exec … tar`), then tells
it to end and deletes the Job and the ConfigMap. An evicted or failed pod loses the run; cancelling deletes the Job.
Inside the image there is no systemd: the container's own limit is the memory cap (a stand-in `systemd-run` only runs
the command), and everything runs as the image's user `carla` (CARLA refuses root).

## Scheduling

- **Slots.** A daemon slot takes one of this process's jobs at a time (the idlest first); a batch node takes up to
  `max_jobs`. A job that finds no free slot waits in the queue ("waiting for a free slot on the cluster").
- **Pools and quotas.** `falcon cluster pools set gpu --max-running 4 --max-per-day 500`: at most 4 at once (the
  rest wait), at most 500 started a day (refused past it; counted in `cluster/usage.json`). 0 is no limit.
  `falcon cluster pools` shows each pool's nodes, slots, limits and today's count.
- **Ids.** A job keeps one global id (20 hex digits); each dispatch is `<id>.<attempt>`.
- **Heartbeats and loss.** A daemon sends a heartbeat every 15 s. A slot not heard from for **60 s** is lost: a
  dropped connection is retried meanwhile, and a daemon that answers again in time is asked what became of its jobs.
  A lost job is queued again when it is resumable — a drive is: it writes only its own folder — at most **twice**
  (its log says "queued again (attempt 2 of 3)"); otherwise it fails saying why. A node that comes back after its jobs
  went elsewhere is told to cancel them.
- **Cancel** reaches the node: the worker there tidies up as it does here, and its run folder comes back marked
  cancelled.

## Results

A finished run's folder comes back from the node (zstd-compressed when both ends have Python 3.14's
`compression.zstd`, else zlib; unpacking refuses absolute names, `..`, links and devices) and is unpacked into the
job's own run folder before the queue hears the job ended, the node's folder replaced by this one in its text files.
`run.json` keeps the config as it was written here (`drive.json`) in `provenance.settings` and its hashes, and says in
a **`node`** block where it ran: the node, kind, pool, slot, dispatch id, the node's host name and Falcon version, and
the values that were the node's own (its CARLA ports, its paths) under `node.settings` with the node's config hashes.
Apart from that block, a run's `run.json` is the one a local run writes (timings aside) — the tests check exactly
this on a 12-run sweep across two daemons.

## Protocol

`falcon_autonomy/cluster/protocol.py` documents the frames (a 4-byte length, a JSON message, an optional payload):
`hello` (with the token), `submit` (a bundle), `cancel`, `attach` (after a reconnect), `release`, `ping` from the
client; `hello`, `accepted`, `refused`, `msg` (each worker protocol line as it came), `done` (+ the run folder),
`heartbeat`, `state`, `error` from the daemon.

## What the tests cover, and what needs your machines

Tested here with fakes only: two daemons on 127.0.0.1 with the fake CARLA (a 12-run sweep equal to the local run but
for `node`; a daemon killed mid-run and its job finished on the other; cancel; pools; refusals; the token and the
allow-list), fake `sbatch`/`squeue`/`sacct`/`scancel` and a fake `kubectl` on PATH running `oneshot` here (arrays, a
lost task, an evicted pod, cancel), and SSH through an in-process SSH server on loopback (tunnels, commands, host keys).
Not yet run against real machines: two LAN workstations with the systemd unit, a real Slurm array, real Kubernetes
pods with the worker image (the Dockerfile has not been built in this build: no registry or network here), and
multi-GPU workstations (each slot shares the GPU 0 CARLA lock; a per-GPU slot needs CARLA's `-graphicsadapter`, a
follow-up in the server's backend work).
