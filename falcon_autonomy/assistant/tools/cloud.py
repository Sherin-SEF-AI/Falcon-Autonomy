"""Cloud-burst tools (the cluster group; 4.0, in progress): a priced estimate and an approved burst on RunPod or
Lambda, under the caps in Settings ▸ Compute (cluster_spot_estimate, cluster_spot_burst — the burst asks the user in
every mode). No tool yet in this build: the module is imported by tools/__init__ so that the workstream adding them
only fills it in (adding its descriptions to _compact.SHORT as tools/cluster.py does)."""
from __future__ import annotations
