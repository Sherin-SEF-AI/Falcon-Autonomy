# Cloud bursts: one study on a rented GPU, priced and approved first

A cloud burst runs one study — a failure-rate estimate, a sweep or a rare-case search of a scenario on a CARLA town —
on a GPU machine rented from **RunPod** or **Lambda**, brings its results into the workspace, and terminates the
machine. Nothing is ever launched without an estimate you approved, under caps that refuse even an approved burst, and
an independent watchdog terminates the machine whatever happens to Falcon. For your own machines see
[cluster.md](cluster.md).

```
falcon cloud estimate --provider runpod --list-gpus                     # the GPUs and their price now
falcon cloud estimate PROJECT --kind failrate --scenario lead_brake --provider runpod --gpu "NVIDIA RTX A5000"
    estimate 3f2c…: failrate on runpod NVIDIA RTX A5000 at $0.27/h (on demand)
      evaluations 300–800 (expected 500), 6 s each (measured …)
      cost: $0.29 low, $0.29 expected, $0.43 high (…)
      TTL 1.31 h: terminated then whatever happens — $0.35 at most
      to run it: falcon cloud burst --approve 3f2c…
falcon cloud burst --approve 3f2c…
falcon cloud status | ledger | orphans [--terminate] | terminate BURST | watchdog
```

Settings ▸ Compute holds the provider, the worker image, the caps (per burst, $10; per month, $50), the TTL factor
(1.5) and the API keys (kept in the keyring, shown only as "stored in …"). The assistant has `cluster_spot_estimate`
and `cluster_spot_burst`; the burst **asks you on a card in every permission mode**, Autopilot included, with no
"approve all".

## The estimate

The study becomes a **study bundle** (`cluster/bundle.py`): the project reduced to the study's world, rig and traffic
profile, the scenario, and allow-listed options — kilobytes, capped at 64 KiB. A CARLA town only (an imported map's
OpenDRIVE file stays here), the criticality objective only (a model's weights do not go), and never anything derived
from your dashcam footage. By default it runs **without rendering** (`no_render`: CARLA started with `-nullrhi` and a
rig without sensors): the criticality needs actor states, not images, and a provider's containers may have no Vulkan
for CARLA to render with. `--render` keeps the rig's cameras, at the risk that CARLA does not start there.

Its evaluations (low, expected, high: a sweep or search is exact; a failure-rate estimate between one stage plus the
final and every stage plus the final) times this machine's measured seconds per evaluation (or a stated prior before a
drive was measured — measured with rendering, so a `no_render` study usually runs faster) give hours, plus a quarter
hour to boot; the provider's price now gives dollars. The **TTL** is the expected hours times the TTL factor; the
**most a burst can cost** is the TTL at that price, because the watchdog terminates it there. The estimate is saved
under an **id** that hashes everything it depends on — the plan, the bundle, the provider, the GPU, the price, the
TTL, the image — so a different price is a different id.

## Approval and caps

`falcon cloud burst --approve ID` prices the plan again. If the id comes out different (the price moved, the study
changed), **nothing is launched** and the new id is printed: approve that one. An approval runs one burst (an id
already used is refused) and is stale after a day. Then the caps: a burst whose worst case is over the cap per burst
is refused; so is one that would take the month's spending — finished bursts' costs plus every running burst's worst
case — over the monthly cap. The tests check that no launch call reaches a provider in any of these cases.

## A burst's life

1. The ledger (`cloud/ledger.jsonl` in Falcon's config folder: append-only, fsynced) records the **intent** before
   anything is launched.
2. An **Ed25519 key of the burst's own** is made (its private half 0600 in `cloud/bursts/<id>/`); your own keys never
   leave this machine.
3. The machine is launched (named `falcon-burst-<id>-<deadline>`) with that public key; the ledger records it, and
   the watchdog's timer starts.
4. When it is running, Falcon puts the bundle on it over SSH and runs `falcon-worker-daemon oneshot` there; each
   progress line is the burst's sign of life.
5. The summary comes back into `studies/<name>/`: the study's files (study.json, results.jsonl, the estimate) and the
   **ten most critical runs**' drive.json, run.json and recorder log — not their frames — with the machine's paths
   made this workspace's; `cloud.json` there says it came from a burst.
6. The machine is **terminated and the termination confirmed** (the provider says terminated, or no longer knows it);
   the ledger records the cost (hours × price). Any error on the way still terminates. A terminate that fails is
   retried with backoff (2, 4, 8 s); if it keeps failing, the burst is marked *terminate_failed*, a **red banner**
   shows under the page header in Falcon (and `falcon cloud status` prints it), a desktop notification is sent, and
   the watchdog keeps trying every minute.

## The watchdog

An independent `systemd-run --user` timer (unit `falcon-cloud-watchdog`, every 60 s, started with the first burst,
stopped when none is left) runs `falcon cloud watchdog`, which terminates a burst that is **past its TTL**, **silent
for 10 minutes** (its driver stopped: Falcon closed, the laptop asleep), **done**, or **over its approved cost or the
monthly cap**. It does not depend on Falcon running. Without systemd, Falcon says so, and the burst still ends itself
and is swept at Falcon's next start. While you are logged out a user timer runs only with lingering
(`loginctl enable-linger "$USER"`).

**Orphans.** At every Falcon start (in the background, and only when a burst is active or a provider key is stored)
Falcon runs the watchdog's pass and lists the Falcon-named machines at each provider: one of this machine's bursts
that is finished or past its TTL is terminated; another machine's (unknown to this ledger) only once past the
deadline in its name plus ten minutes — another machine's burst is never cut short. `falcon cloud orphans
[--terminate]` does the same by hand.

## Keys

The API keys are in the keyring (`runpod_api_key`, `lambda_api_key`; your secrets file, mode 0600, without one), sent
only in the `Authorization` header, and scrubbed from every message, log line and file Falcon writes. The tests run a
whole burst with fake keys and search every file Falcon wrote, and its output, for them.

## The providers

Read from their documentation on **2026-11-12**; field names not shown there are pinned by the fake servers in
`tests/test_cloud.py` until the acceptance run confirms them.

**RunPod — REST API v2** (`https://api.runpod.io/v2`, `Authorization: Bearer`; errors are RFC 9457 problem+json).
REST v1 (`rest.runpod.io/v1`) retires on 2026-11-15, so Falcon speaks only v2 — and **v2 has no spot
("interruptible") pods**: a RunPod burst is on demand (the plan's "interruptible = spot" applied to v1). Prices from
`GET /catalog/gpus` (`price.secure` per GPU-hour; the setting `cloud.runpod_cloud` picks COMMUNITY); `POST /pods`
(`name`, `image`, `gpu {id, count}`, `cloud`, `disk`, `ports ["22/tcp"]`, `env {PUBLIC_KEY}`, `startSsh`);
`GET /pods/{id}` (`status`, `ssh.direct`); `DELETE /pods/{id}`; `GET /pods` (`pods`, `pagination`). The pod runs the
worker image, whose entry point installs the burst's key for its user `carla` and starts sshd.
Pages: docs.runpod.io/api-reference/overview, …/api-reference-v2/migrate-from-v1, …/api-reference-v2/pods/create-a-pod,
…/api-reference-v2/catalog/list-gpu-types, …/pods/configuration/use-ssh. Not confirmed: the exact shape of
`ssh.direct` (`host`, `port`) and the bodies of `DELETE /pods/{id}` and `GET /pods`.

**Lambda — Cloud API v1** (`https://cloud.lambda.com/api/v1` as its documentation shows it; the setting
`cloud.lambda_base_url` overrides it; `Authorization: Bearer`; `{"data": …}` / `{"error": {"code", "message"}}`).
**Lambda has no spot instances**, and **its billing stops only when the instance is terminated** through the API or
the console — shutting it down from inside does not stop it, which is why Falcon always terminates through the API
and confirms. `GET /instance-types` (`price_cents_per_hour`, `regions_with_capacity_available`), `POST /ssh-keys`
(the burst's key, removed after), `POST /instance-operations/launch` (`region_name`, `instance_type_name`,
`ssh_key_names`, `name` → `instance_ids`), `GET /instances/{id}` (`status`, `ip`), `POST
/instance-operations/terminate`, `GET /instances` (`page_token`). Falcon logs in as `ubuntu` and runs the worker image
with `docker run --gpus all`. Pages: docs.lambda.com/api/cloud and its OpenAPI file (docs.lambda.ai redirects there).
Not confirmed: the ssh-keys paths and bodies, `GET /instances/{id}`'s body, the base URL's host
(cloud.lambda.com vs cloud.lambda.ai), and whether the Lambda image's `ubuntu` user may run Docker.

## The worker image

[packaging/cluster/Dockerfile](../packaging/cluster/Dockerfile): `carlasim/carla:0.9.16`, CARLA's client and Falcon
each in a Python 3.12 venv, a stand-in `systemd-run` (the container's limit is the memory cap), sshd for a burst's
key, everything run as the user `carla`. Build it and push it where the provider can pull it; its name is the setting
`cloud.image` (or `--image`).

## What was tested, and what needs your accounts

Tested with fakes only: fake RunPod and Lambda servers on 127.0.0.1 (prices, launch, status, a terminate that fails
twice then succeeds, one that keeps failing), the study run by `oneshot` here on the fake CARLA, the watchdog on a fake
clock (TTL, silence, done, over its cost, over the monthly cap), the orphan sweep, the caps and approval refusals with
zero launch calls, and the keys never written anywhere. Not yet run for real: a RunPod and a Lambda burst (capped at
$5 each and approved at the time, at acceptance), the worker image built and pulled by the providers, CARLA under
`-nullrhi` in their containers, and the field names listed above as not confirmed.
