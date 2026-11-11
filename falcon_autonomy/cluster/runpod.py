"""RunPod's REST API v2 (standard library; docs/cloud.md, "RunPod").

Read from RunPod's documentation on 2026-11-12 (docs/cloud.md lists the pages): base https://api.runpod.io/v2, an
"Authorization: Bearer <key>" header, RFC 9457 problem+json errors (title, status, detail). REST v1
(rest.runpod.io/v1) retires on 2026-11-15, so Falcon speaks v2 only — and v2 has no spot ("interruptible") pods:
a RunPod burst is on demand, priced at the catalogue's secure-cloud price (the setting cloud.runpod_cloud: SECURE or
COMMUNITY).

    GET    /catalog/gpus          {"gpus": [{"id", "name", "memory", "price": {"secure", "community"}, …}]}  USD/h/GPU
    POST   /pods                  {"name", "image", "gpu": {"id", "count"}, "cloud", "disk", "ports": ["22/tcp"],
                                   "env": {"PUBLIC_KEY": …}, "startSsh": true}  -> 201, the pod
    GET    /pods/{id}             the pod: "status" PROVISIONING|STARTING|RUNNING|EXITED|ERROR|TERMINATED, "cost",
                                  "ssh": {"direct": {"host", "port"}} once 22/tcp is published and it runs
    DELETE /pods/{id}             terminate (a 404 means it is gone already)
    GET    /pods                  {"pods": [...], "pagination": {"nextCursor", "hasNextPage"}}

The pod runs Falcon's worker image (packaging/cluster/Dockerfile), whose entry point puts PUBLIC_KEY — the burst's
own Ed25519 key, never the user's — into the image user carla's authorized_keys and starts sshd. Field names not confirmed by a live
call are pinned by the fake server in tests/test_cloud.py; the real check is the acceptance run.
"""
from __future__ import annotations

from .cloud import CloudError, Provider

STATES = {'PROVISIONING': 'booting', 'STARTING': 'booting', 'RUNNING': 'running', 'EXITED': 'stopped',
          'ERROR': 'error', 'TERMINATED': 'terminated'}


class RunPod(Provider):
    name = 'runpod'
    ssh_user = 'carla'                    # the worker image's user (CARLA refuses root): its entry point installs the key
    base = 'https://api.runpod.io/v2'
    cloud = 'SECURE'

    def error_text(self, doc) -> str:
        if isinstance(doc, dict):
            return ' — '.join(str(doc[k]) for k in ('title', 'detail') if doc.get(k)) or str(doc)[:300]
        return 'no details'

    def prices(self) -> list:
        status, doc = self.request('GET', '/catalog/gpus')
        if status != 200 or not isinstance(doc, dict):
            raise self.error('reading the GPU catalogue', status, doc)
        key = self.cloud.lower()
        out = []
        for g in doc.get('gpus') or ():
            price = (g.get('price') or {}).get(key)
            out.append({'gpu': g.get('id'), 'name': g.get('name') or g.get('id'),
                        'usd_per_hour': float(price) if price is not None else None,
                        'vram_gb': g.get('memory'), 'available': bool(g.get(key, True)), 'spot': False})
        return out

    def launch(self, name: str, gpu: str, public_key: str, image: str) -> dict:
        body = {'name': name, 'image': image, 'gpu': {'id': gpu, 'count': 1}, 'cloud': self.cloud, 'disk': 60,
                'ports': ['22/tcp'], 'env': {'PUBLIC_KEY': public_key}, 'startSsh': True}
        status, doc = self.request('POST', '/pods', body)
        if status not in (200, 201) or not isinstance(doc, dict) or not doc.get('id'):
            raise self.error('creating the pod', status, doc)
        return {'id': str(doc['id']), 'cost_per_hour': doc.get('cost')}

    def status(self, instance_id: str) -> dict:
        status, doc = self.request('GET', f'/pods/{instance_id}')
        if status == 404:
            return {'state': 'missing'}
        if status != 200 or not isinstance(doc, dict):
            raise self.error('reading the pod', status, doc)
        direct = ((doc.get('ssh') or {}).get('direct') or {}) if isinstance(doc.get('ssh'), dict) else {}
        return {'state': STATES.get(str(doc.get('status') or '').upper(), 'booting'),
                'ip': direct.get('host') or direct.get('ip'), 'ssh_port': direct.get('port') or 22}

    def terminate(self, instance_id: str):
        status, doc = self.request('DELETE', f'/pods/{instance_id}')
        if status in (200, 202, 204, 404):
            return
        raise self.error('terminating the pod', status, doc)

    def instances(self) -> list:
        out, cursor = [], ''
        for _ in range(50):
            status, doc = self.request('GET', '/pods' + (f'?cursor={cursor}' if cursor else ''))
            if status != 200 or not isinstance(doc, dict):
                raise self.error('listing the pods', status, doc)
            for p in doc.get('pods') or ():
                out.append({'id': str(p.get('id')), 'name': p.get('name') or '',
                            'state': STATES.get(str(p.get('status') or '').upper(), 'booting')})
            pg = doc.get('pagination') or {}
            if not pg.get('hasNextPage') or not pg.get('nextCursor'):
                return out
            cursor = str(pg['nextCursor'])
        raise CloudError('runpod: too many pages of pods')
