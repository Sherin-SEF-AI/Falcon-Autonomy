"""Lambda's Cloud API v1 (standard library; docs/cloud.md, "Lambda").

Read from Lambda's documentation and OpenAPI file on 2026-11-12 (docs/cloud.md lists the pages): base
https://cloud.lambda.com/api/v1 (the setting cloud.lambda_base_url overrides it), an "Authorization: Bearer <key>"
header, {"data": …} on success and {"error": {"code", "message", "suggestion"}} on failure. Lambda has no spot
instances: a burst is on demand, and an instance is billed until it is terminated through the API — shutting it
down from inside does not stop the bill, so Falcon always terminates through the API and confirms.

    GET  /instance-types                 data: {name: {"instance_type": {"name", "price_cents_per_hour", "specs",
                                         "gpu_description"}, "regions_with_capacity_available": [{"name"}]}}
    POST /ssh-keys                       {"name", "public_key"} -> data {"id", …}   (the burst's own key)
    POST /instance-operations/launch     {"region_name", "instance_type_name", "ssh_key_names", "name"}
                                         -> data {"instance_ids": [...]}
    GET  /instances/{id}                 data: {"status": booting|active|unhealthy|terminating|terminated|preempted,
                                         "ip", …}
    POST /instance-operations/terminate  {"instance_ids"} -> data {"terminated_instances": [...]}
    GET  /instances                      data: [...], page_token
    DELETE /ssh-keys/{id}                after the burst

The ssh-keys paths and bodies were not shown in full by the documentation read: they are pinned by the fake server in
tests/test_cloud.py and confirmed at acceptance. The instance is Lambda's Ubuntu image: Falcon logs in as ubuntu and
runs the worker image with Docker (`docker run --gpus all`).
"""
from __future__ import annotations

import shlex
import urllib.parse

from .cloud import CloudError, Provider

STATES = {'booting': 'booting', 'active': 'running', 'unhealthy': 'error', 'terminating': 'terminating',
          'terminated': 'terminated', 'preempted': 'terminated'}


class Lambda(Provider):
    name = 'lambda'
    ssh_user = 'ubuntu'
    base = 'https://cloud.lambda.com/api/v1'

    def error_text(self, doc) -> str:
        e = (doc or {}).get('error') if isinstance(doc, dict) else None
        if isinstance(e, dict):
            return ' — '.join(str(e[k]) for k in ('code', 'message') if e.get(k))
        return 'no details'

    def _types(self) -> dict:
        status, doc = self.request('GET', '/instance-types')
        if status != 200 or not isinstance((doc or {}).get('data'), dict):
            raise self.error('reading the instance types', status, doc)
        return doc['data']

    def prices(self) -> list:
        out = []
        for name, item in sorted(self._types().items()):
            it = item.get('instance_type') or {}
            cents = it.get('price_cents_per_hour')
            out.append({'gpu': it.get('name') or name, 'name': it.get('gpu_description') or it.get('description')
                        or name, 'usd_per_hour': float(cents) / 100.0 if cents is not None else None,
                        'vram_gb': None, 'available': bool(item.get('regions_with_capacity_available')),
                        'regions': [r.get('name') for r in item.get('regions_with_capacity_available') or ()],
                        'spot': False})
        return out

    def launch(self, name: str, gpu: str, public_key: str, image: str) -> dict:
        types = self._types()
        item = types.get(gpu) or {}
        regions = [r.get('name') for r in item.get('regions_with_capacity_available') or () if r.get('name')]
        if not regions:
            raise CloudError(f'lambda: no region has capacity for {gpu} right now')
        status, doc = self.request('POST', '/ssh-keys', {'name': name, 'public_key': public_key})
        key = (doc or {}).get('data') if isinstance(doc, dict) else None
        if status not in (200, 201) or not isinstance(key, dict):
            raise self.error('adding the burst\'s SSH key', status, doc)
        body = {'region_name': regions[0], 'instance_type_name': gpu, 'ssh_key_names': [name], 'name': name}
        status, doc = self.request('POST', '/instance-operations/launch', body)
        ids = (((doc or {}).get('data') or {}).get('instance_ids') or []) if isinstance(doc, dict) else []
        if status != 200 or not ids:
            self.cleanup({'ssh_key_id': key.get('id')})
            raise self.error('launching the instance', status, doc)
        return {'id': str(ids[0]), 'ssh_key_id': key.get('id'), 'region': regions[0]}

    def status(self, instance_id: str) -> dict:
        status, doc = self.request('GET', f'/instances/{urllib.parse.quote(instance_id)}')
        if status == 404:
            return {'state': 'missing'}
        d = (doc or {}).get('data') if isinstance(doc, dict) else None
        if status != 200 or not isinstance(d, dict):
            raise self.error('reading the instance', status, doc)
        return {'state': STATES.get(str(d.get('status') or ''), 'booting'), 'ip': d.get('ip'), 'ssh_port': 22}

    def terminate(self, instance_id: str):
        status, doc = self.request('POST', '/instance-operations/terminate', {'instance_ids': [instance_id]})
        if status == 200:
            return
        if status == 404:
            return                                   # not found: gone already (status() confirms)
        raise self.error('terminating the instance', status, doc)

    def instances(self) -> list:
        out, token = [], ''
        for _ in range(50):
            q = f'?page_token={urllib.parse.quote(token)}' if token else ''
            status, doc = self.request('GET', '/instances' + q)
            if status != 200 or not isinstance((doc or {}).get('data'), list):
                raise self.error('listing the instances', status, doc)
            for d in doc['data']:
                out.append({'id': str(d.get('id')), 'name': d.get('name') or '',
                            'state': STATES.get(str(d.get('status') or ''), 'booting')})
            token = doc.get('page_token') or ''
            if not token:
                return out
        raise CloudError('lambda: too many pages of instances')

    def cleanup(self, launched: dict):
        kid = (launched or {}).get('ssh_key_id')
        if not kid:
            return
        status, doc = self.request('DELETE', f'/ssh-keys/{urllib.parse.quote(str(kid))}')
        if status not in (200, 204, 404):
            raise self.error('removing the burst\'s SSH key', status, doc)

    def remote_command(self, args: str, image: str) -> str:
        return (f'chmod 777 "$HOME/falcon-burst" && docker run --rm --gpus all '
                f'-v "$HOME/falcon-burst:/home/carla/falcon-burst" -w /home/carla '
                f'{shlex.quote(image)} falcon-worker-daemon {args}')
