"""Google Drive persistence for the hosted team tool (D-068).

The host's disk is wiped on restart, so Drive is the durable copy. Every change is
written to Drive first; if that fails, the local record is not changed.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import time

import requests

from .app_core import SCHEMAS, Store

TIMEOUT_S = 120
RETRIES = 3


class DriveError(ValueError):
    """Drive could not confirm the operation. Shown to the user as a save failure."""


class DriveClient:
    """Client for the Apps Script endpoint in deploy/apps_script/Code.gs."""

    def __init__(self, url, key, session=None):
        self.url, self.key = url, key
        self.session = session or requests.Session()

    def _call(self, action, **fields):
        last = 'no response'
        for attempt in range(RETRIES):
            if attempt:
                time.sleep(2 ** attempt)
            try:
                response = self.session.post(self.url, json=dict(key=self.key, action=action, **fields),
                                             timeout=TIMEOUT_S)
                response.raise_for_status()
                body = response.json()
            except (requests.RequestException, ValueError) as error:
                # The exception text can include the request; never surface it.
                last = type(error).__name__
                continue
            if body.get('ok'):
                return body
            raise DriveError(f"Google Drive refused the request: {body.get('error', 'unknown error')}")
        raise DriveError(f'Google Drive could not be reached ({last}). Check the connection and try again.')

    def claim(self):
        return self._call('claim')

    def list(self):
        return self._call('list')['files']

    def get(self, name, expected_sha=None):
        data = base64.b64decode(self._call('get', name=name)['b64'])
        if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
            raise DriveError(f'Checksum mismatch when reading {name} from Google Drive.')
        return data

    def put(self, name, data):
        sha = hashlib.sha256(data).hexdigest()
        body = self._call('put', name=name, sha256=sha, b64=base64.b64encode(data).decode('ascii'))
        if body.get('sha256') != sha:
            raise DriveError(f'Google Drive did not confirm {name}.')


class DriveState:
    """One restored working copy per server process."""

    def __init__(self, client, root=None):
        self.client = client
        self.root = Path(root or tempfile.mkdtemp(prefix='stethoai_')).resolve()
        self.pushed = set()
        self.guard = threading.Lock()
        (self.root / 'recordings').mkdir(parents=True, exist_ok=True)
        self.restore()

    def restore(self):
        """Fail if anything cannot be restored, so empty tables never overwrite Drive."""
        files = self.client.list()
        def fetch(item):
            name = item['name']
            target = self.root / Path(*name.split('/'))
            target.write_bytes(self.client.get(name, item.get('sha256') or None))
            return name
        with ThreadPoolExecutor(max_workers=4) as pool:
            names = list(pool.map(fetch, files))
        self.pushed.update(n for n in names if n.startswith('recordings/'))


class DriveStore(Store):
    """Store whose CSV writes and new recordings reach Drive before they count as saved."""

    def __init__(self, state, **kwargs):
        self.state = state
        super().__init__(state.root, **kwargs)

    def _pushed(self, file_name):
        with self.state.guard:
            return f'recordings/{file_name}' in self.state.pushed

    def _push_recordings(self):
        for path in sorted((self.root / 'recordings').glob('*.wav')):
            name = f'recordings/{path.name}'
            with self.state.guard:
                if name in self.state.pushed:
                    continue
            self.state.client.put(name, path.read_bytes())
            with self.state.guard:
                self.state.pushed.add(name)

    def _write(self, name, rows):
        self._push_recordings()  # A record must never point at a recording Drive lacks.
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=SCHEMAS[name])
        writer.writeheader()
        writer.writerows(rows)
        self.state.client.put(f'{name}.csv', stream.getvalue().encode('utf-8'))
        super()._write(name, rows)
