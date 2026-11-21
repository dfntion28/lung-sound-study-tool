"""Offline acoustic-finding workflow. No UI imports or import-time file access.

CSV writes are atomic and serialized across Store instances in one app process.
Run one Streamlit server per log root. Trial IDs encode recording and rater codes;
the two public tables retain the exact data-dictionary schemas.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import wraps
import csv
import hashlib
import io
import json
from math import gcd
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from uuid import uuid4

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from .audio_qc import inspect_audio
from .locked_eval import FROZEN, load_frozen, output_label, vote, window_probabilities
from .windowing import window_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / 'app/protocol.json'
LABELS = {'N': 'Normal', 'C': 'Crackle', 'W': 'Wheeze', 'B': 'Both',
          'U': 'Uncertain', 'Q': 'QC-fail'}
MANUAL_LABELS = {'N': 'Normal', 'C': 'Crackle', 'W': 'Wheeze', 'B': 'Both', 'U': 'Unsure'}
PREDICTION_COLUMNS = ['prediction_id', 'timestamp', 'recording_id', 'audio_sha256',
                      'model_version', 'model_sha256', 'preprocess_version', 'qc_status',
                      'qc_reason', 'window_outputs', 'finding', 'top_score', 'threshold',
                      'operator_code', 'notes']
RATER_COLUMNS = ['trial_id', 'case_code', 'site_code', 'rater_code', 'manual_label',
                 'manual_locked_at', 'ai_prediction_id', 'ai_finding', 'assisted_label',
                 'assisted_at', 'expert_reference', 'protocol_deviation']
SCHEMAS = {'predictions': PREDICTION_COLUMNS, 'rater_study': RATER_COLUMNS}
PREPROCESS_VERSION = 'mono4000_logmel_3s_halfhop_vote_v1'
NOT_ENOUGH = 'not enough data'
UNDEFINED = 'undefined (no label variation)'
_LOCKS = defaultdict(threading.RLock)
_LOCKS_GUARD = threading.Lock()


def protocol(path=PROTOCOL_PATH):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decodable(path):
    """Return a path soundfile can read: the file itself, or an ffmpeg PCM WAV copy beside it.

    Phone apps often export AAC/M4A, which libsndfile cannot decode (R1, Q-001).
    Without ffmpeg, or if conversion fails, the original path is returned and QC reports it unreadable.
    """
    path = Path(path)
    try:
        sf.info(path)
        return path
    except (RuntimeError, sf.LibsndfileError):
        pass
    target = path.with_name(path.name + '.decoded.wav')
    if shutil.which('ffmpeg') and not target.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(path), '-c:a', 'pcm_s16le', str(target)],
                       capture_output=True)
    return target if target.exists() else path


def check_audio(path, *, protocol_path=PROTOCOL_PATH):
    """QC only: safe to show before a nurse answer, without running the model.

    Existing QC detects duration, clipping and silence; it does not establish
    that speech or dominant noise is absent. Device export compatibility remains
    unproven until a real Hanhong export is tested.
    """
    path = Path(path)
    minimum = float(protocol(protocol_path)['min_duration_s'])
    try:
        metadata = inspect_audio(path)
    except (OSError, RuntimeError, ValueError):
        try:
            sha = _hash(path)
        except OSError:
            sha = ''
        return dict(audio_sha256=sha, duration_s=0., qc_status='fail',
                    qc_reason='unreadable_audio')
    reasons = list(filter(None, metadata['qc_reason'].split(';')))
    if metadata['duration_s'] < minimum:
        reasons.append(f'protocol_too_short: minimum {minimum:g} s')
    # Detect invalid samples and cancellation after the same mono average used
    # by inference; per-channel ingestion QC alone cannot catch anti-phase audio.
    try:
        channels, _ = sf.read(path, dtype='float32', always_2d=True)
        mono = channels.mean(axis=1)
        if not np.isfinite(channels).all():
            reasons.append('nonfinite_audio')
        elif mono.size and np.sqrt(np.mean(mono.astype('float64') ** 2)) <= 1e-4:
            if 'silence' not in reasons:
                reasons.append('mono_silence')
    except (OSError, RuntimeError, ValueError):
        reasons.append('unreadable_audio')
    return dict(audio_sha256=metadata['sha256'], duration_s=metadata['duration_s'],
                qc_status='fail' if reasons else 'pass', qc_reason=';'.join(reasons))


def analyze(path, model):
    """Return an acoustic finding with the frozen Phase 5 inference recipe."""
    result = dict(check_audio(path), window_outputs=[], finding='Q', top_score=None,
                  threshold=FROZEN['threshold'], relisten=False,
                  **{f'v{i}': 0. for i in range(4)})
    if result['qc_status'] != 'pass':
        return result
    channels, sr = sf.read(path, dtype='float32', always_2d=True)
    signal = channels.mean(axis=1)
    if sr != 4000:
        divisor = gcd(sr, 4000)
        signal = resample_poly(signal, 4000 // divisor, sr // divisor)
    signal = signal.astype(np.float32)
    bounds = window_bounds(len(signal), FROZEN['window'])
    probabilities = window_probabilities(model, signal, bounds)
    fractions = vote(probabilities)
    finding = output_label(fractions)
    result.update(window_outputs=[dict(start_s=start / 4000, end_s=end / 4000,
                                       probabilities=list(map(float, values)))
                                  for (start, end), values in zip(bounds, probabilities)],
                  **{f'v{i}': float(v) for i, v in enumerate(fractions)},
                  finding=finding, top_score=float(max(fractions)),
                  relisten=finding in ('C', 'B'))
    return result


def _code(value):
    value = str(value)
    if not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', value):
        raise ValueError('Use uppercase alphanumeric codes, starting with a letter (no names).')
    return value


def _label(value, *, expert=False):
    aliases = {v: k for k, v in MANUAL_LABELS.items()}
    aliases.update(Uncertain='U', **{'QC-fail': 'Q'})
    label = aliases.get(value, value)
    if label not in (LABELS if expert else MANUAL_LABELS):
        raise ValueError('Invalid acoustic label')
    return label


def _now():
    return datetime.now(timezone.utc).isoformat()


def _serialized(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return run


def cohen_kappa(left, right):
    """Paired categorical kappa, including abstentions as their own category."""
    left, right = list(left), list(right)
    if len(left) != len(right):
        raise ValueError('Paired labels must have equal lengths')
    n = len(left)
    if n < 2:
        return NOT_ENOUGH
    a, b = Counter(left), Counter(right)
    expected = sum(a[c] * b[c] for c in a.keys() | b.keys()) / n ** 2
    observed = sum(x == y for x, y in zip(left, right)) / n
    return (observed - expected) / (1 - expected) if expected < 1 else UNDEFINED


def _percent(left, right):
    return (100 * sum(a == b for a, b in zip(left, right)) / len(left)
            if len(left) >= 2 else NOT_ENOUGH)


class Store:
    """Append-only manual answers and recording identities; configurable CSV root.

    Canonical WAV uploads are byte copies. Other libsndfile-readable formats are
    decoded to lossless float WAV without changing rate or channel layout. The
    prediction hash always identifies this stored, replayable WAV.
    """

    def __init__(self, root=PROJECT_ROOT / 'logs/app', *, protocol_path=PROTOCOL_PATH,
                 model_version='cnn_3s_vote_v1', model_sha256=FROZEN['model_sha256']):
        self.root = Path(root).resolve()
        self.protocol = protocol(protocol_path)
        self.model_version, self.model_sha256 = model_version, model_sha256
        with _LOCKS_GUARD:
            self._lock = _LOCKS[str(self.root)]
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / 'recordings').mkdir(exist_ok=True)
            for name in SCHEMAS:
                if not (self.root / f'{name}.csv').exists():
                    self._write(name, [])
                else:
                    self.table(name)  # Refuse incompatible CSVs rather than overwriting.

    def _write(self, name, rows):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=self.root, suffix='.csv', delete=False) as f:
            pending = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=SCHEMAS[name])
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.replace(pending, self.root / f'{name}.csv')
        finally:
            pending.unlink(missing_ok=True)

    @_serialized
    def table(self, name):
        if name not in SCHEMAS:
            raise ValueError('Unknown table')
        with (self.root / f'{name}.csv').open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != SCHEMAS[name]:
                raise ValueError(f'CSV schema mismatch: {name}')
            return list(reader)

    def _trial(self, trial_id):
        rows = self.table('rater_study')
        for row in rows:
            if row['trial_id'] == trial_id:
                return row
        raise ValueError('Unknown trial')

    def _update_trial(self, changed):
        self._write('rater_study', [changed if r['trial_id'] == changed['trial_id'] else r
                                    for r in self.table('rater_study')])

    @staticmethod
    def recording_id(trial_id):
        rid, sep, rater = str(trial_id).partition('__')
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}_[A-Z][A-Z0-9]{0,15}_T\d{2,}_(BELL|DIA)', rid):
            raise ValueError('Invalid trial ID')
        _code(rater)
        return rid

    def recording_path(self, trial_id):
        return self.root / 'recordings' / f'{self.recording_id(trial_id)}.wav'

    @staticmethod
    def trial_number(trial_id):
        return int(Store.recording_id(trial_id).split('_')[2][1:])

    @staticmethod
    def base_trial(number):
        """Protocol trial of a stored number. A retake of trial t is stored as t*100+k (T101 retakes T01)."""
        return number // 100 if number >= 100 else number

    @_serialized
    def create_trial(self, path, case, site, trial, mode, rater, *, unique=False):
        case, site, rater = _code(case), _code(site), _code(rater)
        if site not in self.protocol['site_codes'] or mode not in self.protocol['modes']:
            raise ValueError('Site/mode must be in the provisional protocol')
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
            raise ValueError('Trial number must be a positive integer')
        rid = f'{case}_{site}_T{trial:02d}_{mode}'
        tid = f'{rid}__{rater}'
        destination = self.recording_path(tid)
        source = Path(path)
        # Only decode valid audio; the UI reports unreadable files before here.
        values, sr = sf.read(source, dtype='float32', always_2d=True)
        if not np.isfinite(values).all():
            raise ValueError('Recording contains nonfinite samples')
        if sf.info(source).format in ('WAV', 'WAVEX', 'RF64'):
            raw = source.read_bytes()
        else:
            stream = io.BytesIO()
            sf.write(stream, values, sr, format='WAV', subtype='FLOAT')
            raw = stream.getvalue()
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError('Recording exists; cannot overwrite. Use a new trial number.')
        else:
            # The app passes unique=True: one file must not stand for two sites or trials (fake repeatability).
            for other in (sorted((self.root / 'recordings').glob(f'{case}_*.wav')) if unique else ()):
                if other.read_bytes() == raw:
                    raise ValueError(f'This file is already saved as {other.stem}. Attach the recording made here.')
            destination.write_bytes(raw)
        rows = self.table('rater_study')
        if not any(r['trial_id'] == tid for r in rows):
            row = dict.fromkeys(RATER_COLUMNS, '')
            row.update(trial_id=tid, case_code=case, site_code=site, rater_code=rater)
            self._write('rater_study', [*rows, row])
        return tid

    @_serialized
    def lock_manual(self, trial_id, label):
        label = _label(label)
        row = self._trial(trial_id)
        if row['manual_locked_at']:
            if row['manual_label'] != label:
                raise ValueError('Manual label is locked and cannot change')
            return row
        row.update(manual_label=label, manual_locked_at=_now())
        self._update_trial(row)
        return row

    def _locked(self, trial_id):
        row = self._trial(trial_id)
        if not row['manual_label'] or not row['manual_locked_at']:
            raise ValueError('Lock a manual label before AI is shown or recorded')
        return row

    def _decode_prediction(self, row, trial_id):
        if row['audio_sha256'] != _hash(self.recording_path(trial_id)):
            raise ValueError('Audio hash mismatch: stored recording changed')
        result = dict(row)
        result['window_outputs'] = json.loads(result['window_outputs'])
        result['top_score'] = float(result['top_score']) if result['top_score'] else None
        result['threshold'] = float(result['threshold'])
        result['relisten'] = result['finding'] in ('C', 'B')
        fractions = (vote([w['probabilities'] for w in result['window_outputs']])
                     if result['window_outputs'] else np.zeros(4))
        result.update({f'v{i}': float(v) for i, v in enumerate(fractions)})
        result['duration_s'] = sf.info(self.recording_path(trial_id)).duration
        return result

    @_serialized
    def get_ai(self, trial_id):
        rater = self._locked(trial_id)
        if not rater['ai_prediction_id']:
            return None
        for row in self.table('predictions'):
            if row['prediction_id'] == rater['ai_prediction_id']:
                return self._decode_prediction(row, trial_id)
        raise ValueError('Missing linked AI prediction')

    @_serialized
    def record_ai(self, trial_id, result):
        rater = self._locked(trial_id)
        if result['audio_sha256'] != _hash(self.recording_path(trial_id)):
            raise ValueError('Audio hash mismatch')
        if result['finding'] not in LABELS or (result['qc_status'] != 'pass' and result['finding'] != 'Q'):
            raise ValueError('Invalid finding/QC combination')
        if result['threshold'] != FROZEN['threshold']:
            raise ValueError('Frozen threshold mismatch')
        rid = self.recording_id(trial_id)
        rows = self.table('predictions')
        same = [r for r in rows if r['recording_id'] == rid]
        if same:
            row = same[0]
            if row['finding'] != result['finding'] or row['audio_sha256'] != result['audio_sha256']:
                raise ValueError('AI prediction already exists for this recording')
        else:
            row = dict(prediction_id=uuid4().hex, timestamp=_now(), recording_id=rid,
                       audio_sha256=result['audio_sha256'], model_version=self.model_version,
                       model_sha256=self.model_sha256, preprocess_version=PREPROCESS_VERSION,
                       qc_status=result['qc_status'], qc_reason=result['qc_reason'],
                       window_outputs=json.dumps(result['window_outputs'], allow_nan=False),
                       finding=result['finding'], top_score='' if result['top_score'] is None else result['top_score'],
                       threshold=result['threshold'], operator_code=rater['rater_code'], notes='')
            self._write('predictions', [*rows, row])
        rater.update(ai_prediction_id=row['prediction_id'], ai_finding=row['finding'])
        self._update_trial(rater)
        return self._decode_prediction(row, trial_id)

    @_serialized
    def analyze_trial(self, trial_id, model):
        self._locked(trial_id)  # This guard precedes inference, cache access and display.
        current = self.get_ai(trial_id)
        if current is not None:
            return current
        rid = self.recording_id(trial_id)
        same = [r for r in self.table('predictions') if r['recording_id'] == rid]
        if same:
            result = self._decode_prediction(same[0], trial_id)
        else:
            result = analyze(self.recording_path(trial_id), model)
        return self.record_ai(trial_id, result)

    @_serialized
    def set_assisted(self, trial_id, label):
        row = self._locked(trial_id)
        if not row['ai_prediction_id']:
            raise ValueError('View the AI finding before recording an assisted answer')
        label = _label(label)
        if row['assisted_label']:  # A signature is final: repeating it changes nothing, changing it is refused.
            if row['assisted_label'] != label:
                raise ValueError('Final finding is already signed and cannot change')
            return
        row.update(assisted_label=label, assisted_at=_now())
        self._update_trial(row)

    @_serialized
    def set_expert(self, trial_id, label):
        row = self._trial(trial_id)
        row['expert_reference'] = _label(label, expert=True)
        self._update_trial(row)

    @_serialized
    def review_queue(self):
        return [r for r in self.table('rater_study') if r['ai_prediction_id'] and
                (r['manual_label'] != r['ai_finding'] or r['manual_label'] == 'U' or r['ai_finding'] == 'U')]

    @_serialized
    def previous_recording(self, case, site, before_trial):
        case, site = _code(case), _code(site)
        candidates = []
        for path in (self.root / 'recordings').glob(f'{case}_{site}_T*_*.wav'):
            match = re.fullmatch(rf'{case}_{site}_T(\d+)_(BELL|DIA)\.wav', path.name)
            if match and int(match[1]) < before_trial:
                candidates.append((int(match[1]), path.name, path))
        return max(candidates)[2] if candidates else None

    @_serialized
    def analytics(self):
        all_rows = self.table('rater_study')
        rows = [r for r in all_rows if r['ai_prediction_id']]
        # Manual Unsure and AI Uncertain are different judgments, even though the
        # stored code U is shared. Keep them distinct in agreement and kappa.
        manual = ['manual_unsure' if r['manual_label'] == 'U' else r['manual_label'] for r in rows]
        ai = [r['ai_finding'] for r in rows]
        metrics = dict(n_manual_ai=len(rows), agreement_pct=_percent(manual, ai),
                       cohen_kappa_manual_ai=cohen_kappa(manual, ai))
        for field, name in [('ai_finding', 'ai'), ('manual_label', 'manual'), ('assisted_label', 'assisted')]:
            eligible = [r for r in all_rows if r['expert_reference'] and r[field]]
            left = [f'{name}_unsure' if name != 'ai' and r[field] == 'U' else r[field] for r in eligible]
            right = [r['expert_reference'] for r in eligible]
            metrics[f'n_{name}_expert'] = len(eligible)
            metrics[f'accuracy_{name}_expert_pct'] = _percent(left, right)
            if name == 'assisted':
                metrics['cohen_kappa_assisted_expert'] = cohen_kappa(left, right)
        complete = [r for r in rows if r['expert_reference'] and r['assisted_label']]
        metrics['right_to_wrong_count'] = (sum(r['manual_label'] == r['expert_reference'] and
                                               r['manual_label'] != 'U' and
                                               (r['assisted_label'] != r['expert_reference'] or r['assisted_label'] == 'U')
                                               for r in complete) if len(complete) >= 2 else NOT_ENOUGH)
        metrics['repeatability'] = self._repeatability(rows)
        return metrics

    def _repeatability(self, rows):
        groups = defaultdict(dict)
        latest = {}  # A retake replaces its original, so the pair is not a repeat.
        for row in rows:
            case, site, attempt, mode = self.recording_id(row['trial_id']).split('_')
            number = int(attempt[1:])
            key = (case, site, mode, self.base_trial(number))
            latest[key] = max(latest.get(key, 0), number)
        for row in rows:
            rid = self.recording_id(row['trial_id'])
            case, site, attempt, mode = rid.split('_')
            number = int(attempt[1:])
            if number != latest[(case, site, mode, self.base_trial(number))]:
                continue
            groups[(row['case_code'], row['site_code'])][rid] = row['ai_finding']
        repeats = [list(g.values()) for g in groups.values() if len(g) >= 2]
        output = dict(n_groups=len(repeats), n_recordings=sum(map(len, repeats)),
                      self_agreement_pct=NOT_ENOUGH, fleiss_kappa=NOT_ENOUGH,
                      method='Self-agreement pools all within-case/site recording pairs. '
                             'Fleiss kappa averages within-group agreement with pooled label marginals; '
                             'unequal repeat counts use this descriptive extension. At least two groups '
                             'are needed for kappa. N/C/W/B/U/Q are separate AI categories.')
        if not repeats:
            return output
        pairs = sum(len(g) * (len(g) - 1) for g in repeats)
        matching = sum(sum(n * (n - 1) for n in Counter(g).values()) for g in repeats)
        output['self_agreement_pct'] = 100 * matching / pairs
        if len(repeats) < 2:
            return output
        pooled = Counter(v for g in repeats for v in g)
        total = sum(pooled.values())
        expected = sum((n / total) ** 2 for n in pooled.values())
        observed = np.mean([sum(n * (n - 1) for n in Counter(g).values()) /
                            (len(g) * (len(g) - 1)) for g in repeats])
        output['fleiss_kappa'] = ((float(observed) - expected) / (1 - expected)
                                  if expected < 1 else UNDEFINED)
        return output

    @_serialized
    def export_csv(self, table):
        """UTF-8 CSV download, including the derived disagreement table."""
        if table == 'review_queue':
            rows, columns = self.review_queue(), RATER_COLUMNS
        elif table in SCHEMAS:
            rows, columns = self.table(table), SCHEMAS[table]
        else:
            raise ValueError('Unknown table')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode('utf-8-sig')
