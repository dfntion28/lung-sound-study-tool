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
from types import MappingProxyType
from uuid import uuid4

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from .audio_qc import inspect_audio
from .cnn_v2 import load_v2, predict_signal
from .locked_eval import vote
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
# Adopted model (D-114): frozen v2 under D-113. Constants are checked against config.json at load time.
MODEL = MappingProxyType(dict(
    version='cnn_3s_vote_v2', directory=PROJECT_ROOT / 'models/cnn_3s_vote_v2',
    sha256='943db6f8c1c9967f1434aba8e1b67fbe8f9b73d21ea627f341ad276cb9c07bc6', threshold=.62, window=3))
PREPROCESS_VERSION = 'mono4000_logmel_looppad_peakfloor80_3s_halfhop_vote_v2'
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


def load_model(directory=None):
    """Load the adopted frozen model; refuse any file or config that is not the D-113 freeze."""
    model, config = load_v2(directory or MODEL['directory'])
    found = (config['model_version'], config['model_sha256'], config['abstain_threshold'], config['window_s'])
    if found != (MODEL['version'], MODEL['sha256'], MODEL['threshold'], MODEL['window']):
        raise ValueError('Model config does not match the adopted frozen model')
    return model


def analyze(path, model, progress=None):
    """Return an acoustic finding with the frozen v2 inference recipe (cnn_v2.predict_signal, D-113/D-114).

    `progress(stage, data)` is told what each stage produced, for display only (D-111);
    it never changes the computation.
    """
    report = progress or (lambda stage, data: None)
    result = dict(check_audio(path), window_outputs=[], finding='Q', top_score=None,
                  threshold=MODEL['threshold'], relisten=False,
                  **{f'v{i}': 0. for i in range(4)})
    report('quality', dict(result))
    if result['qc_status'] != 'pass':
        return result
    channels, sr = sf.read(path, dtype='float32', always_2d=True)
    signal = channels.mean(axis=1)
    if sr != 4000:
        divisor = gcd(sr, 4000)
        signal = resample_poly(signal, 4000 // divisor, sr // divisor)
    signal = signal.astype(np.float32)
    bounds = window_bounds(len(signal), MODEL['window'])
    report('prepared', dict(rate=sr, duration_s=len(signal) / 4000, segments=len(bounds)))
    prediction = predict_signal(model, signal, MODEL['threshold'])
    assert prediction['bounds'] == bounds
    probabilities, fractions, finding = prediction['window_probabilities'], prediction['vote'], prediction['label']
    result.update(window_outputs=[dict(start_s=start / 4000, end_s=end / 4000,
                                       probabilities=list(map(float, values)))
                                  for (start, end), values in zip(bounds, probabilities)],
                  **{f'v{i}': float(v) for i, v in enumerate(fractions)},
                  finding=finding, top_score=float(max(fractions)),
                  relisten=finding in ('C', 'B'))
    return result


# Listen-again sites (D-119): 2 front and 2 back sites per case, examiner and mode, drawn by a fixed seeded order
# before anything is recorded, so the repeats never depend on a finding, a model version or a code rule change.
REPEAT_SIDES = (('ARU', 'ALU', 'ARL', 'ALL'), ('PRU', 'PLU', 'PRL', 'PLL'))
REPEATS_PER_SIDE = 2


def repeat_sites(case, examiner, mode):
    """The sites that get a second recording, in recording order. sha256, not hash(), so it is stable across runs."""
    chosen = set()
    for codes in REPEAT_SIDES:
        ranked = sorted(codes, key=lambda code: hashlib.sha256(f'{case}|{examiner}|{mode}|{code}'.encode()).hexdigest())
        chosen.update(ranked[:REPEATS_PER_SIDE])
    return tuple(code for codes in REPEAT_SIDES for code in codes if code in chosen)


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
                 model_version=MODEL['version'], model_sha256=MODEL['sha256']):
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

    def stored_name(self, trial_id, rows=None):
        """File stem of a trial's recording. New files carry the examiner, so two examiners can record the
        same case, site, trial and mode (D-109). A trial saved before that keeps its old name."""
        rid = self.recording_id(trial_id)
        folder = self.root / 'recordings'
        if not (folder / f'{trial_id}.wav').exists() and (folder / f'{rid}.wav').exists():
            rows = self.table('rater_study') if rows is None else rows
            if any(r['trial_id'] == trial_id for r in rows):
                return rid
        return trial_id

    def recording_path(self, trial_id, rows=None):
        return self.root / 'recordings' / f'{self.stored_name(trial_id, rows)}.wav'

    def _pushed(self, file_name):
        """True when the durable copy already holds this recording. Local storage has none."""
        return False

    def _taken(self, stem):
        return (self.root / 'recordings' / f'{stem}.wav').exists() or self._pushed(f'{stem}.wav')

    def _landing(self, tid, raw, rows):
        """Trial ID for a new record. A name left by a save whose record failed is reused when it holds the
        same file or never reached durable storage; otherwise the next free retake number is used, because
        durable storage never replaces a recording (D-109)."""
        if not self._taken(tid):
            return tid
        left = self.root / 'recordings' / f'{tid}.wav'
        if left.exists() and left.read_bytes() == raw:
            return tid
        if left.exists() and not self._pushed(left.name):
            left.unlink()
            return tid
        case, site, attempt, mode = self.recording_id(tid).split('_')
        rater = tid.partition('__')[2]
        known = {r['trial_id'] for r in rows}
        base = self.base_trial(int(attempt[1:]))
        for number in range(base * 100 + 1, base * 100 + 100):
            candidate = f'{case}_{site}_T{number:02d}_{mode}__{rater}'
            if candidate not in known and not self._taken(candidate):
                return candidate
        raise ValueError('No free trial number for this site')

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
        tid = f'{case}_{site}_T{trial:02d}_{mode}__{rater}'
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
        rows = self.table('rater_study')
        known = any(r['trial_id'] == tid for r in rows)
        if not known:
            tid = self._landing(tid, raw, rows)
        destination = self.recording_path(tid, rows)
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError('Recording exists; cannot overwrite. Use a new trial number.')
        if unique and not known:
            # One file must not stand for two sites or trials (fake repeatability). The same file for the same
            # site, trial and mode by another examiner is allowed: two examiners grading one recording.
            here = (site, self.base_trial(self.trial_number(tid)), mode)
            for other in rows:
                info = self.recording_id(other['trial_id']).split('_')
                same_place = (info[1], self.base_trial(int(info[2][1:])), info[3]) == here
                if other['case_code'] != case or (same_place and other['rater_code'] != rater):
                    continue
                path = self.recording_path(other['trial_id'], rows)
                if path.exists() and path.read_bytes() == raw:
                    raise ValueError(f"This file is already saved as {self.recording_id(other['trial_id'])}. "
                                     'Attach the recording made here.')
        created = not destination.exists()
        if created:
            destination.write_bytes(raw)
        if not known:
            row = dict.fromkeys(RATER_COLUMNS, '')
            row.update(trial_id=tid, case_code=case, site_code=site, rater_code=rater)
            try:
                self._write('rater_study', [*rows, row])
            except BaseException:
                # Leave no recording without a record, unless it already reached durable storage.
                if created and not self._pushed(destination.name):
                    destination.unlink(missing_ok=True)
                raise
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
        if result['threshold'] != MODEL['threshold']:
            raise ValueError('Frozen threshold mismatch')
        rid = self.stored_name(trial_id)  # The stored file, so each examiner's recording has its own analysis.
        rows = self.table('predictions')
        # A stored analysis keeps its own model tag (D-114); only the same model's row is reused.
        same = [r for r in rows if r['recording_id'] == rid and r['model_version'] == self.model_version]
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
    def analyze_trial(self, trial_id, model, progress=None):
        self._locked(trial_id)  # This guard precedes inference, cache access and display.
        report = progress or (lambda stage, data: None)
        current = self.get_ai(trial_id)
        if current is not None:
            return current
        rid = self.stored_name(trial_id)
        same = [r for r in self.table('predictions') if r['recording_id'] == rid]
        if same:
            result = self._decode_prediction(same[0], trial_id)
            report('stored', dict(result))
        else:
            result = analyze(self.recording_path(trial_id), model, progress)
            report('analyzed', dict(result))
        report('saving', None)
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
        rows = self.table('rater_study')
        return [r for r in rows if (r['ai_prediction_id'] and
                (r['manual_label'] != r['ai_finding'] or r['manual_label'] == 'U' or r['ai_finding'] == 'U'))
                or self._repeat_differs(r, rows)]

    def _repeat_differs(self, row, rows):
        """A second recording whose signed finding differs from the signed first recording at that site (D-119)."""
        if not row['assisted_label']:
            return False
        case, site, attempt, mode = self.recording_id(row['trial_id']).split('_')
        if self.base_trial(int(attempt[1:])) < 2:
            return False
        firsts = [(int(a[1:]), r) for r in rows for c, s, a, m in [self.recording_id(r['trial_id']).split('_')]
                  if (c, s, m, r['rater_code']) == (case, site, mode, row['rater_code']) and self.base_trial(int(a[1:])) == 1]
        first = max(firsts, key=lambda pair: pair[0])[1] if firsts else None
        return bool(first and first['assisted_label'] and first['assisted_label'] != row['assisted_label'])

    @_serialized
    def previous_recording(self, case, site, before_trial, rater=None):
        case, site = _code(case), _code(site)
        candidates = []
        for path in (self.root / 'recordings').glob(f'{case}_{site}_T*_*.wav'):
            match = re.fullmatch(rf'{case}_{site}_T(\d+)_(BELL|DIA)(?:__([A-Z][A-Z0-9]{{0,15}}))?\.wav', path.name)
            if match and int(match[1]) < before_trial and (rater is None or match[3] in (None, rater)):
                candidates.append((int(match[1]), path.name, path))
        return max(candidates)[2] if candidates else None

    @_serialized
    def analytics(self):
        # AI-dependent numbers use only analyses made by this store's model version (D-117), so first-version
        # and second-version answers are never pooled. Manual-against-expert accuracy uses every trial.
        version = {r['prediction_id']: r['model_version'] for r in self.table('predictions')}
        every_row = self.table('rater_study')
        current = [r for r in every_row if not r['ai_prediction_id'] or version.get(r['ai_prediction_id']) == self.model_version]
        all_rows = current
        rows = [r for r in current if r['ai_prediction_id']]
        # Manual Unsure and AI Uncertain are different judgments, even though the
        # stored code U is shared. Keep them distinct in agreement and kappa.
        manual = ['manual_unsure' if r['manual_label'] == 'U' else r['manual_label'] for r in rows]
        ai = [r['ai_finding'] for r in rows]
        metrics = dict(n_manual_ai=len(rows), agreement_pct=_percent(manual, ai),
                       cohen_kappa_manual_ai=cohen_kappa(manual, ai))
        for field, name in [('ai_finding', 'ai'), ('manual_label', 'manual'), ('assisted_label', 'assisted')]:
            pool = every_row if name == 'manual' else all_rows
            eligible = [r for r in pool if r['expert_reference'] and r[field]]
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
        metrics['model_version'] = self.model_version
        metrics['n_other_model_excluded'] = len(every_row) - len(current)
        return metrics

    def _repeatability(self, rows):
        groups = defaultdict(dict)
        latest = {}  # A retake replaces its original, so the pair is not a repeat.
        for row in rows:
            case, site, attempt, mode = self.recording_id(row['trial_id']).split('_')
            number = int(attempt[1:])
            key = (case, site, mode, self.base_trial(number), row['rater_code'])
            latest[key] = max(latest.get(key, 0), number)
        for row in rows:
            rid = self.recording_id(row['trial_id'])
            case, site, attempt, mode = rid.split('_')
            number = int(attempt[1:])
            if number != latest[(case, site, mode, self.base_trial(number), row['rater_code'])]:
                continue
            # Repeats are one examiner's recordings of one site; other examiners' files are not repeats.
            groups[(row['case_code'], row['site_code'], row['rater_code'])][rid] = row['ai_finding']
        repeats = [list(g.values()) for g in groups.values() if len(g) >= 2]
        output = dict(n_groups=len(repeats), n_recordings=sum(map(len, repeats)),
                      self_agreement_pct=NOT_ENOUGH, fleiss_kappa=NOT_ENOUGH,
                      method='Self-agreement pools all within-case/site recording pairs of one examiner. '
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
