"""D-046 locked evaluation. Importing this module performs no file I/O.

All input authorities are injectable for synthetic tests. No training, fitting,
feature cache, or threshold selection occurs here. Run only after authorization
to open the locked sets; implementation tests must use synthetic fixtures.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import zlib
from math import gcd
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf
import torch

from . import evaluation, fraiwan, hf_lung, icbhi
from .cnn import CNN_CONFIG, SmallCNN, logmel_input, seed_everything
from .features import FEATURE_CONFIG
from .feature_store import LABEL_RULE_VERSION
from .training_data import CLASSES, TrainingAudio, adult_training_rows, sha256_file
from .windowing import label_interval, window_bounds

FROZEN = MappingProxyType(dict(
    model_path=Path('models/cnn_3s_vote_v1/model.pt'),
    model_sha256='bc58005e125291f862cc357ee45bb492cdaeea405a666c687f67581c65578c64',
    config_path=Path('models/cnn_3s_vote_v1/config.json'),
    manifest_path=Path('data/processed/manifest.csv'),
    manifest_sha256='7b120474bcc9ad099d6997db4f78f56b10c9ab0296347566a6656cf738be7650',
    threshold=.66, window=3, aggregation='vote', classes=CLASSES))
ADMITTED = frozenset({('icbhi2017', 'test'), ('hf_lung_v1', 'test'),
                      ('fraiwan2021', 'external')})
DATASETS = ('icbhi2017', 'hf_lung_v1', 'fraiwan2021')
PERTURBATIONS = ('offset_0.5', 'offset_1.0', 'gain_0.5', 'gain_2.0', 'noise_20db', 'noise_10db')
IDENTITY_COLUMNS = ['dataset', 'recording_key', 'patient_key', 'age_group', 'view']
VOTE_COLUMNS = [f'v{i}' for i in range(4)]
RECORDING_COLUMNS = [*IDENTITY_COLUMNS, 'mode', 'perturbation', 'truth', 'n_windows',
                     *VOTE_COLUMNS, 'pred_t0', 'output_066']
SEGMENT_COLUMNS = [*IDENTITY_COLUMNS, 'segment_id', 'seg_start_s', 'seg_end_s',
                   'truth', 'n_windows', *VOTE_COLUMNS]
EXCLUSION_COLUMNS = ['dataset', 'recording_key', 'age_group', 'view', 'reason']
OUTPUT_FILES = {'recording': ('predictions_recording.csv', RECORDING_COLUMNS),
                'segment': ('predictions_segment.csv', SEGMENT_COLUMNS),
                'exclusions': ('exclusions.csv', EXCLUSION_COLUMNS)}


def _check(condition, message):
    # Explicit exceptions keep the locked guards active under python -O too.
    if not condition:
        raise ValueError(message)


def _hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _checked_bytes(path, expected_sha):
    raw = Path(path).read_bytes()
    _check(_hash_bytes(raw) == expected_sha, f'hash mismatch: {path}')
    return raw


def _config_value(config, names, expected):
    values = [config[name] for name in names if name in config]
    _check(bool(values) and all(value == expected for value in values),
           f'frozen config mismatch: {names[0]}')


def load_frozen(model_path=FROZEN['model_path'], expected_sha=FROZEN['model_sha256'],
                config_path=FROZEN['config_path'], *, window=FROZEN['window'],
                threshold=FROZEN['threshold'], aggregation=FROZEN['aggregation'],
                classes=FROZEN['classes']):
    """Verify the exact bytes before safe deserialization; strict CPU/eval load.

    Both window/window_s and threshold/abstain_threshold are accepted config
    spellings; if both are present both must agree. No missing values default.
    """
    raw = _checked_bytes(model_path, expected_sha)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    _config_value(config, ('window', 'window_s'), window)
    _config_value(config, ('aggregation',), aggregation)
    _config_value(config, ('threshold', 'abstain_threshold'), threshold)
    _config_value(config, ('classes',), list(classes))
    for name in ('features', 'feature_config'):
        if name in config:
            _check(config[name] == FEATURE_CONFIG, 'feature config mismatch')
    if 'label_rule_version' in config:
        _check(config['label_rule_version'] == LABEL_RULE_VERSION, 'label rule mismatch')
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    _check(saved['window'] == window and saved['aggregation'] == aggregation,
           'checkpoint window/aggregation mismatch')
    _check(saved['config'] == CNN_CONFIG, 'checkpoint CNN config mismatch')
    model = SmallCNN()
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model


def scoped_rows(rows):
    mask = [(str(d), str(s)) in ADMITTED for d, s in zip(rows.dataset, rows.split)]
    return rows.loc[mask].copy()


class LockedAudio:
    """Manifest-bound, cache-free reads; scope/path/identity checks precede I/O."""

    def __init__(self, root: Path, manifest: pd.DataFrame):
        self.root = Path(root).resolve()
        admitted = scoped_rows(manifest)
        self.allowed = {self._identity(row) for row in admitted.to_dict('records')}
        # A scoped path cannot alias an excluded/train recording in the authority.
        outside = manifest.loc[~manifest.index.isin(admitted.index)]
        outside_paths = {self._path(p) for p in outside.audio_path}
        _check(not {i[4] for i in self.allowed} & outside_paths, 'locked path aliases out-of-scope audio')

    def _path(self, value):
        path = (self.root / str(value)).resolve()
        _check(path.is_relative_to(self.root), 'audio path escapes project root')
        return path

    def _identity(self, row):
        return (str(row['dataset']), str(row['split']), str(row['recording_id']),
                str(row['patient_id']), self._path(row['audio_path']), str(row['sha256']))

    def load(self, row):
        _check((str(row['dataset']), str(row['split'])) in ADMITTED, 'audio dataset/split not admitted')
        identity = self._identity(row)
        _check(identity in self.allowed, 'audio not in locked manifest authority')
        raw = _checked_bytes(identity[4], str(row['sha256']))
        channels, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
        signal = channels.mean(axis=1)
        if sr != 4000:
            divisor = gcd(sr, 4000)
            signal = resample_poly(signal, 4000 // divisor, sr // divisor)
        signal = signal.astype(np.float32)
        _check(signal.ndim == 1 and signal.size > 0 and np.isfinite(signal).all(), 'invalid locked audio')
        return signal


def source_label(row):
    """Use precisely the original ingest mappings, never generic token guesses."""
    dataset, source = str(row['dataset']), str(row['source_label'])
    if dataset == 'icbhi2017':
        fields = dict(token.strip().split('=', 1) for token in source.split(','))
        return icbhi.LABELS[(fields['crackles'], fields['wheezes'])]
    if dataset == 'hf_lung_v1':
        events = [hf_lung.Event(token, float(row['seg_start_s']), float(row['seg_end_s']))
                  for token in source.split('+') if token and token not in {'I', 'E'}]
        return hf_lung._label(events)
    if dataset == 'fraiwan2021':
        return fraiwan._label(source)
    raise ValueError(f'unsupported source dataset: {dataset}')


def _metadata(group, view):
    first = group.iloc[0]
    return dict(dataset=str(first.dataset), recording_key=f'{first.dataset}::{first.recording_id}',
                patient_key=f'{first.dataset}::{first.patient_id}', age_group=str(first.age_group), view=view)


def _exclusion(group, view, reason):
    return {**{k: v for k, v in _metadata(group, view).items() if k != 'patient_key'}, 'reason': reason}


def view_labels(rows, view):
    """Return (eligible segment rows, recording-level exclusion-reason table).

    Other/X and failed QC are separate reasons, potentially on the same recording.
    X recovered from a QC-overwritten source still excludes both views.
    """
    _check(view in ('A', 'B'), 'unknown label view')
    frame = scoped_rows(rows)
    source = frame.label.copy()
    for index, row in frame.iterrows():
        if row.qc_status != 'pass':
            source.loc[index] = source_label(row)
    admitted, excluded = [], []
    for _, group in frame.groupby(['dataset', 'recording_id'], sort=True):
        _check(all(group[c].astype(str).nunique() == 1 for c in
                   ('patient_id', 'age_group', 'audio_path', 'sha256', 'split')),
               'inconsistent recording identity')
        recovered = source.loc[group.index]
        reasons = []
        if group.label.eq('X').any() or recovered.eq('X').any():
            reasons.append('other_x')
        if view == 'B' and (group.qc_status.ne('pass').any() or group.label.eq('Q').any()):
            reasons.append('qc_fail')
        labels = recovered if view == 'A' else group.label
        if not labels.isin((*CLASSES, 'X', 'Q')).all() or (view == 'A' and labels.eq('Q').any()):
            reasons.append('invalid_label')
        if reasons:
            excluded.extend(_exclusion(group, view, reason) for reason in reasons)
        else:
            selected = group.copy()
            selected['label'] = labels
            admitted.append(selected)
    return (pd.concat(admitted).reset_index(drop=True) if admitted else frame.iloc[:0].copy(),
            pd.DataFrame(excluded, columns=EXCLUSION_COLUMNS))


def output_label(votes, threshold=FROZEN['threshold']):
    votes = np.asarray(votes, dtype=float)
    return CLASSES[int(votes.argmax())] if votes.max() >= threshold else 'U'


def vote(probabilities):
    _check(len(probabilities) > 0, 'recording has no labelled windows')
    return evaluation.aggregate(np.asarray(probabilities), np.zeros(len(probabilities), dtype=int), 1, 'vote')[0]


def window_probabilities(model, signal, bounds, *, window=FROZEN['window'], batch_size=32):
    """Softmax each provided half-hop window exactly once, with right-zero-pad."""
    results = []
    _check(batch_size > 0, 'batch_size must be positive')
    with torch.no_grad():
        for offset in range(0, len(bounds), batch_size):
            batch = []
            for start, end in bounds[offset:offset + batch_size]:
                _check(0 <= start < end <= len(signal) and end-start <= window*4000, 'invalid window bounds')
                padded = np.zeros(window * 4000, dtype=np.float32)
                padded[:end-start] = signal[start:end]
                batch.append(logmel_input(padded))
            results.append(torch.softmax(model(torch.from_numpy(np.stack(batch))), dim=1).cpu().numpy())
    return np.concatenate(results) if results else np.empty((0, 4), dtype=float)


def _record_prediction(meta, probabilities, truth, mode, perturbation, threshold):
    fractions = vote(probabilities)
    return dict(meta, mode=mode, perturbation=perturbation, truth=CLASSES.index(truth),
                n_windows=len(probabilities), **dict(zip(VOTE_COLUMNS, map(float, fractions))),
                pred_t0=int(fractions.argmax()), output_066=output_label(fractions, threshold))


def _labelled_indices(rows, bounds):
    return [i for i, (start, end) in enumerate(bounds)
            if label_interval(rows, start / 4000, end / 4000) is not None]


def _truth(rows):
    label = label_interval(rows, -np.inf, np.inf)
    _check(label in CLASSES, 'invalid recording truth')
    return label


def infer_recording(model, signal, rows, *, window=FROZEN['window'], threshold=FROZEN['threshold'],
                    batch_size=32):
    """One shared full-recording grid for both views, plus unique segment crops."""
    views = {v: view_labels(rows, v)[0] for v in ('A', 'B')}
    if not any(len(r) for r in views.values()):
        return [], []
    bounds = window_bounds(len(signal), window)
    probabilities = window_probabilities(model, signal, bounds, window=window, batch_size=batch_size)
    window_cache = dict(zip(bounds, probabilities))
    records, segments, crops = [], [], {}
    for view, eligible in views.items():
        if eligible.empty:
            continue
        meta = _metadata(eligible, view)
        indices = _labelled_indices(eligible, bounds)
        records.append(_record_prediction(meta, probabilities[indices], _truth(eligible), 'labelled', 'none', threshold))
        if view == 'B':
            records.append(_record_prediction(meta, probabilities, _truth(eligible), 'all', 'none', threshold))
        for row in eligible.itertuples(index=False):
            start_s, end_s = float(row.seg_start_s), float(row.seg_end_s)
            _check(np.isfinite([start_s, end_s]).all() and 0 <= start_s < end_s, 'invalid segment span')
            start, end = max(0, int(start_s*4000)), min(len(signal), int(end_s*4000))
            _check(start < end, 'segment crop is empty')
            key = (start, end)
            if key not in crops:
                crop_bounds = [(start+a, start+b) for a, b in window_bounds(end-start, window)]
                missing = [bound for bound in crop_bounds if bound not in window_cache]
                if missing:
                    extra = window_probabilities(model, signal, missing, window=window, batch_size=batch_size)
                    window_cache.update(zip(missing, extra))
                p = np.asarray([window_cache[bound] for bound in crop_bounds])
                crops[key] = (len(p), vote(p))
            count, fractions = crops[key]
            segments.append(dict(meta, segment_id=str(row.segment_id), seg_start_s=start_s, seg_end_s=end_s,
                                 truth=CLASSES.index(row.label), n_windows=count,
                                 **dict(zip(VOTE_COLUMNS, map(float, fractions)))))
    return records, segments


def perturb(signal, segments, name, recording_key):
    """Pure perturbation of a 4 kHz signal; original signal/table stay untouched."""
    _check(name in ('none', *PERTURBATIONS), 'unknown perturbation')
    result, shifted = signal.copy(), segments.copy(deep=True)
    if name.startswith('offset_'):
        seconds = float(name.split('_')[1])
        result = result[int(seconds*4000):]
        shifted['seg_start_s'] = (shifted.seg_start_s.astype(float) - seconds).clip(lower=0)
        shifted['seg_end_s'] = (shifted.seg_end_s.astype(float) - seconds).clip(lower=0, upper=len(result)/4000)
        shifted = shifted.loc[shifted.seg_end_s > shifted.seg_start_s].reset_index(drop=True)
    elif name.startswith('gain_'):
        result = result * float(name.split('_')[1])
        if name == 'gain_2.0':
            result = np.clip(result, -1, 1)
    elif name.startswith('noise_'):
        snr = float(name.split('_')[1][:-2])
        rng = np.random.default_rng([42, zlib.crc32(recording_key.encode())])
        power = float(np.mean(np.square(signal.astype(np.float64))))
        result = (result + rng.normal(0, np.sqrt(power / 10**(snr/10)), len(signal))).astype(np.float32)
    return result, shifted


def write_once(path, content: bytes):
    """Publish a fully flushed temp file atomically, without overwriting a target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
            name = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f'write-once content differs: {path}') from None
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def write_csv_once(path, rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    write_once(path, stream.getvalue().encode('utf-8'))


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def read_manifest(manifest_path=FROZEN['manifest_path'], expected_manifest_sha=FROZEN['manifest_sha256']):
    raw = _checked_bytes(manifest_path, expected_manifest_sha)
    frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False, dtype={
        c: str for c in ('dataset', 'recording_id', 'patient_id', 'segment_id', 'sha256')})
    _check(not frame.segment_id.duplicated().any(), 'duplicate segment identity')
    for column in ('seg_start_s', 'seg_end_s'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    return frame


def dry_run(manifest_path=FROZEN['manifest_path'], expected_manifest_sha=FROZEN['manifest_sha256']):
    """Metadata only. No model/config/audio/cache access."""
    frame = read_manifest(manifest_path, expected_manifest_sha)
    summary = []
    for view in ('A', 'B'):
        eligible, exclusions = view_labels(frame, view)
        for dataset in DATASETS:
            for age in ('full', 'adult'):
                source = scoped_rows(frame)
                source = source.loc[source.dataset.eq(dataset)]
                selected = eligible.loc[eligible.dataset.eq(dataset)]
                excluded = exclusions.loc[exclusions.dataset.eq(dataset)]
                if age == 'adult':
                    source = source.loc[source.age_group.eq('adult')]
                    selected = selected.loc[selected.age_group.eq('adult')]
                    excluded = excluded.loc[excluded.age_group.eq('adult')]
                summary.append(dict(dataset=dataset, view=view, age_scope=age,
                                    total_recordings=int(source.recording_id.nunique()),
                                    eligible_recordings=int(selected.recording_id.nunique()),
                                    eligible_segments=len(selected),
                                    eligible_patients=int(selected.patient_id.nunique()),
                                    excluded_recordings=int(excluded.recording_key.nunique()),
                                    exclusions={str(k): int(v) for k, v in excluded.groupby('reason').recording_key.nunique().items()}))
    return summary


def _root_path(root, path):
    return (Path(root) / Path(path)).resolve()


def preflight(root=Path('.'), model_path=FROZEN['model_path'], expected_sha=FROZEN['model_sha256'],
              config_path=FROZEN['config_path'], manifest_path=FROZEN['manifest_path'],
              expected_manifest_sha=FROZEN['manifest_sha256'], *, window=FROZEN['window'],
              threshold=FROZEN['threshold']):
    """Five train recordings via TrainingAudio; two independently loaded passes.

    Uses TrainingAudio's existing train-only cache, never the phase5 output area.
    """
    root = Path(root).resolve()
    frame = read_manifest(_root_path(root, manifest_path), expected_manifest_sha)
    admitted = adult_training_rows(frame)
    keys = list(admitted.groupby(['dataset', 'recording_id'], sort=True).groups)[:5]
    _check(len(keys) == 5, 'preflight needs five admitted train recordings')
    passes = []
    for _ in range(2):
        seed_everything()
        model = load_frozen(_root_path(root, model_path), expected_sha, _root_path(root, config_path),
                            window=window, threshold=threshold)
        loader = TrainingAudio(root, frame)
        predictions = []
        for dataset, recording in keys:
            rows = admitted.loc[admitted.dataset.eq(dataset) & admitted.recording_id.eq(recording)]
            signal = loader.load(rows.iloc[0])
            bounds = window_bounds(len(signal), window)
            p = window_probabilities(model, signal, bounds, window=window)
            predictions.append(_record_prediction(_metadata(rows, 'B'), p[_labelled_indices(rows, bounds)],
                                                  _truth(rows), 'labelled', 'none', threshold))
        passes.append(predictions)
    _check(_json_bytes(passes[0]) == _json_bytes(passes[1]), 'preflight outputs are not identical')
    for prediction in passes[0]:
        print(json.dumps(prediction, sort_keys=True), flush=True)
    return passes[0]


def _code_hashes():
    # All local inference/label/metric dependencies, not only the new runner.
    package = Path(__file__).resolve().parent
    names = ('locked_eval.py', 'phase5_scoring.py', 'cnn.py', 'windowing.py', 'evaluation.py',
             'features.py', 'feature_store.py', 'training_data.py', 'icbhi.py', 'hf_lung.py',
             'fraiwan.py', 'learning.py', 'manifest.py')
    result = {f'src/stethoai/{name}': sha256_file(package / name) for name in names}
    script = package.parents[1] / 'scripts/run_locked_eval.py'
    result['scripts/run_locked_eval.py'] = sha256_file(script)
    return result


def run_evaluation(root=Path('.'), model_path=FROZEN['model_path'], expected_sha=FROZEN['model_sha256'],
                   config_path=FROZEN['config_path'], manifest_path=FROZEN['manifest_path'],
                   expected_manifest_sha=FROZEN['manifest_sha256'], *, out_dir=Path('reports/phase5'),
                   window=FROZEN['window'], threshold=FROZEN['threshold'], batch_size=32, progress=print):
    """Commit each dataset once, then assemble immutable canonical CSVs.

    A completed checkpoint is resumed without any waveform/model load. Partial
    (uncommitted) datasets may be recomputed, but already published CSVs must be
    byte-identical. A changed authority, code file, or checkpoint fails closed.
    """
    root = Path(root).resolve()
    out = _root_path(root, out_dir)
    _check(out.is_relative_to(root), 'output must be inside project root')
    frame = read_manifest(_root_path(root, manifest_path), expected_manifest_sha)
    model_path, config_path = _root_path(root, model_path), _root_path(root, config_path)
    _checked_bytes(model_path, expected_sha)  # hashing is not inference/deserialization
    signature = dict(model_sha256=expected_sha, config_sha256=sha256_file(config_path),
                     manifest_sha256=expected_manifest_sha, code_sha256=_code_hashes(),
                     window=window, threshold=threshold, aggregation='vote', classes=list(CLASSES),
                     feature_config=FEATURE_CONFIG, label_rule_version=LABEL_RULE_VERSION,
                     batch_size=batch_size)
    selected = scoped_rows(frame)
    combined = {key: [] for key in OUTPUT_FILES}
    checkpoints, loader, model, processed = {}, None, None, 0
    for dataset in DATASETS:
        directory = out / 'datasets' / dataset
        checkpoint_path = directory / 'checkpoint.json'
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
            _check(checkpoint['signature'] == signature, 'checkpoint authority/code mismatch')
            for name, (filename, _) in OUTPUT_FILES.items():
                raw = _checked_bytes(directory / filename, checkpoint['output_sha256'][filename])
                rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
                _check(len(rows) == checkpoint['row_counts'][name], 'checkpoint row count mismatch')
                combined[name].extend(rows)
            checkpoints[dataset] = checkpoint
            progress(f'{dataset}: verified completed dataset checkpoint; skipped inference')
            continue
        started = time.perf_counter()
        rows = selected.loc[selected.dataset.eq(dataset)]
        output = {key: [] for key in OUTPUT_FILES}
        views = {}
        for view in ('A', 'B'):
            views[view], exclusions = view_labels(rows, view)
            output['exclusions'].extend(exclusions.to_dict('records'))
        admitted_keys = set()
        for view_rows in views.values():
            admitted_keys.update(zip(view_rows.dataset, view_rows.recording_id))
        for (_, recording), group in rows.groupby(['dataset', 'recording_id'], sort=True):
            processed += 1
            if (dataset, recording) in admitted_keys:
                if model is None:
                    seed_everything()
                    model = load_frozen(model_path, expected_sha, config_path, window=window, threshold=threshold)
                    loader = LockedAudio(root, frame)
                signal = loader.load(group.iloc[0])
                records, segments = infer_recording(model, signal, group, window=window,
                                                     threshold=threshold, batch_size=batch_size)
                output['recording'].extend(records)
                output['segment'].extend(segments)
                b_rows = views['B'].loc[views['B'].recording_id.eq(recording)]
                if not b_rows.empty and b_rows.age_group.eq('adult').all():
                    meta = _metadata(b_rows, 'B')
                    for name in PERTURBATIONS:
                        shifted, shifted_rows = perturb(signal, b_rows, name, meta['recording_key'])
                        bounds = window_bounds(len(shifted), window) if len(shifted) else []
                        indices = _labelled_indices(shifted_rows, bounds)
                        if not indices:
                            output['exclusions'].append(_exclusion(b_rows, 'B', f'{name}:no_labelled_windows'))
                            continue
                        # Infer only the labelled subset needed for this proxy.
                        p = window_probabilities(model, shifted, [bounds[i] for i in indices],
                                                 window=window, batch_size=batch_size)
                        output['recording'].append(_record_prediction(meta, p, _truth(b_rows),
                                                                      'labelled', name, threshold))
            if processed % 100 == 0:
                progress(f'{processed} recordings processed (current dataset: {dataset})')
        for name, (filename, columns) in OUTPUT_FILES.items():
            write_csv_once(directory / filename, output[name], columns)
        checkpoint = dict(signature=signature, dataset=dataset,
                          output_sha256={filename: sha256_file(directory / filename)
                                         for filename, _ in OUTPUT_FILES.values()},
                          row_counts={name: len(table) for name, table in output.items()},
                          recordings_seen=int(rows.recording_id.nunique()),
                          timings={'inference_and_write_seconds': time.perf_counter() - started})
        # Commit last. If an interrupted run already published equal CSVs, retain
        # no invented timing; this dataset's eventual successful attempt is timed.
        write_once(checkpoint_path, _json_bytes(checkpoint))
        checkpoints[dataset] = checkpoint
        for name in combined:
            # Canonicalize values through CSV so first run and resume are identical.
            combined[name].extend(list(csv.DictReader(io.StringIO(
                (directory / OUTPUT_FILES[name][0]).read_text(encoding='utf-8')))))
        progress(f'{dataset}: committed {checkpoint["recordings_seen"]} recordings')
    for name, (filename, columns) in OUTPUT_FILES.items():
        write_csv_once(out / filename, combined[name], columns)
    manifest = dict(signature, row_counts={name: len(table) for name, table in combined.items()},
                    datasets=checkpoints,
                    output_sha256={filename: sha256_file(out / filename) for filename, _ in OUTPUT_FILES.values()},
                    timings={'inference_and_write_seconds': sum(
                        c['timings']['inference_and_write_seconds'] for c in checkpoints.values())})
    write_once(out / 'run_manifest.json', _json_bytes(manifest))
    return manifest
