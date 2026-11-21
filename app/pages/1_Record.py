from pathlib import Path
import tempfile

import streamlit as st

from workflow_ui import setup, site_note, qc_words, next_link
from stethoai.app_core import check_audio, decodable

store = setup('Record a site', step=1)
minimum = store.protocol['min_duration_s']
st.write(f'Record at least {minimum} seconds at one chest site, then upload the file. '
         'Use the codes the study team gave you, never names.')
with st.expander('Codes for this session', expanded='case_code' not in st.session_state):
    case = st.text_input('Case code', value=st.session_state.get('case_code', 'S03'))
    rater = st.text_input('Your rater code', value=st.session_state.get('rater_code', 'R01'))
    number = int(st.number_input('Trial number', min_value=1, value=1, step=1,
                                 help='Use a new trial number when you repeat a site.'))
    mode = st.radio('Stethoscope mode', store.protocol['modes'], horizontal=True,
                    format_func={'BELL': 'Bell', 'DIA': 'Diaphragm'}.get)
sites = store.protocol['site_codes']
existing = [r for r in store.table('rater_study') if r['case_code'] == case and
            r['rater_code'] == rater and store.trial_number(r['trial_id']) == number and
            store.recording_id(r['trial_id']).endswith('_' + mode)]
done = {r['site_code'] for r in existing}
next_site = next((s for s in sites if s not in done), sites[0])
st.progress(len(done) / len(sites), text=f'{len(done)} of {len(sites)} sites done · {case} · trial {number} · {mode}')
site = st.selectbox('Chest site', sites, index=sites.index(next_site),
                    help='Order: ' + ' → '.join(sites))
site_note(store)
upload = st.file_uploader('Upload the recording', type=None,
                          help='WAV, FLAC, MP3, or phone formats such as M4A/AAC (converted automatically). '
                               'Remove names from file details before use.')
st.caption('The app checks length, silence and distortion. Listen for talking or loud noise yourself and re-record if present.')
if upload is not None:
    # Uploaded names are untrusted and never used as filesystem paths.
    with tempfile.NamedTemporaryFile(dir=store.root, suffix='.upload', delete=False) as f:
        pending = Path(f.name)
        f.write(upload.getvalue())
    uploaded, pending = pending, decodable(pending)
    try:
        qc = check_audio(pending)
        if qc['qc_status'] == 'fail':
            st.error(f"Recording not usable: {qc_words(qc['qc_reason'])}. "
                     f'Record this site again for at least {minimum} seconds, then upload it.')
        else:
            st.success(f"Quality check passed: {qc['duration_s']:.1f} seconds. Listen to confirm it sounds right.")
            st.audio(upload.getvalue())
            if st.button(f'Save site {site} and continue', type='primary', icon=':material/save:'):
                try:
                    tid = store.create_trial(pending, case, site, number, mode, rater)
                    st.session_state.update(active_trial=tid, case_code=case, rater_code=rater)
                    st.success(f'Saved as {store.recording_path(tid).name}')
                    next_link('pages/2_Nurse_answer.py', 'Next: your answer')
                except (ValueError, OSError, RuntimeError) as error:
                    st.error(f'Could not save: {error}')
    finally:
        pending.unlink(missing_ok=True)
        uploaded.unlink(missing_ok=True)
