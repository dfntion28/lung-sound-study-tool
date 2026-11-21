from pathlib import Path
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components

from lung_chart import PLACEMENT, chest_chart
from stethoai.app_core import check_audio, decodable
from workflow_ui import (FINDINGS, MODE_NAMES, SITE_GROUPS, SITE_NAMES, SITE_ORDER, analysis_panel, empty_state, footer,
                         get_model, is_complete, next_retake, parse_trial, qc_words, setup, site_rows, site_status)

store = setup('Assessment')
assessment = st.session_state.get('assessment')
if not assessment:
    st.title('Assessment')
    empty_state('No assessment is open', ' Choose or create one on the Assessments screen.')
    st.page_link('Home.py', label='Go to Assessments')
    footer()
    st.stop()

order = list(SITE_ORDER)
minimum = store.protocol['min_duration_s']
rows = site_rows(store, assessment)
site = st.session_state.get('site')
if site not in order:
    site = next((s for s in order if not is_complete(rows.get(s))), order[0])
    st.session_state['site'] = site
complete = sum(is_complete(rows.get(s)) for s in order)
st.title(f"Case {assessment['case']}")
known_rounds = sorted({parse_trial(r['trial_id'])['round'] for r in store.table('rater_study')
                       if all(parse_trial(r['trial_id'])[k] == assessment[k] for k in ('case', 'examiner', 'mode'))}
                      | {assessment['round']})
st.caption(f"Trial {assessment['round']}, {MODE_NAMES[assessment['mode']].lower()}, examiner {assessment['examiner']}. "
           f'{complete} of {len(order)} sites signed.')
if len(known_rounds) > 1:
    chosen_round = st.selectbox('Trial', known_rounds, index=known_rounds.index(assessment['round']),
                                format_func=lambda n: f'Trial {n}', key=f"round_{assessment['round']}")
    if chosen_round != assessment['round']:
        st.session_state['assessment'] = {**assessment, 'round': chosen_round}
        st.session_state['site'] = None
        st.rerun()


def side(code):
    return 'Front' if code.startswith('A') else 'Back'


def next_open(current):
    start = order.index(current)
    for code in order[start + 1:] + order[:start]:
        if not is_complete(rows.get(code)):
            return code
    return None


def capture(number=None):
    number = number or assessment['round']
    st.subheader('Attach the recording')
    st.write(f'Record for at least {minimum} seconds, then attach a WAV, FLAC, MP3 or M4A file.')
    key = f"upload_{assessment['case']}_{site}_{number}_{assessment['mode']}_{assessment['examiner']}"
    upload = st.file_uploader('Recording file', key=key)
    if upload is None:
        return
    with tempfile.NamedTemporaryFile(dir=store.root, suffix='.upload', delete=False) as handle:
        pending = Path(handle.name)
        handle.write(upload.getvalue())
    uploaded, pending = pending, decodable(pending)
    try:
        quality = check_audio(pending)
        if quality['qc_status'] == 'fail':
            st.markdown(f'<p class="problem" role="alert">{qc_words(quality["qc_reason"])}. '
                        f'Record this site again for at least {minimum} seconds.</p>', unsafe_allow_html=True)
            return
        st.caption(f"Quality check passed at {quality['duration_s']:.1f} seconds. Listen for talking or room noise.")
        st.audio(upload.getvalue())
        if st.button('Save recording', type='primary', key=f'save_{key}'):
            try:
                store.create_trial(pending, assessment['case'], site, number, assessment['mode'],
                                   assessment['examiner'])
                st.session_state.pop('retake_for', None)
                st.rerun()
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f'The recording was not saved. {error}')
    finally:
        pending.unlink(missing_ok=True)
        uploaded.unlink(missing_ok=True)


def document(row):
    trial_id = row['trial_id']
    st.subheader('What did you hear?')
    st.audio(store.recording_path(trial_id).read_bytes(), format='audio/wav')
    label = st.radio('Your finding', list(FINDINGS), format_func=FINDINGS.get, index=None,
                     key=f'initial_{trial_id}')
    st.caption('This first finding cannot be changed after saving. The analysis runs next.')
    if st.button('Save and run analysis', type='primary', disabled=label is None, key=f'save_initial_{trial_id}'):
        try:
            store.lock_manual(trial_id, label)
            st.rerun()
        except ValueError as error:
            st.error(f'Your finding was not saved. {error}')


def review(row):
    trial_id = row['trial_id']
    st.markdown(f'<dl class="entry"><dt>Your first finding</dt><dd>{FINDINGS[row["manual_label"]]}</dd></dl>',
                unsafe_allow_html=True)
    result = store.get_ai(trial_id)
    if result is None:
        try:
            with st.spinner('Analyzing the recording...'):
                result = store.analyze_trial(trial_id, get_model())
            st.rerun()
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'The analysis did not finish. Your finding is saved. {error}')
            st.button('Try analysis again', key=f'retry_{trial_id}')
            return
    st.subheader('Acoustic analysis')
    analysis_panel(result, store.recording_path(trial_id), trial_id)
    if result['finding'] == 'Q':
        st.caption(f"The unusable recording stays in the study record. The new one replaces it for trial {assessment['round']}.")
        if st.button('Record this site again', type='primary', key=f'again_{trial_id}'):
            st.session_state['retake_for'] = trial_id
            st.session_state['focus_work'] = True
            st.rerun()
        return
    if row['assisted_label']:
        st.markdown(f'<dl class="entry"><dt>Signed finding</dt><dd>{FINDINGS[row["assisted_label"]]}</dd></dl>',
                    unsafe_allow_html=True)
        return
    st.subheader('Final finding')
    codes = list(FINDINGS)
    final = st.radio('Choose what you will sign', codes, format_func=FINDINGS.get, index=None,
                     key=f'final_{trial_id}')
    st.caption('Your signed finding cannot be changed after signing.')
    if st.button('Sign final finding', type='primary', disabled=final is None, key=f'sign_{trial_id}'):
        try:
            store.set_assisted(trial_id, final)
            st.rerun()
        except ValueError as error:
            st.error(f'Your final finding was not saved. {error}')


with st.container(key='sheet'):
    sheet, work = st.columns([7, 5], gap='large')
    with sheet:
        picked = chest_chart(SITE_GROUPS, SITE_NAMES, {code: site_status(rows.get(code)) for code in order}, site)
        st.caption('Site positions are provisional until an expert confirms them.')
        if picked:
            st.session_state['site'] = picked
            st.session_state['chart_view'] = 'front' if picked.startswith('A') else 'back'
            st.session_state['focus_work'] = True
            st.rerun()
    with work:
        with st.container(key='work'):
            row = rows.get(site)
            st.markdown(f'<p class="work-site">{side(site)}, {SITE_NAMES[site].lower()}</p>'
                        f'<p class="work-meta">{site_status(row)}</p>'
                        + (f'<p class="place">{PLACEMENT[site]}</p>' if not row else ''), unsafe_allow_html=True)
            if not row:
                capture()
            elif st.session_state.get('retake_for') == row['trial_id']:
                st.markdown(f'<p class="place">{PLACEMENT[site]}</p>', unsafe_allow_html=True)
                capture(next_retake(store, assessment, site))
                if st.button('Cancel retake', key=f"cancel_{row['trial_id']}"):
                    st.session_state.pop('retake_for', None)
                    st.rerun()
            elif not row['manual_locked_at']:
                document(row)
            else:
                review(row)
                if row['assisted_label']:
                    following = next_open(site)
                    if following and st.button(f'Next site: {side(following).lower()}, {SITE_NAMES[following].lower()}',
                                               type='primary', key=f'next_{site}'):
                        st.session_state['site'] = following
                        st.session_state['chart_view'] = 'front' if following.startswith('A') else 'back'
                        st.session_state['focus_work'] = True
                        st.rerun()
                    if not following:
                        st.success('All eight sites are signed.')
footer()
if st.session_state.pop('focus_work', False):
    # On phones the work sits below the chart; bring the chosen site into view.
    components.html(f'<script>/*{time.time_ns()}*/const w=window.parent.document.querySelector(".st-key-work");'
                    f'if(w){{w.tabIndex=-1;w.focus({{preventScroll:true}});'
                    f'if(window.parent.innerWidth<=640)w.scrollIntoView({{block:"start"}});}}</script>', height=0)
