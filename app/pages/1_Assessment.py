from html import escape
import time

import streamlit as st
import streamlit.components.v1 as components

from lung_chart import PLACEMENT, chest_chart
from workflow_ui import (DRAFT_STATUS, FINDINGS, MODE_NAMES, SITE_GROUPS, SITE_NAMES, SITE_ORDER, action_button,
                         analysis_panel, attach_file, draft_key, drafts, empty_state, fail, flash, flash_text, footer,
                         get_model, held_anywhere, is_complete, is_conflict, keep_pick, next_retake, parse_trial, picks,
                         plain_error, prune_saved, qc_words, restore_pick, save_draft, saving_words, setup, site_rows,
                         site_status, take_failure, unsaved_sites)

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
    key = draft_key(assessment, site, number)
    draft = drafts().get(key)
    st.subheader('Attach the recording' if draft is None else 'Check the recording')
    if draft is None:
        st.write(f'Record for at least {minimum} seconds, then attach a WAV, FLAC, MP3 or M4A file.')
        upload = st.file_uploader('Recording file', key=f'upload_{key}')
        if upload is None:
            return
        with st.spinner('Checking the recording...'):
            drafts()[key] = attach_file(store, upload.name, upload.getvalue())
        st.rerun()
    quality = draft['quality']
    st.markdown(f'<dl class="entry draft-file"><dt>Attached file</dt><dd>{escape(draft["name"])}</dd></dl>',
                unsafe_allow_html=True)
    if quality['qc_status'] == 'fail':
        st.markdown(f'<p class="problem" role="alert">{qc_words(quality["qc_reason"])}. '
                    f'Record this site again for at least {minimum} seconds.</p>', unsafe_allow_html=True)
    else:
        st.caption(f"Quality check passed at {quality['duration_s']:.1f} seconds. Listen for talking or room noise.")
        st.audio(draft['data'])
        duplicate = failure and failure['key'] == f'save_{key}' and failure['kind'] == 'duplicate'
        # After a same-file refusal Save cannot succeed; Replace file becomes the main action.
        running = False if duplicate else action_button('Save recording', f'save_{key}')
        if running:
            try:
                with st.spinner(saving_words(store, 'the recording')):
                    save_draft(store, assessment, site, number, draft)
            except (ValueError, OSError, RuntimeError) as error:
                words = plain_error(error)
                if 'file is already saved for' in words:
                    fail(site, f'save_{key}', f'The recording was not saved. {words} '
                         'Use Replace file to attach the recording made at this site.', 'duplicate')
                fail(site, f'save_{key}', f'The recording was not saved. {words} The file is still attached. '
                     'Press Save recording to try again.')
            else:
                drafts().pop(key, None)
                st.session_state.pop('retake_for', None)
                flash('Recording saved.')
                st.session_state['focus_work'] = True
                st.rerun()
    duplicate = failure and failure['key'] == f'save_{key}' and failure['kind'] == 'duplicate'
    if st.button('Replace file', key=f'replace_{key}', type='primary' if duplicate else 'secondary'):
        drafts().pop(key, None)
        st.rerun()


def document(row):
    trial_id = row['trial_id']
    st.subheader('What did you hear?')
    st.audio(store.recording_path(trial_id).read_bytes(), format='audio/wav')
    key = f'initial_{trial_id}'
    restore_pick(key)
    label = st.radio('Your finding', list(FINDINGS), format_func=FINDINGS.get, index=None, key=key,
                     on_change=keep_pick, args=(key,))
    chosen = f'You chose {FINDINGS[label].lower()}. ' if label else ''
    st.caption(f'{chosen}This first finding cannot be changed after saving. The analysis runs next.')
    if action_button('Save and run analysis', f'save_initial_{trial_id}', disabled=label is None):
        try:
            with st.spinner(saving_words(store, 'your finding')):
                store.lock_manual(trial_id, label)
        except ValueError as error:
            fail(site, f'save_initial_{trial_id}', f'Your finding was not saved. {plain_error(error)}'
                 + ('' if is_conflict(error) else ' Press Save and run analysis to try again.'))
        else:
            picks().pop(key, None)
            st.session_state['focus_work'] = True
            st.rerun()


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
    analysis_panel(result, store.recording_path(trial_id), trial_id, signed=bool(row['assisted_label']))
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
    key = f'final_{trial_id}'
    restore_pick(key)
    final = st.radio('Choose what you will sign', list(FINDINGS), format_func=FINDINGS.get, index=None, key=key,
                     on_change=keep_pick, args=(key,))
    chosen = f'You will sign {FINDINGS[final].lower()}. ' if final else ''
    st.caption(f'{chosen}A signed finding cannot be changed.')
    if action_button('Sign final finding', f'sign_{trial_id}', disabled=final is None):
        try:
            with st.spinner(saving_words(store, 'your signed finding')):
                store.set_assisted(trial_id, final)
        except ValueError as error:
            fail(site, f'sign_{trial_id}', f'Your final finding was not saved. {plain_error(error)}'
                 + ('' if is_conflict(error) else ' Press Sign final finding to try again.'))
        else:
            picks().pop(key, None)
            st.session_state['just_signed'] = site  # The status reads Signed and the marker fills in.
            st.session_state['focus_work'] = True
            st.rerun()


prune_saved(assessment, rows)  # Drafts whose save finished during an interrupted run are cleared here.
unsaved = unsaved_sites(assessment, rows)
failure = take_failure(site)


def status_of(code):
    row = rows.get(code)
    has_file = any(k.startswith(draft_key(assessment, code, '')) for k in drafts())
    if has_file and (not row or st.session_state.get('retake_for') == row['trial_id']):
        return DRAFT_STATUS
    return site_status(row)


def place(code):
    return f'{side(code).lower()}, {SITE_NAMES[code].lower()}'


with st.container(key='sheet'):
    sheet, work = st.columns([7, 5], gap='large')
    with sheet:
        picked = chest_chart(SITE_GROUPS, SITE_NAMES, {code: status_of(code) for code in order}, site)
        stamped = st.session_state.pop('just_signed', None)
        if stamped:  # The marker of a site just signed fills in with a short stamp.
            st.html(f'<style>.st-key-pick_{stamped} button::before{{animation:stamp .42s cubic-bezier(.16,1,.3,1);}}</style>')
        st.caption('Site positions are provisional until an expert confirms them.')
        others = [code for code in unsaved if code != site]
        if others:
            with st.container(key='unsaved_sites'):
                st.markdown('<p class="unsaved">' + ('Another site has' if len(others) == 1 else 'Other sites have')
                            + ' work that is not saved yet.</p>', unsafe_allow_html=True)
                with st.container(key='unsaved_links', horizontal=True):
                    for code in others:
                        if st.button(f'Go to {place(code)}', key=f'goto_{code}'):
                            picked = code
        if picked:
            st.session_state['site'] = picked
            st.session_state['chart_view'] = 'front' if picked.startswith('A') else 'back'
            st.session_state['focus_work'] = True
            st.rerun()
    with work:
        with st.container(key='work'), st.container(key=f'site_{site}'):
            row = rows.get(site)
            note = flash_text()  # A save is confirmed in the status line, not in an extra banner.
            st.markdown(f'<p class="work-site">{side(site)}, {SITE_NAMES[site].lower()}</p>'
                        f'<p class="work-meta" role="status">' + (f'<span class="saved">{note}</span> ' if note else '')
                        + f'{status_of(site)}</p>'
                        + (f'<p class="place">{PLACEMENT[site]}</p>' if not row else ''), unsafe_allow_html=True)
            if failure:
                st.markdown(f'<p class="problem" role="alert">{escape(failure["message"])}</p>', unsafe_allow_html=True)
            if not row:
                capture()
            elif st.session_state.get('retake_for') == row['trial_id']:
                st.markdown(f'<p class="place">{PLACEMENT[site]}</p>', unsafe_allow_html=True)
                retake = next_retake(store, assessment, site)
                capture(retake)
                if st.button('Cancel retake', key=f"cancel_{row['trial_id']}"):
                    drafts().pop(draft_key(assessment, site, retake), None)
                    st.session_state.pop('retake_for', None)
                    st.rerun()
            elif not row['manual_locked_at']:
                document(row)
            else:
                review(row)
                if row['assisted_label']:
                    following = next_open(site)
                    if following and st.button(f'Next site: {place(following)}', type='primary', key=f'next_{site}'):
                        st.session_state['site'] = following
                        st.session_state['chart_view'] = 'front' if following.startswith('A') else 'back'
                        st.session_state['focus_work'] = True
                        st.rerun()
                    if not following:
                        st.markdown('<p class="saved" role="status">All eight sites are signed for this trial.</p>',
                                    unsafe_allow_html=True)
                        st.page_link('Home.py', label='Back to Assessments')
footer()
# A refresh or closed tab would lose unsaved work, so the browser asks first while any is held.
components.html(f'<script>/*{time.time_ns()}*/window.parent.onbeforeunload='
                + ('function(e){e.preventDefault();e.returnValue="";}' if unsaved or held_anywhere() else 'null') + ';</script>', height=0)
if st.session_state.pop('focus_work', False):
    # On phones the work sits below the chart; bring the chosen site into view.
    components.html(f'<script>/*{time.time_ns()}*/const w=window.parent.document.querySelector(".st-key-work");'
                    f'if(w){{w.tabIndex=-1;w.focus({{preventScroll:true}});'
                    f'if(window.parent.innerWidth<=640)w.scrollIntoView({{block:"start"}});}}</script>', height=0)
