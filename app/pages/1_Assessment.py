import fresh_modules

fresh_modules.refresh()

from html import escape
import time

import streamlit as st
import streamlit.components.v1 as components

from lung_chart import PLACEMENT, chest_chart
from workflow_ui import (DRAFT_STATUS, LISTEN_AGAIN, open_number, recording_words, steps_at, walk, FINDINGS, MODE_NAMES, SITE_GROUPS, SITE_NAMES, SITE_ORDER, action_button,
                         analysis_panel, assessment_from_url, attach_file, draft_key, drafts, empty_state, fail, flash, flash_text, footer,
                         AnalysisTrace, record_place, get_model, model_state, held_anywhere, is_complete, is_conflict, keep_in_url, keep_pick, next_retake, parse_trial, picks,
                         next_step, plain_error, prune_saved, qc_words, restore_pick, save_draft, saving_words, session_examiner,
                         setup, signed_pairs, site_rows, site_status, take_failure, unsaved_sites)

store = setup('Assessment')
assessment = st.session_state.get('assessment')
if not assessment:  # After a reload the session is new; reopen the case and site named in the URL (D-108).
    assessment, url_site = assessment_from_url(store)
    if assessment:
        st.session_state['assessment'], st.session_state['site'] = assessment, url_site
        # A reload of the examiner's own tab keeps their code; a different code already in use is refused below.
        st.session_state.setdefault('examiner', assessment['examiner'])
        if url_site:
            st.session_state['chart_view'] = 'front' if url_site.startswith('A') else 'back'
            st.session_state['focus_work'] = True  # On phones, bring the reopened site into view.
if not assessment:
    st.title('Assessment')
    empty_state('No assessment is open', ' Open or start one on the Assessments screen.')
    st.page_link('pages/0_Assessments.py', label='Go to Assessments')
    footer()
    st.stop()
if assessment['examiner'] != session_examiner():
    # Blinding (D-106): another examiner's findings are never opened from this session.
    st.title('Assessment')
    empty_state('This assessment belongs to another examiner code',
                ' Each examiner sees only their own assessments.')
    st.page_link('pages/0_Assessments.py', label='Go to Assessments')
    footer()
    st.stop()

order = list(SITE_ORDER)
minimum = store.protocol['min_duration_s']
if st.session_state.get('site') not in order:
    # Open at the first recording not signed, in recording order (D-119).
    first_site, first_round = next_step(store, assessment) or (order[0], 1)
    st.session_state['site'] = first_site
    if assessment['round'] is None:
        assessment = {**assessment, 'round': first_round}
site = st.session_state['site']
if assessment['round'] not in steps_at(store, assessment, site):
    # No recording number, or one this site does not take (a second recording only where drawn, D-119).
    assessment = {**assessment, 'round': open_number(store, assessment, site)}
st.session_state['assessment'] = assessment
rows = site_rows(store, assessment)
keep_in_url(assessment, site)
steps = walk(store, assessment)
complete = len(signed_pairs(store, assessment))
st.title(f"Case {assessment['case']}")
st.caption(f"{MODE_NAMES[assessment['mode']]}, examiner {assessment['examiner']}. "
           f'{complete} of {len(steps)} recordings signed.')


def retakes():
    """Pending retakes, one per assessment and site, so a retake at one site never cancels another (D-109)."""
    return st.session_state.setdefault('retakes', {})


def retake_slot(code):
    return f"{assessment['case']}_{assessment['examiner']}_{assessment['round']}_{assessment['mode']}_{code}"


def side(code):
    return 'Front' if code.startswith('A') else 'Back'


def capture(number=None):
    number = number or assessment['round']
    key = draft_key(assessment, site, number)
    draft = drafts().get(key)
    st.subheader('Attach the recording' if draft is None else 'Check the recording')
    if draft is None:
        st.write(f'Attach a recording of at least {minimum} seconds (WAV, FLAC, MP3 or M4A).')
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
        st.caption('The quality check passed. Record again if you hear talking or noise.')
        st.audio(draft['data'])
        # The case and site survive a reload through the URL; the attached file does not (D-098, D-108).
        st.caption('An unsaved file is lost on page reload.')
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
                retakes().pop(retake_slot(site), None)
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
    st.caption('Your first finding is locked when saved, before you see the analysis.')
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
        st.subheader('Acoustic analysis')
        trace = AnalysisTrace(store.recording_path(trial_id), record_place(store))
        for _ in range(3):  # Blank the first-finding controls left greyed from the last run while the analysis runs (D-116).
            st.empty()
        try:
            model = trace.load_model(get_model, model_state().get('ready', False))
            store.analyze_trial(trial_id, model, trace.update)
            trace.finish()
            st.rerun()
        except (ValueError, OSError, RuntimeError) as error:
            trace.fail()
            st.error(f'The analysis did not finish. Your finding is saved. {error}')
            st.button('Try analysis again', key=f'retry_{trial_id}')
            return
    st.subheader('Acoustic analysis')
    analysis_panel(result, store.recording_path(trial_id), trial_id, signed=bool(row['assisted_label']))
    if result['finding'] == 'Q':
        st.caption('The unusable recording stays in the study record, and the new one replaces it here.')
        if st.button('Record this site again', type='primary', key=f'again_{trial_id}'):
            retakes()[retake_slot(parse_trial(trial_id)['site'])] = trial_id
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
    final = st.radio('Choose the finding to sign', list(FINDINGS), format_func=FINDINGS.get, index=None, key=key,
                     on_change=keep_pick, args=(key,))
    st.caption('A signed finding is locked in the study record.')
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
    number = assessment['round'] if code == site else open_number(store, assessment, code)
    row = rows.get(code) if code == site else site_rows(store, {**assessment, 'round': number}).get(code)
    if not row and number >= 2 and not any(k.startswith(draft_key(assessment, code, '')) for k in drafts()):
        return LISTEN_AGAIN
    has_file = any(k.startswith(draft_key(assessment, code, '')) for k in drafts())
    if has_file and (not row or retakes().get(retake_slot(code)) == row['trial_id']):
        return DRAFT_STATUS
    return site_status(row)


def where_words():
    """Placement for a first recording; the repeat instruction for a second one (D-119: no cough, same breathing)."""
    if assessment['round'] >= 2:
        return 'Lift the chestpiece, place it again at the same spot, and record without a cough.'
    return PLACEMENT[site]


def place(code):
    return f'{side(code).lower()}, {SITE_NAMES[code].lower()}'


with st.container(key='sheet'):
    sheet, work = st.columns([7, 5], gap='large')
    with sheet:
        picked = chest_chart(SITE_GROUPS, SITE_NAMES, {code: status_of(code) for code in order}, site)
        stamped = st.session_state.pop('just_signed', None)
        if stamped:  # The marker of a site just signed fills in with a short stamp.
            st.html(f'<style>.st-key-pick_{stamped} button::before{{animation:stamp .42s cubic-bezier(.16,1,.3,1);}}</style>')
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
            # A site opens at its first recording not signed, else its last (D-119).
            if picked != site:
                st.session_state['assessment'] = {**assessment, 'round': open_number(store, assessment, picked)}
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
                        + (f'<p class="place">{where_words()}</p>' if not row else ''), unsafe_allow_html=True)
            numbers = steps_at(store, assessment, site)
            if len(numbers) > 1:  # Two joined buttons, one per recording at this site (the former trial pair, D-112).
                with st.container(key='trial_switch', horizontal=True):
                    for number in numbers:
                        current = number == assessment['round']
                        if st.button(recording_words(number).capitalize(), key=f'trial_pick_{number}',
                                     type='primary' if current else 'secondary') and not current:
                            st.session_state['assessment'] = {**assessment, 'round': number}
                            st.rerun()
            if failure:
                st.markdown(f'<p class="problem" role="alert">{escape(failure["message"])}</p>', unsafe_allow_html=True)
            if not row:
                capture()
            elif retakes().get(retake_slot(site)) == row['trial_id']:
                st.markdown(f'<p class="place">{where_words()}</p>', unsafe_allow_html=True)
                retake = next_retake(store, assessment, site)
                capture(retake)
                if st.button('Cancel retake', key=f"cancel_{row['trial_id']}"):
                    drafts().pop(draft_key(assessment, site, retake), None)
                    retakes().pop(retake_slot(site), None)
                    st.rerun()
            elif not row['manual_locked_at']:
                document(row)
            else:
                review(row)
                if row['assisted_label']:
                    following = next_step(store, assessment, (site, assessment['round']))
                    if following:
                        code, number = following
                        words = (f'Listen again: {place(code)}' if number >= 2 else f'Next site: {place(code)}')
                        if side(code) != side(site) and number == 1:
                            st.caption('Turn the patient to record the back.' if code.startswith('P')
                                       else 'Turn the patient to record the front.')
                        if st.button(words, type='primary', key=f"next_{site}_{assessment['round']}"):
                            st.session_state['assessment'] = {**assessment, 'round': number}
                            st.session_state['site'] = code
                            st.session_state['chart_view'] = 'front' if code.startswith('A') else 'back'
                            st.session_state['focus_work'] = True
                            st.rerun()
                    else:
                        st.markdown(f'<p class="saved" role="status">All {len(steps)} recordings are signed.</p>',
                                    unsafe_allow_html=True)
                        st.page_link('pages/0_Assessments.py', label='Back to Assessments')
footer()
# A refresh or closed tab would lose unsaved work, so the browser asks first while any is held.
components.html(f'<script>/*{time.time_ns()}*/window.parent.onbeforeunload='
                + ('function(e){e.preventDefault();e.returnValue="";}' if unsaved or held_anywhere() else 'null') + ';</script>', height=0)
# Joined pairs (Front/Back, First/Second recording) announce the chosen button, not only its colour (WCAG 4.1.2).
components.html('<script>const d=window.parent.document;const mark=()=>d.querySelectorAll('
                '".st-key-view_switch button,.st-key-trial_switch button").forEach(b=>{const p=String((b.dataset.testid||"")'
                '.includes("primary"));if(b.getAttribute("aria-pressed")!==p)b.setAttribute("aria-pressed",p);});mark();'
                'if(window.parent.__pressedWatch)window.parent.__pressedWatch.disconnect();'
                'window.parent.__pressedWatch=new MutationObserver(mark);window.parent.__pressedWatch.observe(d.body,'
                '{subtree:true,childList:true,attributes:true,attributeFilter:["data-testid","class"]});</script>', height=0)
if st.session_state.pop('focus_work', False):
    # On phones the work sits below the chart; bring the chosen site into view.
    components.html(f'<script>/*{time.time_ns()}*/const w=window.parent.document.querySelector(".st-key-work");'
                    f'if(w){{w.tabIndex=-1;w.focus({{preventScroll:true}});'
                    f'if(window.parent.innerWidth<=640)w.scrollIntoView({{block:"start"}});}}</script>', height=0)
