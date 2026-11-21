import fresh_modules

fresh_modules.refresh()

from collections import defaultdict

import streamlit as st

from workflow_ui import (MODE_HELP, MODE_NAMES, MODE_ORDER, clean_code, empty_state, footer, held_anywhere, is_complete,
                         open_assessment, own_rows, parse_trial, session_examiner, setup, trial_count)

store = setup('Assessments')
st.title('Assessments')
sites = store.protocol['site_codes']
examiner = session_examiner()
if not examiner:
    # Blinding (D-106): the code is entered once per session and only its own records are listed.
    form_column, _ = st.columns([5, 7], gap='large')
    with form_column:
        with st.form('examiner_code'):
            entered = st.text_input('Examiner code', max_chars=16, placeholder='For example, R01')
            st.caption('Enter the study code assigned to you as an examiner and use the same code at every session. '
                       'Only assessments saved under this code are listed, so each examiner\'s findings stay hidden '
                       'from the other examiners.')
            use = st.form_submit_button('Use this code', type='primary')
        if use:
            code = clean_code(entered)
            if code:
                st.session_state['examiner'] = code
                st.rerun()
            st.error('Enter your examiner code. It starts with a letter and uses only letters and numbers.')
    footer()
    st.stop()

with st.container(key='examiner_line', horizontal=True, vertical_alignment='center'):
    st.markdown(f'<p class="examiner">Examiner {examiner}</p>', unsafe_allow_html=True)
    if st.button('Change examiner', key='change_examiner'):
        if held_anywhere():
            st.session_state['change_refused'] = True
        else:
            for key in ('examiner', 'assessment', 'site', 'chart_view'):
                st.session_state.pop(key, None)
            st.rerun()
if st.session_state.pop('change_refused', False):
    st.markdown('<p class="problem" role="alert">The examiner code was not changed. An attached file or a chosen '
                'finding is not saved yet. Save it at its site, then change the code.</p>',
                unsafe_allow_html=True)
left, right = st.columns([7, 5], gap='large')

with right:
    st.subheader('New assessment')
    with st.form('new_assessment'):
        case = st.text_input('Case code', max_chars=16, placeholder='For example, S01')
        st.caption('Enter the subject\'s study code exactly as it appears in the recording log. '
                   'Do not enter the subject\'s name, because the study keeps coded records only.')
        # Stored values stay BELL and DIA so older records read the same (D-097).
        modes = [m for m in MODE_ORDER if m in store.protocol['modes']]
        mode = st.radio('Stethoscope mode', modes, format_func=MODE_NAMES.get, index=modes.index('DIA'),
                        captions=[MODE_HELP[m] for m in modes], key='mode_choice')
        st.caption('Select the mode the stethoscope is set to. The recording protocol requires diaphragm mode '
                   'for lung sounds.')
        start = st.form_submit_button('Start assessment', type='primary')
    st.caption('Each site is recorded twice. Trial 1 is the first recording; trial 2 follows immediately, after the '
               'chestpiece has been lifted and placed again. The assessment presents the sites and trials in this order.')
    if start:
        case_code = clean_code(case)
        if not case_code:
            st.error('Enter the case code. It starts with a letter and uses only letters and numbers.')
        else:
            open_assessment(case_code, examiner, None, mode)

with left:
    st.subheader('Your saved assessments')
    groups = defaultdict(list)
    for row in own_rows(store):
        info = parse_trial(row['trial_id'])
        groups[(info['case'], info['mode'])].append(row)
    if not groups:
        empty_state('No assessments yet', f' No assessment has been saved under examiner {examiner}. '
                    'Enter a case code to start one.')
    for case_code, mode in sorted(groups):
        total = len(sites) * trial_count(store, dict(case=case_code, examiner=examiner, mode=mode))
        done = sum(is_complete(row) for row in groups[(case_code, mode)])
        label = f'Case {case_code}, {MODE_NAMES[mode].lower()}. {done} of {total} recordings signed.'
        with st.container(key=f'row_home_{case_code}_{mode}'):
            if st.button(label, key=f'open_{case_code}_{mode}'):
                open_assessment(case_code, examiner, None, mode)
footer()
