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
            st.caption('Use the examiner code assigned to you at every session.')
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
    st.markdown('<p class="problem" role="alert">The code was not changed because a file or finding is not saved. '
                'Save it first.</p>', unsafe_allow_html=True)
left, right = st.columns([7, 5], gap='large')

with right:
    st.subheader('New assessment')
    with st.form('new_assessment'):
        case = st.text_input('Case code', max_chars=16, placeholder='For example, S01')
        st.caption('Enter the code from the recording log, not the subject\'s name.')
        # Stored values stay BELL and DIA so older records read the same (D-097).
        modes = [m for m in MODE_ORDER if m in store.protocol['modes']]
        mode = st.radio('Stethoscope mode', modes, format_func=MODE_NAMES.get, index=modes.index('DIA'),
                        captions=[MODE_HELP[m] for m in modes], key='mode_choice')
        start = st.form_submit_button('Start assessment', type='primary')
    st.caption('Each site is recorded twice. Lift and place the chestpiece again before trial 2.')
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
        empty_state('No assessments yet', ' Enter a case code to start one.')
    for case_code, mode in sorted(groups):
        total = len(sites) * trial_count(store, dict(case=case_code, examiner=examiner, mode=mode))
        done = sum(is_complete(row) for row in groups[(case_code, mode)])
        label = f'Case {case_code}, {MODE_NAMES[mode].lower()}. {done} of {total} recordings signed.'
        with st.container(key=f'row_home_{case_code}_{mode}'):
            if st.button(label, key=f'open_{case_code}_{mode}'):
                open_assessment(case_code, examiner, None, mode)
footer()
