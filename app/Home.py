from collections import defaultdict

import streamlit as st

from workflow_ui import (MODE_NAMES, clean_code, empty_state, footer, is_complete, open_assessment,
                         parse_trial, setup)

store = setup('Assessments')
st.title('Assessments')
sites = store.protocol['site_codes']
left, right = st.columns([7, 5], gap='large')

with right:
    st.subheader('New assessment')
    with st.form('new_assessment'):
        case = st.text_input('Case code', max_chars=16)
        examiner = st.text_input('Examiner code', max_chars=16, value=st.session_state.get('examiner', ''))
        with st.container(key='mode_row'):
            mode = st.radio('Stethoscope mode', store.protocol['modes'], horizontal=True, format_func=MODE_NAMES.get,
                            index=store.protocol['modes'].index('DIA'))
        round_number = int(st.number_input('Trial', min_value=1, max_value=99, value=1, step=1))
        start = st.form_submit_button('Start assessment', type='primary')
    st.caption('Trial 1 is the first recording at each site. Trial 2 is the repeat after lifting and replacing the chestpiece.')
    if start:
        case_code, examiner_code = clean_code(case), clean_code(examiner)
        if not case_code or not examiner_code:
            st.error('Enter both codes. Each starts with a letter and uses only letters and numbers.')
        else:
            st.session_state['examiner'] = examiner_code
            open_assessment(case_code, examiner_code, round_number, mode)

with left:
    st.subheader('Open a saved assessment')
    groups = defaultdict(list)
    for row in store.table('rater_study'):
        info = parse_trial(row['trial_id'])
        groups[(info['case'], info['examiner'], info['round'], info['mode'])].append(row)
    if not groups:
        empty_state('No assessments yet', ' Start one with a case code and your examiner code.')
    for key in sorted(groups):
        case_code, examiner_code, round_number, mode = key
        done = sum(is_complete(row) for row in groups[key])
        label = (f'Case {case_code}, trial {round_number}, {MODE_NAMES[mode].lower()}, examiner {examiner_code}. '
                 f'{done} of {len(sites)} sites signed.')
        with st.container(key=f'row_home_{case_code}_{examiner_code}_{round_number}_{mode}'):
            if st.button(label, key=f'open_{case_code}_{examiner_code}_{round_number}_{mode}'):
                open_assessment(case_code, examiner_code, round_number, mode)
footer()
