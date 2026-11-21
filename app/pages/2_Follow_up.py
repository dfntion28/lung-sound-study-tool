import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import (MODE_NAMES, SITE_NAMES, empty_state, footer, is_replaced, open_assessment, parse_trial,
                         session_examiner, setup, site_status, trial_words)

store = setup('Follow-up')
st.title('Follow-up')
st.markdown('<p class="prose">These recordings, saved under your examiner code, should be heard again because your '
            'first finding and the analysis did not agree, or one of them was uncertain.</p>', unsafe_allow_html=True)
examiner = session_examiner()
all_rows = store.table('rater_study')
# Blinding (D-106): another examiner's recordings and findings are not listed here.
rows = [r for r in store.review_queue() if not is_replaced(r, all_rows)
        and examiner and parse_trial(r['trial_id'])['examiner'] == examiner]
if not examiner:
    empty_state('No examiner code yet', ' Enter your examiner code on the Assessments screen to see your recordings.')
    st.page_link('Home.py', label='Go to Assessments')
elif not rows:
    empty_state('No sites need follow-up', ' A recording is listed here when your first finding and the analysis disagree, or when either is uncertain.')
for row in rows:
    info = parse_trial(row['trial_id'])
    side = 'Front' if info['site'].startswith('A') else 'Back'
    label = (f"Case {info['case']}, {side.lower()} {SITE_NAMES[info['site']].lower()}, {trial_words(info)}, "
             f"{MODE_NAMES[info['mode']].lower()}. {site_status(row)}.")
    with st.container(key=f"row_follow_{row['trial_id']}"):
        if st.button(label, key=f"follow_{row['trial_id']}"):
            open_assessment(info['case'], info['examiner'], info['round'], info['mode'], info['site'])
footer()
