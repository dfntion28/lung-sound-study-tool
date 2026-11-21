import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import (heading, MODE_NAMES, SITE_NAMES, empty_state, footer, is_replaced, open_assessment, parse_trial,
                         session_examiner, setup, site_status, trial_words)

store = setup('Follow-up')
heading('Follow-up')
st.markdown('<p class="prose">Each recording here differs from its analysis or from the other recording at its site, '
            'or is uncertain.</p>', unsafe_allow_html=True)
examiner = session_examiner()
all_rows = store.table('rater_study')
# Blinding (D-106): another examiner's recordings and findings are not listed here.
rows = [r for r in store.review_queue() if not is_replaced(r, all_rows)
        and examiner and parse_trial(r['trial_id'])['examiner'] == examiner]
if not examiner:
    empty_state('No examiner code yet', ' Enter your code on the Assessments screen.')
    st.page_link('pages/0_Assessments.py', label='Go to Assessments')
elif not rows:
    empty_state('No sites need follow-up', ' A recording is listed here when it should be heard again.')
for row in rows:
    info = parse_trial(row['trial_id'])
    side = 'Front' if info['site'].startswith('A') else 'Back'
    label = (f"Case {info['case']}, {side.lower()} {SITE_NAMES[info['site']].lower()}, {trial_words(info)}, "
             f"{MODE_NAMES[info['mode']].lower()}. {site_status(row)}.")
    with st.container(key=f"row_follow_{row['trial_id']}"):
        if st.button(label, key=f"follow_{row['trial_id']}"):
            open_assessment(info['case'], info['examiner'], info['round'], info['mode'], info['site'])
footer()
