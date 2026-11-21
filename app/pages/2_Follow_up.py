import streamlit as st

from workflow_ui import (MODE_NAMES, SITE_NAMES, empty_state, footer, is_replaced, open_assessment, parse_trial, setup,
                         site_status, trial_words)

store = setup('Follow-up')
st.title('Follow-up')
st.markdown('<p class="prose">An examiner or expert should hear these recordings again.</p>',
            unsafe_allow_html=True)
all_rows = store.table('rater_study')
rows = [r for r in store.review_queue() if not is_replaced(r, all_rows)]
if not rows:
    empty_state('No sites need follow-up', ' A recording appears here after analysis if it needs another listen.')
for row in rows:
    info = parse_trial(row['trial_id'])
    side = 'Front' if info['site'].startswith('A') else 'Back'
    label = (f"Case {info['case']}, {side.lower()} {SITE_NAMES[info['site']].lower()}, {trial_words(info)}, "
             f"{MODE_NAMES[info['mode']].lower()}, examiner {info['examiner']}. {site_status(row)}.")
    with st.container(key=f"row_follow_{row['trial_id']}"):
        if st.button(label, key=f"follow_{row['trial_id']}"):
            open_assessment(info['case'], info['examiner'], info['round'], info['mode'], info['site'])
footer()
