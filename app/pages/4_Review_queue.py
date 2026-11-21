import streamlit as st
from workflow_ui import setup, replay, empty_state, LABELS, MANUAL_LABELS

store = setup('Review', step=4)
st.write('Recordings where your locked answer and the AI differ, or where either was unsure, are listed here.')
rows = store.review_queue()
if not rows:
    empty_state('Nothing to review', ' When your answer and the AI finding differ, the recording appears here.')
else:
    st.dataframe([{'Recording': r['trial_id'], 'Site': r['site_code'], 'You': MANUAL_LABELS[r['manual_label']],
                   'AI': LABELS[store.get_ai(r['trial_id'])['finding']]} for r in rows], hide_index=True)
    tid = st.selectbox('Recording to review', [r['trial_id'] for r in rows])
    row = next(r for r in rows if r['trial_id'] == tid)
    result = store.get_ai(tid)  # API rechecks the manual lock.
    st.markdown(f"You: <span class='chip'>{MANUAL_LABELS[row['manual_label']]}</span> &nbsp; "
                f"AI: <span class='chip'>{LABELS[result['finding']]}</span>", unsafe_allow_html=True)
    if row['assisted_label']:
        st.caption(f"Your answer after AI: {MANUAL_LABELS[row['assisted_label']]}")
    current, previous = st.columns(2)
    with current:
        replay(store.recording_path(tid), 'This recording')
    with previous:
        path = store.previous_recording(row['case_code'], row['site_code'], store.trial_number(tid))
        replay(path, 'Earlier recording, same site')
        if path:
            st.caption(path.stem)
    st.caption('Compare by listening. The app does not raise automatic change alerts.')
st.page_link('pages/1_Record.py', label='Record the next site', icon=':material/mic:')
