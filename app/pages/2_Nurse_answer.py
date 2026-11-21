import streamlit as st
from workflow_ui import setup, select_trial, replay, answer_select, next_link, MANUAL_LABELS

store = setup('Your answer', step=2)
st.write('Listen, choose what you hear, then lock it. The AI finding stays hidden until you do.')
row = select_trial(store, key='nurse_trial')
if row:
    tid = row['trial_id']
    replay(store.recording_path(tid))
    if row['manual_locked_at']:
        st.success(f"Your answer is locked: **{MANUAL_LABELS[row['manual_label']]}**")
        next_link('pages/3_AI_finding.py', 'Next: see AI finding')
    else:
        label = answer_select('What do you hear?', key=f'manual_{tid}')
        st.caption('Locking cannot be undone for this recording. Choose Unsure if you cannot decide.')
        if st.button('Lock my answer', disabled=label is None, type='primary', icon=':material/lock:'):
            try:
                store.lock_manual(tid, label)
                st.rerun()
            except ValueError as error:
                st.error(f'Could not lock the answer: {error}')
