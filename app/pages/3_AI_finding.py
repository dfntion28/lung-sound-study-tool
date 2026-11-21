import streamlit as st
from workflow_ui import setup, select_trial, get_model, show_finding, answer_select, next_link, MANUAL_LABELS

store = setup('AI finding', step=3)
row = select_trial(store, locked=True, key='ai_trial')
if row:
    tid = row['trial_id']
    st.markdown(f"Your locked answer: <span class='chip'>{MANUAL_LABELS[row['manual_label']]}</span>",
                unsafe_allow_html=True)
    result = store.get_ai(tid)
    if result is None and st.button('Show AI finding', type='primary', icon=':material/visibility:'):
        try:
            with st.spinner('Analyzing the recording…'):
                result = store.analyze_trial(tid, get_model())
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'The AI could not analyze this recording: {error}. Try recording the site again.')
    if result:
        show_finding(result, store.recording_path(tid))
        if result['finding'] == 'Q':
            next_link('pages/1_Record.py', 'Record this site again')
        else:
            label = answer_select('After seeing the AI, what is your answer now?', key=f'assisted_{tid}')
            if st.button('Save answer', disabled=label is None, type='primary'):
                try:
                    store.set_assisted(tid, label)
                    st.success('Saved. Your first answer stays locked and unchanged.')
                except ValueError as error:
                    st.error(f'Your answer was not saved. {error}')
            current = next(r for r in store.table('rater_study') if r['trial_id'] == tid)
            if current['assisted_label']:
                st.caption(f"Saved answer after AI: {MANUAL_LABELS[current['assisted_label']]}")
            next_link('pages/4_Review_queue.py', 'Next: review')
            st.page_link('pages/1_Record.py', label='Record the next site', icon=':material/mic:')
