import pandas as pd
import streamlit as st
from workflow_ui import setup, LABELS

store = setup('Study results')
st.write('Local study results are separate from the model benchmarks. Agreement is not evidence of accuracy without an expert reference.')
metrics = store.analytics()


def fmt(value, digits=2):
    return round(value, digits) if isinstance(value, float) else value


st.subheader('Nurse and AI agreement')
cols = st.columns(3)
cols[0].metric('Paired trials', metrics['n_manual_ai'])
cols[1].metric('Agreement (%)', fmt(metrics['agreement_pct'], 1))
cols[2].metric("Cohen’s κ", fmt(metrics['cohen_kappa_manual_ai']))
st.subheader('Expert reference')
st.dataframe(pd.DataFrame([{'Answer': title, 'Paired trials': metrics[f'n_{name}_expert'],
                           'Accuracy (%)': fmt(metrics[f'accuracy_{name}_expert_pct'], 1)}
                          for name, title in (('manual', 'Nurse (before AI)'), ('ai', 'AI'), ('assisted', 'Nurse (after AI)'))]), hide_index=True)
st.write(f"Assisted vs expert Cohen’s κ: {fmt(metrics['cohen_kappa_assisted_expert'])}")
st.write(f"Right → wrong changes: {metrics['right_to_wrong_count']}")
st.caption('A right → wrong change means the manual answer matched the expert and the assisted answer did not. Metrics need at least two eligible observations; κ is undefined when there is no label variation. Unsure and Uncertain count as disagreement.')
st.subheader('AI repeatability')
repeat = metrics['repeatability']
st.write(f"Repeated case/site groups: {repeat['n_groups']} · Distinct recordings: {repeat['n_recordings']}")
st.write(f"AI self-agreement (%): {fmt(repeat['self_agreement_pct'], 1)}")
st.write(f"Fleiss κ: {fmt(repeat['fleiss_kappa'])}")
st.caption(repeat['method'] + ' Multiple nurse ratings of one recording count as one AI observation. These are descriptive local results, not Hanhong validation.')
with st.expander('Enter an expert reference later'):
    rows = store.table('rater_study')
    if rows:
        tid = st.selectbox('Coded trial', [r['trial_id'] for r in rows], key='expert_trial')
        label = st.selectbox('Expert acoustic finding', list(LABELS), format_func=LABELS.get, index=None)
        if st.button('Save expert reference', disabled=label is None):
            store.set_expert(tid, label)
            st.rerun()
    else:
        st.info('Add a coded trial before entering a reference.')
st.subheader('CSV exports')
for name in ('predictions', 'rater_study', 'review_queue'):
    st.download_button(f'Download {name.replace("_", " ")}', store.export_csv(name),
                       file_name=f'{name}.csv', mime='text/csv', key=f'export_{name}')
