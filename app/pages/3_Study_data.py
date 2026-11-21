import fresh_modules

fresh_modules.refresh()

from html import escape

import streamlit as st

from workflow_ui import heading, LABELS, MODE_NAMES, SITE_NAMES, footer, parse_trial, setup, trial_words

store = setup('Study data')
heading('Study data')
st.markdown('<p class="prose">These results come from this tool\'s recordings, not from the model benchmark.</p>', unsafe_allow_html=True)
metrics = store.analytics()


def fmt(value, digits=2):
    return round(value, digits) if isinstance(value, float) else value


def entries(pairs):
    st.markdown(''.join(f'<dl class="entry"><dt>{escape(str(k))}</dt><dd>{escape(str(v))}</dd></dl>' for k, v in pairs),
                unsafe_allow_html=True)


st.subheader('Examiner and analysis agreement')
entries((('Paired recordings', metrics['n_manual_ai']), ('Agreement (%)', fmt(metrics['agreement_pct'], 1)),
         ('Cohen’s κ', fmt(metrics['cohen_kappa_manual_ai']))))
st.caption('Agreement does not show accuracy; accuracy needs the expert reference.')
st.subheader('Against the expert reference')
entries([(f"{title} (n = {metrics[f'n_{name}_expert']})", fmt(metrics[f'accuracy_{name}_expert_pct'], 1))
         for name, title in (('manual', 'Examiner, first entry'), ('ai', 'Acoustic analysis'),
                             ('assisted', 'Examiner, final signed'))])
st.caption('Each value is the percentage of recordings that matched the expert reference.')
entries((('Final finding against expert, Cohen’s κ', fmt(metrics['cohen_kappa_assisted_expert'])),
         ('Correct first findings later changed', metrics['right_to_wrong_count'])))
st.subheader('Analysis repeatability')
repeat = metrics['repeatability']
entries((('Repeated case and site groups', repeat['n_groups']), ('Distinct recordings', repeat['n_recordings']),
         ('Self-agreement (%)', fmt(repeat['self_agreement_pct'], 1)), ('Fleiss’ κ', fmt(repeat['fleiss_kappa']))))
with st.expander('How these measures are calculated'):
    st.markdown('- Cohen’s κ is agreement corrected for chance. It is undefined when every finding is the same.\n'
                '- Each measure needs at least two eligible recordings.\n'
                '- "Unable to determine" and "No clear pattern" are counted as disagreement.\n'
                '- Self-agreement is the percentage of recording pairs from the same examiner, case and site on which '
                'the analysis gave the same result.\n'
                '- Fleiss’ κ corrects self-agreement for chance and needs at least two repeated case and site groups.\n'
                '- These results describe recordings saved in this tool. They do not validate the analysis on the '
                'Hanhong stethoscope.')
st.subheader('Expert reference')
rows = store.table('rater_study')
if rows:
    by_id = {r['trial_id']: r for r in rows}


    def recording_name(trial_id):
        info = parse_trial(trial_id)
        side = 'front' if info['site'].startswith('A') else 'back'
        saved = LABELS.get(by_id[trial_id]['expert_reference'], 'none')
        return (f"Case {info['case']}, {side} {SITE_NAMES[info['site']].lower()}, {trial_words(info)}, "
                f"{MODE_NAMES[info['mode']].lower()}, examiner {info['examiner']}. Reference: {saved}")


    saved_for = st.session_state.pop('expert_saved', None)
    if saved_for:
        st.success(f'Expert reference saved for {saved_for}')
    trial = st.selectbox('Coded recording', list(by_id), format_func=recording_name, key='expert_trial')
    label = st.selectbox('Expert acoustic finding', list(LABELS), format_func=LABELS.get, index=None)
    if st.button('Save expert reference', disabled=label is None, type='primary'):
        try:
            store.set_expert(trial, label)
            st.session_state['expert_saved'] = recording_name(trial).split('. Reference')[0] + '.'
            st.rerun()
        except ValueError as error:
            st.error(f'The reference was not saved. {error}')
else:
    st.caption('Expert references can be entered after the first recording is saved.')
st.subheader('Exports')
titles = dict(predictions='Analysis results', rater_study='Examiner records', review_queue='Follow-up list')
for name, title in titles.items():
    st.download_button(f'Download {title} (CSV)', store.export_csv(name), file_name=f'{name}.csv', mime='text/csv',
                       key=f'export_{name}')
footer()
