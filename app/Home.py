import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import PAGES, frame

# Entry script (D-112): Streamlit Cloud runs this file. It draws the masthead once and runs the chosen page,
# so switching screens keeps the masthead in place instead of clearing the whole page.
page = st.navigation([st.Page(path, title=title, url_path=url, default=url is None) for path, title, url in PAGES],
                     position='hidden')
frame(page.title)
# Continuity, not decoration: the new page body fades in once when the screen changes, never on an ordinary rerun.
# Content stays visible by default; reduced motion turns it off.
# The style element is written on every run so the page body keeps its position and Streamlit updates it in place.
turn = st.session_state.get('page_turn', 0)
if st.session_state.get('page_shown') != page.title:
    turn += 1
    st.session_state['page_shown'], st.session_state['page_turn'] = page.title, turn
st.html(f'<style>@keyframes arrive{turn}{{from{{opacity:0;}}to{{opacity:1;}}}}'
        f'.st-key-page_body{{animation:arrive{turn} .2s cubic-bezier(.16,1,.3,1);}}'
        '@media (prefers-reduced-motion:reduce){.st-key-page_body{animation:none!important;}}</style>')
with st.container(key='page_body'):
    page.run()
