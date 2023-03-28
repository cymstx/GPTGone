import streamlit as st
import pandas as pd
import numpy as np
import utils
import details


st.set_page_config(page_title='GPTGone', page_icon='🚀', menu_items={
    # GPTGone is a project created by 4 students in SUTD for the AI module.
    'About': details.about_text,
})


st.title('🚀 GPTGone')

intro = st.container()
with intro:
    st.markdown(
        details.introduction_text)

# enter the text to be cheked
text_to_check = st.text_area('Text to analyze')

# variable to check if the text is written by AI
written_by_ai = False

check_col, reset_col = st.columns(2)

button_pressed = check_col.button(
    'Check if written by AI', disabled=len(text_to_check) == 0, type='primary')

if button_pressed:
    # check if the text is written by AI
    written_by_ai = utils.check_if_ai(text_to_check)


if written_by_ai:
    st.warning('The text is written by AI')
    st.metric(label='AI', value='100%')

elif not written_by_ai and button_pressed:
    st.success('The text is written by a human')

if reset_col.button('Reset'):
    written_by_ai = False
    button_pressed = False
