import streamlit as st
import time
from tool import *
import openai


st.set_page_config(layout='wide')
st.title('LG IT Div. PC ODM Dashboard')
st.subheader('Welcome to LG Electronics PC ODM Dashboard Web pages.')   

st.subheader(f'Today is :red[{datetime.date.today().strftime("%Y-%m-%d-%a")}]')
st.subheader(f'Now, you are in the week of  :red[{get_weekname(datetime.date.today())}]')


# Load your API key from an environment variable or secret management service
openai.api_key = "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"

messages = []
messages.append({'role':'user', 'content':st.text_input('Type the text here')})
if st.button('Ask GPT'):
    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    st.write(chat_response)