import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
import time
from tool import *
import openai


st.set_page_config(layout='wide')
st.title('LG IT Div. PC ODM Dashboard')
st.subheader('Welcome to LG Electronics PC ODM Dashboard Web pages.')   

st.subheader(f'Today is :red[{datetime.date.today().strftime("%Y-%m-%d-%a")}]')
st.subheader(f'Now, you are in the week of  :red[{get_weekname(datetime.date.today())}]')

with open('C:\\Users\\paul76.lee\\AutoWork\\openapikey.txt', 'r') as f:
    key = f.read()

# Load your API key from an environment variable or secret management service
openai.api_key = key

st.markdown('-----------')
messages = []
messages.append({'role':'user', 'content':st.text_input('If you have any issue or question about your work, type the text here to ask CHATGPT')})
if st.button('Ask GPT'):
    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    st.write(chat_response)

    # system_prompot = '''
    # You are a world-class Korean-English translator and interpreter, with exceptional skills in business English interpretation and email writing.
    # You have extensive experience in this field, having worked with many Korean managers.
    #  You can express our requirements or inquiries in a sophisticated and accurate manner. From now on, please translate the sentence written in Korean into a conventional business English sentence. 
    # '''