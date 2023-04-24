import streamlit as st
from AutoWork.tool import *

st.header('Quanta PDR Search!')

df = get_pdr()
model_name = st.selectbox('Select Model', options=df['Series'].unique())
if st.button('Search'):
    df = df[df['Series'] == model_name].reset_index(drop=True)
    st.dataframe(df)