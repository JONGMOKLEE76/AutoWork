import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
from tool import *
import streamlit as st

st.subheader('Quanta Input Plan')

with open('D:/Data/Quanta Input Plan.bin', 'rb') as f:
    input_df = pickle.load(f)

ref_date = st.selectbox('Choose the reference date' , options=input_df['Created_at'].sort_values(ascending=False).unique())

input_df = input_df[input_df['Created_at'] == ref_date]
input_df['Series'] = input_df['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model)
model_names = st.multiselect('Choose the model', options=input_df['Series'].unique())
if st.button('Search'):
    input_df = input_df[input_df['Series'].isin(model_names)]
    st.dataframe(input_df.pivot_table('QTY', index=['Series', 'Mapping Model.Suffix'], columns='LG Week', aggfunc=sum).fillna(0))