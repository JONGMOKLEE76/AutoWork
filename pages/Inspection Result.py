import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *

st.title('ODM Model Korea Inspection Result!')

with open('D:/Data/PC Inspection DB.bin', 'rb') as f:
    df = pickle.load(f)

col1, col2 = st.columns(2)
with col1:
    search_year = st.selectbox('Select Year to search', options=df['Inspection Date'].dropna().dt.year.sort_values().unique())
with col2:
    search_month = st.selectbox('Select Month to search',options=df['Inspection Date'].dropna().dt.month.sort_values().unique())

if st.button('Search'):
    cond1 = df['Inspection Date'].dt.year.astype(str).str.contains(str(search_year), na=True)
    cond2 = df['Inspection Date'].dt.month.astype(str).str.contains(str(search_month), na=True)
    df_new = df[cond1 & cond2]
    df_new['Series'] = df_new['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model)
    df_new.loc[:,'Vendor'] = df_new['Series'].replace(vendor_find)
    df_new['DOA Number'].fillna('-', inplace=True)
    df_new = pd.pivot_table(df_new, index=['Inspection Date', 'Vendor','Series', 'Inspection_Reason', 'Responsibility', 'DOA Number'], 
                            columns=['Judge'], values='S/N', aggfunc='count')
    df_new = df_new.fillna(0)
    df_new = df_new.convert_dtypes()
    df_new['Total'] = df_new['NG'] + df_new['OK']
    df_new.loc[:, 'Defective Rate(%)'] = df_new['NG'] * 100 / df_new['Total']
    st.dataframe(df_new)