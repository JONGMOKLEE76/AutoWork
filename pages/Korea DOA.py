import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *

with open('D:/Data/PC ODM DOA return DB.bin', 'rb') as f:
    df1 = pickle.load(f)

df1['OBD'] = pd.to_datetime(df1['OBD']).dt.strftime('%Y-%m-%d')
df1 = df1.fillna('TBA').pivot_table('S/N', index=['Vendor', 'OBD', 'DOA Number'], aggfunc='count').reset_index()
df1 = df1.rename(columns={'S/N':'DOA QTY'})

with open('D:/Data/SD_raw_data.bin', 'rb') as f:
    df2 = pickle.load(f)

df2 = df2[df2['Import Biz Type']=='NCV'].pivot_table('Shipment Quantity', index=['DOA Number'], aggfunc=sum).reset_index()
df2= df2.rename(columns={'Inspection':'DOA Number', 'Shipment Quantity':'Return QTY'})

df3 = df1.merge(df2, how='left')
df3 = df3.fillna(0)
df3.sort_values(['Vendor', 'OBD'])

with open('D:/Data/PC Inspection DB.bin', 'rb') as f: # 전수검사 DB를 먼저 가져옴
    df4 = pickle.load(f)
df4 = df4[df4['DOA Number'].notnull()]

df4 = df4.pivot_table('S/N', index='DOA Number', aggfunc='count').reset_index()
df4 = df4.rename(columns={'S/N':'Inspection QTY'})

df5 = df3.merge(df4, how='left')
df5 = df5.fillna(0)
df5['Balance QTY'] = df5['DOA QTY'] - df5['Inspection QTY']
df5 = df5.convert_dtypes()
st.header('ODM Model DOA return Status')
st.dataframe(df5)