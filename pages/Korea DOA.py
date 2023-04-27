import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *
from io import BytesIO
import xlsxwriter

def get_excel_file(df, index=True):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=index, sheet_name='Sheet1')
    writer.save()
    excel_data = output.getvalue()
    return excel_data

st.header('1. ODM Model DOA return Status')
vendor = st.radio('Select Vendor to search', options=['Quanta', 'Pegatron'])

if st.button('Search', key=1):
    with open('D:/Data/PC ODM DOA return DB.bin', 'rb') as f:
        df1 = pickle.load(f)

    df1 = df1[df1['Vendor'] == vendor]
    df1['OBD'] = pd.to_datetime(df1['OBD']).dt.strftime('%Y-%m-%d')
    df1 = df1.fillna('TBA').pivot_table('S/N', index=['Vendor', 'OBD', 'DOA Number', 'FOC Number'], aggfunc='count').reset_index()
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

    st.dataframe(df5)

st.header('2. Searh DOA List!')

col1, col2 = st.columns(2)
with col1:
    vendor = st.selectbox('Chosse the vendor', options=['Quanta', 'Pegatron'])

with open('D:/Data/PC ODM DOA return DB.bin', 'rb') as f:
        df1 = pickle.load(f)
df1 = df1[df1['Vendor']==vendor]

with col2:
    doa_num = st.selectbox('Choose the DOA number', options=df1['DOA Number'].unique())

if st.button('Search', key=2):
     st.dataframe(df1[df1['DOA Number']==doa_num].reset_index(drop=True))
     st.download_button(
        label="Download data as Excel",
        data=get_excel_file(df1[df1['DOA Number']==doa_num].reset_index(drop=True), False),
        file_name='doa_list.xlsx',
        mime='xlsx')