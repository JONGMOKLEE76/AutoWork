import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *
from io import BytesIO
import xlsxwriter

st.header('Quanta PDR Search!')

def get_excel_file(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    excel_data = output.getvalue()
    return excel_data

df = get_pdr().drop(columns=['Model', 'Bar Code', 'Marketing Spec.', 'PDR No.', 'Disabled Status', 'Grade', 'Product Variance', 'Project Code', 'Shipping Date', 'Production Type', 'Producing Center'])
col1, col2 = st.columns(2)
with col1:
    model = st.selectbox('Select Model', options=df['Series'].unique())
with col2:
    suffix = st.selectbox('Choose the Model.Suffix Name',options=df[df['Series']==model]['Model.Suffix'].unique())

if st.button('Search'):
    df = df[df['Series'] == model].reset_index(drop=True)
    st.dataframe(df)

    st.download_button(
        label="Download data as Excel",
        data=get_excel_file(df),
        file_name='pdr.xlsx',
        mime='xlsx',
    )
    st.header('FAI Spec Check!')
    st.table(df[df['Model.Suffix']==suffix][['Sales Model.Suffix', 'BASE UNIT', 'KEYBOARD', 'ADAPTER', 'OS TYPE', 'ACCESSORY KIT', 'PACKING', 'CUSTOMER', 'USB MOUSE', 'Nation', 'Image ID']].T)



    