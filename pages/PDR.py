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
model_name = st.selectbox('Select Model', options=df['Series'].unique())

if st.button('Search'):
    df = df[df['Series'] == model_name].reset_index(drop=True)
    st.dataframe(df)

    st.download_button(
        label="Download data as Excel",
        data=get_excel_file(df),
        file_name='pdr.xlsx',
        mime='xlsx',
    )