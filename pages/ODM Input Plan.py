import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
from tool import *
import streamlit as st
import plotly.express as px
from io import BytesIO
import xlsxwriter

def get_excel_file(df, index=True):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=index, sheet_name='Sheet1')
    writer.save()
    excel_data = output.getvalue()
    return excel_data

st.header('Quanta Input Plan')

st.subheader('1. Search Input Plan for certain model')
with open('D:/Data/Quanta Input Plan.bin', 'rb') as f:
    input_df = pickle.load(f)

ref_date = st.selectbox('Choose the reference date to search' , options=input_df['Created_at'].sort_values(ascending=False).unique())

df1 = input_df[input_df['Created_at'] == ref_date]
df1['Series'] = df1['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model)
model_names = st.multiselect('Choose the model, If you want to see all model, do NOT choose', options=df1['Series'].unique())

if st.button('Search', key=1):
    if model_names:
        df1 = df1[df1['Series'].isin(model_names)]
    st.dataframe(df1.pivot_table('QTY', index=['Series', 'Mapping Model.Suffix', 'Quanta P/N'], columns='LG Week', aggfunc=sum).fillna(0))

st.subheader('2. Compare desired Input plan with other reference date\'s plan')
date1 = st.selectbox('1) Choose the reference Input date to search' , options=input_df['Created_at'].sort_values(ascending=False).unique())
date2 = st.selectbox('2) Choose the reference Input date to compare' , options=input_df['Created_at'].sort_values(ascending=False).unique())

if st.button('Search', key=2):
    df2 = input_df[input_df['Created_at'].isin([date1, date2])]
    df2['Series'] = df2['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model)
    compare_table = df2.groupby(['Created_at', 'Series', 'LG Week']).sum(numeric_only=True).unstack(['LG Week', 'Created_at']).fillna(0).sort_index(axis=1)
    compare_table.loc['Total'] = compare_table.sum()
    st.table(compare_table)

    st.download_button(
        label="Download data as Excel",
        data=get_excel_file(compare_table, True),
        file_name='input_compare.xlsx',
        mime='xlsx')

st.subheader('3. See how input plan is changed')

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Select the start date you want to search from')
with col2:
    end_date = st.date_input('Select the end date you want to search to')

if st.button('Plot the Input Plan Variation!'):
    with open('D:/Data/Quanta Input Plan.bin', 'rb') as f:
        input_df = pickle.load(f)
    day_list = [start_date + datetime.timedelta(i) for i in range((end_date - start_date).days + 1)]
    input_df = input_df.sort_values('LG Week')
    fig = px.bar(input_df[input_df['Created_at'].isin(day_list)], x='Created_at', y='QTY', color='LG Week', width=1000, height=500)
    fig.update_layout(yaxis_showline=False)
    st.plotly_chart(fig, use_container_width=True)

