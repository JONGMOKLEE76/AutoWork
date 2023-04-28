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


tab1, tab2, tab3 = st.tabs(["Quanta", "Pegatron", "Wingtech"])

with tab1:
    st.subheader('1. ODM Forecast')
    ver = st.selectbox('Which version you would like to choose?', ('Final', 'Latest', 'ODM Release'), key='1')
    wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
    week = st.selectbox('Which week you would like to choose?', wklist, key='2')

    if st.button('Search', key='3'):
        try:
            demand = get_sp_from_GSCP_DB(week, ver, 'Quanta', -1)
            st.write('Monthly Summary')
            st.dataframe(monthly_sum(demand, ['Series', 'Region']).reset_index())
            st.write('Weekly Demand')
            st.dataframe(demand)
        except:
            st.write('Not Available')
    
    st.subheader('2. SP PO gap status for Frozen weeks')
    confirm_period = st.radio('Choose the confirm period from this week', options=[3,4,5], index=1)
    gap = get_sp_po_gap(['Quanta'], confirm_period+1)
    gap = gap[gap[('SUM', 'GAP')] != 0]
    st.table(gap)
    st.download_button(
    label="Download data as Excel",
    data=get_excel_file(gap, True),
    file_name='sp_po_gap.xlsx',
    mime='xlsx')

with tab2:
    st.subheader('1. ODM Forecast')
    ver = st.selectbox('Which version you would like to choose?', ('Final', 'Latest', 'ODM Release'), key='4')
    wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
    week = st.selectbox('Which week you would like to choose?', wklist, key='5')

    if st.button('Search', key='6'):
        try:
            demand = get_sp_from_GSCP_DB(week, ver, 'Pegatron', -1)
            st.write('Monthly Summary')
            st.dataframe(monthly_sum(demand, ['Series', 'Region']).reset_index())
            st.write('Weekly Demand')
            st.dataframe(demand)
        except:
            st.write('Not Available')
    
    st.subheader('2. SP PO gap status for Frozen weeks')
    confirm_period = st.radio('Choose the confirm period from this week', options=[3,4,5], index=1, key='7')
    gap = get_sp_po_gap(['Pegatron'], confirm_period+1)
    gap = gap[gap[('SUM', 'GAP')] != 0]
    st.table(gap)
    if gap.shape[0] != 0:
        st.download_button(
        label="Download data as Excel",
        data=get_excel_file(gap, True),
        file_name='sp_po_gap.xlsx',
        mime='xlsx')

with tab3:
    st.subheader('1. ODM Forecast')
    ver = st.selectbox('Which version you would like to choose?', ('Final', 'Latest', 'ODM Release'), key='8')
    wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
    week = st.selectbox('Which week you would like to choose?', wklist, key='9')

    if st.button('Search', key='10'):
        try:
            demand = get_sp_from_GSCP_DB(week, ver, 'Wingtech', -1)
            st.write('Monthly Summary')
            st.dataframe(monthly_sum(demand, ['Series', 'Region']).reset_index())
            st.write('Weekly Demand')
            st.dataframe(demand)
        except:
            st.write('Not Available')

    st.subheader('2. SP PO gap status for Frozen weeks')
    confirm_period = st.radio('Choose the confirm period from this week', options=[3,4,5], index=1, key='11')
    gap = get_sp_po_gap(['Wingtech'], confirm_period+1)
    gap = gap[gap[('SUM', 'GAP')] != 0]
    st.table(gap)
    if gap.shape[0] != 0:
        st.download_button(
        label="Download data as Excel",
        data=get_excel_file(gap, True),
        file_name='sp_po_gap.xlsx',
        mime='xlsx')

