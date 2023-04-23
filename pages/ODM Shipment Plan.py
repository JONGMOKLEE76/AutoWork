import streamlit as st
from tool import *

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

with tab3:
    st.subheader('1. ODM Forecast')
    ver = st.selectbox('Which version you would like to choose?', ('Final', 'Latest', 'ODM Release'), key='7')
    wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
    week = st.selectbox('Which week you would like to choose?', wklist, key='8')

    if st.button('Search', key='9'):
        try:
            demand = get_sp_from_GSCP_DB(week, ver, 'Wingtech', -1)
            st.write('Monthly Summary')
            st.dataframe(monthly_sum(demand, ['Series', 'Region']).reset_index())
            st.write('Weekly Demand')
            st.dataframe(demand)
        except:
            st.write('Not Available')