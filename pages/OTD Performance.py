import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tool import *

tab1, tab2, tab3 = st.tabs(["Quanta", "Pegatron", "Wingtech"])

with tab1:
    st.subheader('Quanta Weekly OTD Result!')

    with open('D:/Data/OTD_result_db.bin', 'rb') as f:
        otd = pickle.load(f)
    
    otd_week = otd.pivot_table(['Target', 'Ship'], index='Ref', aggfunc=sum)
    otd_week['OTD(Result)'] = round(otd_week['Ship'] / otd_week['Target'] * 100, 1)
    layout = go.Layout(xaxis={'title':'Week'}, yaxis={'title':'OTD(%)'} )
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Bar(x=otd_week.reset_index()['Ref'], y=otd_week.reset_index()['OTD(Result)'])
    )
    fig.add_trace(
        go.Line(x=otd_week.reset_index()['Ref'], y=otd_week.reset_index()['OTD(Result)'])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('OTD Raw Data')
    otd_model = otd.pivot_table(['Target', 'Ship'], index=['Ref', 'Series'], aggfunc=sum)
    otd_model['OTD(Result)'] = round(otd_model['Ship'] / otd_model['Target'] * 100, 1)
    week = st.selectbox('Choose the week!', options=otd_model.reset_index()['Ref'].sort_values(ascending=False).unique())
    st.dataframe(otd_model.loc[week])
