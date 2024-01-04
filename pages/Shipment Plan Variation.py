import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *
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

st.subheader('ODM Shipment Plan Variation!')

with open('D:/Data/GSCP raw data.bin', 'rb') as f:
    df = pickle.load(f)

vendor = st.radio('Choose the vendor', options=['Pegatron', 'Quanta', 'Wingtech'])

df = df[df['From Site'] == vendor]

startweek = st.selectbox('Choose the start week you want search', options=df['Ref'].sort_values(ascending=False).unique()) # 금주 이후 몇주 동안의 변동을 조회할 지 설정
search_window = st.number_input('How long the width of window you want to see?', min_value=1, max_value=52, value=20, step=1) # 금주 이후 몇주 동안의 변동을 조회할 지 설정
period = st.number_input('How many weeks do you want to keep track from the starting week?', min_value=1, max_value=52, value=10, step=1) # 미래 몇주 간의 변동을 조회할 지 설정
ver = st.radio('Choose the version', options=['ODM Release', 'Final', 'Latest']) # 물동 변동을 조회할 version 설정(Latest, Final, ODM Release)
model_names = st.multiselect('Select the model name you want to see. If you want to see all model, do NOT choose', options=df['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model).unique()) # Forecast 변동을 조회할 모델 설정(.로 하면 모든 모델)
site_names = st.multiselect('Select the site name you want to see. If you want to see all country, do NOT choose', options=df['To Site'].unique())
target_period = [get_weekname_from(startweek, i) for i in range(search_window)]
variation_period = [get_weekname_from(startweek, i) for i in range(period)]

if st.button('Generate Chart!'):
    try:
        df['Series'] = df['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model)
        if model_names:
            print(model_names)
            df = df[df['Series'].isin(model_names)] # 조회할 모델로만 Sorting 함

        # # 데이터프레임 추출 조건 설정
        c1 = df['Updated_at'].isin(df[['Ref', 'Ver', 'Updated_at']].drop_duplicates().groupby(['Ref', 'Ver'])['Updated_at'].max().unstack()[ver]) # 찾고자 하는 ver별의 가장 최근 update 된 시간으로 조회하기 위한 조건
        c2 = (df['Ver'] == ver) # 조회할 Version 만 sorting 하기 위한 조건
        c3 = df['Ref'].isin(variation_period) # Forecast 변동 이력을 조회할 구간만 sorting 하기 위한 조건
        if site_names:
            c4 = df['To Site'].isin(site_names)
        else:
            c4 = c3

        df = df[c1 & c2 & c3 & c4].set_index(['Ref', 'Series', 'Ver', 'Updated_at', 'Frozen', 'From Site', 'Mapping Model.Suffix', 'To Site'])[target_period].reset_index()
        df1 = df.groupby('Ref').sum()
        for wk in df1.index:
            if (datetime.date.fromisoformat(wk[:10]).isocalendar().year, get_month_from_date(datetime.date.fromisoformat(wk[:10]))) >= (move_month(datetime.date.fromisoformat(startweek[:10]).isocalendar()[0], get_month_from_date(datetime.date.fromisoformat(startweek[:10])), 2)):
                df1.loc[wk, startweek:get_weekname_from(get_weeklist_for_certain_month(get_lastmonth(wk)[0], get_lastmonth(wk)[1])[0], -1)] = df1.loc[get_weekname_from(wk, -1), startweek:get_weekname_from(get_weeklist_for_certain_month(get_lastmonth(wk)[0], get_lastmonth(wk)[1])[0], -1)]
        df1 = monthly_sum(df1, 'Ref').stack().reset_index()
        df1.rename(columns={'level_1':'Month', 0:'QTY'}, inplace=True)
        fig = px.bar(df1, x='Ref', y='QTY', color='Month', width=500, height=600)
        fig.update_xaxes(tickangle=70)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monthly_sum(df, ['Ref', 'Series']))

        st.download_button(
            label="Download data as Excel",
            data=get_excel_file(monthly_sum(df, ['Ref', 'Series']), True),
            file_name='Demand_variation_list.xlsx',
            mime='xlsx')
        
    except:
        st.write('Not Available')