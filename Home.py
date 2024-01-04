import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
import time
from tool import *
import plotly.express as px

def plot_odm_shipment_for_recent_three_years(vendor, wk_name, thismonth, ver):
    sp = monthly_sum(get_sp_from_GSCP_DB(wk_name, ver, vendor), 'Series').loc[:, thismonth:]
    sp = sp.stack().reset_index()
    sp.rename(columns={'level_1':'Year', 0:'QTY'}, inplace=True)
    sp.loc[:, ['Year', 'Month']] = sp['Year'].str.split('-', expand=True).values
    sp['Year'] = sp['Year'].astype(int)
    sp['Month'] = sp['Month'].astype(int)
    sp = sp[sp['Year']==int(thismonth.split('-')[0])]
    sp = sp.sort_values(['Year', 'Month'])
    sp['Month'] = sp['Month'].apply(lambda x:datetime.datetime.strptime('{0:02d}'.format(x), '%m').strftime('%b'))

    with open(f'D:/Data/{vendor} shipment result DB.bin', 'rb') as f:
        sr = pickle.load(f)

    cond = ((sr['Ship Year'] == int(thismonth.split('-')[0])) & (sr['Ship Month'] == int(thismonth.split('-')[-1]))) # 이번달 실적
    sr = sr[~cond] # 이번달 실적은 제외함
    cond = sr['Ship Year'] >= (int(thismonth.split('-')[0]) - 2) # 재작년부터 현재까지의 실적
    sr = sr[cond]
    sr = sr.pivot_table('Ship', index=['Series', 'Ship Year', 'Ship Month'], aggfunc='sum').fillna(0).reset_index()
    sr = sr.rename({'Ship Year':'Year', 'Ship Month':'Month', 'Ship':'QTY'}, axis=1)
    sr = sr.sort_values(['Year', 'Month'])
    sr['Month'] = sr['Month'].apply(lambda x:datetime.datetime.strptime('{0:02d}'.format(int(x)), '%m').strftime('%b'))
    result = pd.concat([sr, sp]).reset_index(drop=True)
    fig = px.histogram(result, x='Month', y='QTY', color='Year', text_auto=True, barmode='group')
    return fig.update_xaxes(categoryarray= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

st.set_page_config(layout='wide')
st.title('LG IT Div. PC ODM Dashboard')
st.subheader('IT아웃소싱 PC ODM업체의 공급 현황을 공유하는 웹싸이트 입니다.')   

st.subheader(f'오늘은 :red[{datetime.date.today().strftime("%Y-%m-%d-%a")}] 입니다.')
st.subheader(f'이번주는 :red[{get_weekname(datetime.date.today())}] 입니다.')

st.divider()
st.subheader('Quanta 최근 3개년 월별 선적 현황')

with open('D:/Data/GSCP raw data.bin', 'rb') as f:
    sp = pickle.load(f)

sp = sp[sp['Ver'] == 'Final']
lt_wk = sp['Ref'].max()
dt_obj = datetime.datetime.strptime(lt_wk[:10], '%Y-%m-%d')
thismonth = str(dt_obj.isocalendar().year) +  '-' + '{0:02d}'.format(get_month_from_date(dt_obj))

fig = plot_odm_shipment_for_recent_three_years('Quanta', lt_wk, thismonth, 'Final')
st.plotly_chart(fig, use_container_width=True)
st.subheader('Pegatron 최근 3개년 월별 선적 현황(Thin Client 포함)')
fig = plot_odm_shipment_for_recent_three_years('Pegatron', lt_wk, thismonth, 'Final')
st.plotly_chart(fig, use_container_width=True)
st.subheader('Wingtech 최근 3개년 월별 선적 현황')
fig = plot_odm_shipment_for_recent_three_years('Wingtech', lt_wk, thismonth, 'Final')
st.plotly_chart(fig, use_container_width=True)
