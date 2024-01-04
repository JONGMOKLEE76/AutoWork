import sys
sys.path.append('C:\\Users\\paul76.lee\\AutoWork\\')
import streamlit as st
from tool import *
from io import BytesIO
import xlsxwriter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_excel_file(df, index=True):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=index, sheet_name='Sheet1')
    writer.save()
    excel_data = output.getvalue()
    return excel_data

def sp_get(ver, vendor, week_name):
    with open('D:/Data/GSCP raw data.bin', 'rb') as f:
        sp_all = pickle.load(f)
    # GSCP ver과 기준시점 주차와 vendor명으로 1차 추출
    c1 = (sp_all['Ver'] == ver)
    c2 = (sp_all['Ref'] == week_name)
    c3 = (sp_all['From Site'].str.contains(vendor))
    sp = sp_all[c1 & c2 & c3].reset_index(drop=True).copy()
    if sp.shape[0] != 0:
        sp = sp[sp['Updated_at'].isin(sp[['Ref', 'Ver', 'Updated_at']].drop_duplicates().groupby(['Ref', 'Ver'])['Updated_at'].max().unstack()[ver])].reset_index(drop=True)
        sp = sp[['Ref', 'Mapping Model.Suffix', 'To Site'] + get_weeklist(sp)].fillna(0)
        sp = sp.groupby(['Ref', 'Mapping Model.Suffix', 'To Site']).sum(numeric_only=True)
        min_wk = sp.sum() > 0
        min_wk = min_wk[min_wk == True].index.min()
        sp = sp.loc[:, min_wk:]
    return sp

tab1, tab2, tab3 = st.tabs(["Quanta", "Pegatron", "Wingtech"])

with tab1:
    st.header('물동 변동 현황')
    ver = st.selectbox('SP의 Version을 선택하세요.', ('Final', 'Latest', 'ODM Release'), key='100')
    planweeks = get_weekname(datetime.date.today())
    planweeks = [get_weekname_from(planweeks, i) for i in range(-12, 1)]
    col1 = ['Ref', 'Mapping Model.Suffix', 'To Site']

    for i, wk_name in enumerate(tqdm(planweeks)):
        if i == 0: # 누적할 sp의 초기값 셋팅
            acc_sp = sp_get(ver, 'Quanta', wk_name)
        else:
            sp = sp_get(ver, 'Quanta', wk_name)
            if sp.shape[0] == 0:
                continue
            deleted_wk = []
            for item in acc_sp.columns:
                if item not in sp.columns:
                    deleted_wk.append(item)
                    
            org_idx = acc_sp.reset_index().set_index(['Mapping Model.Suffix', 'To Site']).index
            new_idx = sp.reset_index().set_index(['Mapping Model.Suffix', 'To Site']).index
            intersecting_idx = org_idx.intersection(new_idx)
            diff_idx_org = org_idx.difference(intersecting_idx) # 첫 번째 인덱스에서 2번째 인덱스를 뺀 차집합 인덱스
            
            # 누적 데이터프레임에서 동일 index값이 가장 최근에 언제 주차에 있었는지 참조하기 위한 Series
            find_most_recent_week = acc_sp.reset_index()[col1].groupby(['Mapping Model.Suffix', 'To Site'])['Ref'].max()
            # 누적 데이터프레임에서 신규SP에 없는 row 중에서 가장 최근 주차의 행을 지정하기 위한 멀티인덱스를 만들기 위한 데이터프레임1
            idx_frame1 = find_most_recent_week[diff_idx_org].reset_index()[col1]
            m_idx1 = pd.MultiIndex.from_frame(idx_frame1)
            add_df = acc_sp.loc[m_idx1, :get_weekname_from(wk_name, -2)] # 이전주차까지의 값만 슬라이싱하여 추가할 df를 만듬
            for w in find_most_recent_week[diff_idx_org]:  # 누적 데이터프레임에서 차집합 인덱스를 반복하면서
                if w < get_weekname_from(wk_name, -1):     # 해당 최근주차가 지난주보다 작을 때
                    add_df.loc[w, w:] = 0                  # 해당 최근주차부터 이후 주차의 값은 '0' 으로 만듬(왜냐하면, 기존값은 실적이 아니므로)
            add_df = add_df[add_df.sum(axis=1) > 0]
            
            # deleted_wk 에 값이 있을 경우 df에 이전주차까지의 누적 데이터프레임의 삭제된 주차들의 선적 실적을 추가하는 작업
            if len(deleted_wk) > 0:
                # 누적 데이터프레임과 신규SP와 공통 row중에서 가장 최근 주차의 행을 지정하기 위한 멀티인덱스를 만들기 위한 데이터프레임1
                idx_frame2 = find_most_recent_week[intersecting_idx].reset_index()[col1]
                idx_frame3 = idx_frame2.copy()
                idx_frame3['Ref'] = wk_name
                m_idx2 = pd.MultiIndex.from_frame(idx_frame2)
                m_idx3 = pd.MultiIndex.from_frame(idx_frame3)
                sp.loc[m_idx3, deleted_wk] = acc_sp.loc[m_idx2, deleted_wk].values
                
            sp = pd.concat([add_df, sp]).reset_index()
            sp['Ref'] = wk_name
            sp = sp.set_index(col1)
            acc_sp = pd.concat([acc_sp, sp])

    df = acc_sp.reset_index().fillna(0)
    df.insert(1, 'Series', df['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model))
    df.insert(3, 'Region', df['To Site'].replace(site_map).replace(country_map))
    df = monthly_sum(df, ['Ref', 'Series', 'Region'])
    df.columns.name = 'Month'
    df = df.loc[:, df.sum() > 0].stack().reset_index()
    df.rename({0:'QTY'}, axis=1, inplace=True)

    st.subheader('1. 최근 4주간의 월별 물동 분포')
    fig = go.Figure()
    clr = px.colors.sequential.Reds
    for wk in df['Ref'].unique()[-4:]:
        df_wk = df[df['Ref'] == wk]
        fig.add_trace(go.Histogram(histfunc='sum', x=df_wk['Month'], y=df_wk['QTY'], name=wk, )) # 데이터프레임의 각 row의 값을 누적하여 Bar graph를 나타날 때 사용
    fig.update_layout(colorway=clr, width=1000, barmode='group')
    fig.update_xaxes(nticks=len(df_wk['Month'].unique()) + 1, tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('2. 최근 12주간의 SP 변동 현황')
    fig = px.histogram(df, x='Ref', y='QTY', barmode='relative', text_auto='QTY')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h6> - 월별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Month', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 모델별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Series', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 지역별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Region', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 모델별 월별 물동 분포의 변동 </h6>", unsafe_allow_html=True )
    fig = make_subplots(rows=(len(df['Series'].unique()) + 3) // 4, cols=4, subplot_titles=df['Series'].unique(), shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.05)
    i = 0
    row_num = 1
    col_num = 1
    for sr in df['Series'].unique():
        df_sr = df[df['Series'] == sr]
        for mon in df_sr['Month'].unique():
            df1 = df_sr[df_sr['Month'] == mon]
            fig.add_trace(go.Histogram(histfunc='sum', x=df1['Ref'], y=df1['QTY'], name=mon, hovertext=df1['Month']), row=row_num, col=col_num)
        i += 1
        row_num = i // 4 + 1
        col_num = i % 4 + 1
    fig.update_layout(barmode='relative', height=1000, showlegend=False)
    fig.update_xaxes(tickangle=-65)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h6> - 모델별 지역별 물동 분포의 변동 </h6>", unsafe_allow_html=True )
    fig = make_subplots(rows=(len(df['Series'].unique()) + 3) // 4, cols=4, subplot_titles=df['Series'].unique(), shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.05)
    i = 0
    row_num = 1
    col_num = 1
    for sr in df['Series'].unique():
        df_sr = df[df['Series'] == sr]
        for reg in df_sr['Region'].unique():
            df1 = df_sr[df_sr['Region'] == reg]
            fig.add_trace(go.Histogram(histfunc='sum', x=df1['Ref'], y=df1['QTY'], name=mon, hovertext=df1['Region']), row=row_num, col=col_num)
        i += 1
        row_num = i // 4 + 1
        col_num = i % 4 + 1
    fig.update_layout(barmode='relative', height=1000, showlegend=False)
    fig.update_xaxes(tickangle=-65)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('1. ODM Forecast')
    ver = st.selectbox('SP의 Version을 선택하세요.', ('Final', 'Latest', 'ODM Release'), key='1')
    wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
    week = st.selectbox('몇 주차의 SP를 검색할지 선택하세요.', wklist, key='2')

    if st.button('검색', key='3'):
        try:
            demand = get_sp_from_GSCP_DB(week, ver, 'Quanta', -1)
            st.write('월별 물동 요약')
            st.dataframe(monthly_sum(demand, ['Series', 'Region']).reset_index())
            st.write('주별 물동')
            st.dataframe(demand)
        except:
            st.write('자료가 없습니다.')
    
    st.subheader('2. 확정구간에서 SP와 PO의 차이 내역')
    confirm_period = st.radio('확정 구간을 선택하세요.', options=[3,4,5], index=1)
    gap = get_sp_po_gap(['Quanta'], confirm_period+1)
    gap = gap[gap[('SUM', 'GAP')] != 0]
    st.table(gap)
    if gap.shape[0] != 0:
        st.download_button(
        label="Download data as Excel",
        data=get_excel_file(gap, True),
        file_name='sp_po_gap.xlsx',
        mime='xlsx')

       
with tab2:
    st.header('물동 변동 현황')
    ver = st.selectbox('SP의 Version을 선택하세요.', ('Final', 'Latest', 'ODM Release'), key='101')
    planweeks = get_weekname(datetime.date.today())
    planweeks = [get_weekname_from(planweeks, i) for i in range(-12, 1)]
    col1 = ['Ref', 'Mapping Model.Suffix', 'To Site']

    for i, wk_name in enumerate(tqdm(planweeks)):
        if i == 0: # 누적할 sp의 초기값 셋팅
            acc_sp = sp_get(ver, 'Pegatron', wk_name)
        else:
            sp = sp_get(ver, 'Pegatron', wk_name)
            if sp.shape[0] == 0:
                continue
            deleted_wk = []
            for item in acc_sp.columns:
                if item not in sp.columns:
                    deleted_wk.append(item)
                    
            org_idx = acc_sp.reset_index().set_index(['Mapping Model.Suffix', 'To Site']).index
            new_idx = sp.reset_index().set_index(['Mapping Model.Suffix', 'To Site']).index
            intersecting_idx = org_idx.intersection(new_idx)
            diff_idx_org = org_idx.difference(intersecting_idx) # 첫 번째 인덱스에서 2번째 인덱스를 뺀 차집합 인덱스
            
            # 누적 데이터프레임에서 동일 index값이 가장 최근에 언제 주차에 있었는지 참조하기 위한 Series
            find_most_recent_week = acc_sp.reset_index()[col1].groupby(['Mapping Model.Suffix', 'To Site'])['Ref'].max()
            # 누적 데이터프레임에서 신규SP에 없는 row 중에서 가장 최근 주차의 행을 지정하기 위한 멀티인덱스를 만들기 위한 데이터프레임1
            idx_frame1 = find_most_recent_week[diff_idx_org].reset_index()[col1]
            m_idx1 = pd.MultiIndex.from_frame(idx_frame1)
            add_df = acc_sp.loc[m_idx1, :get_weekname_from(wk_name, -2)] # 이전주차까지의 값만 슬라이싱하여 추가할 df를 만듬
            for w in find_most_recent_week[diff_idx_org]:  # 누적 데이터프레임에서 차집합 인덱스를 반복하면서
                if w < get_weekname_from(wk_name, -1):     # 해당 최근주차가 지난주보다 작을 때
                    add_df.loc[w, w:] = 0                  # 해당 최근주차부터 이후 주차의 값은 '0' 으로 만듬(왜냐하면, 기존값은 실적이 아니므로)
            add_df = add_df[add_df.sum(axis=1) > 0]
            
            # deleted_wk 에 값이 있을 경우 df에 이전주차까지의 누적 데이터프레임의 삭제된 주차들의 선적 실적을 추가하는 작업
            if len(deleted_wk) > 0:
                # 누적 데이터프레임과 신규SP와 공통 row중에서 가장 최근 주차의 행을 지정하기 위한 멀티인덱스를 만들기 위한 데이터프레임1
                idx_frame2 = find_most_recent_week[intersecting_idx].reset_index()[col1]
                idx_frame3 = idx_frame2.copy()
                idx_frame3['Ref'] = wk_name
                m_idx2 = pd.MultiIndex.from_frame(idx_frame2)
                m_idx3 = pd.MultiIndex.from_frame(idx_frame3)
                sp.loc[m_idx3, deleted_wk] = acc_sp.loc[m_idx2, deleted_wk].values
                
            sp = pd.concat([add_df, sp]).reset_index()
            sp['Ref'] = wk_name
            sp = sp.set_index(col1)
            acc_sp = pd.concat([acc_sp, sp])

    df = acc_sp.reset_index().fillna(0)
    df.insert(1, 'Series', df['Mapping Model.Suffix'].apply(lambda x:x.split('-')[0]).replace(srt_model))
    df.insert(3, 'Region', df['To Site'].replace(site_map).replace(country_map))
    df = monthly_sum(df, ['Ref', 'Series', 'Region'])
    df.columns.name = 'Month'
    df = df.loc[:, df.sum() > 0].stack().reset_index()
    df.rename({0:'QTY'}, axis=1, inplace=True)

    st.subheader('1. 최근 4주간의 월별 물동 분포')
    fig = go.Figure()
    clr = px.colors.sequential.Reds
    for wk in df['Ref'].unique()[-4:]:
        df_wk = df[df['Ref'] == wk]
        fig.add_trace(go.Histogram(histfunc='sum', x=df_wk['Month'], y=df_wk['QTY'], name=wk, )) # 데이터프레임의 각 row의 값을 누적하여 Bar graph를 나타날 때 사용
    fig.update_layout(colorway=clr, width=1000, barmode='group')
    fig.update_xaxes(nticks=len(df_wk['Month'].unique()) + 1, tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('2. 최근 12주간의 SP 변동 현황')
    fig = px.histogram(df, x='Ref', y='QTY', barmode='relative', text_auto='QTY')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h6> - 월별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Month', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 모델별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Series', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 지역별 SP 분포</h6>", unsafe_allow_html=True )
    fig = px.histogram(df, x='Ref', y='QTY', color='Region', barmode='relative')
    st.plotly_chart(fig, theme=None)

    st.markdown("<h6> - 모델별 월별 물동 분포의 변동 </h6>", unsafe_allow_html=True )
    fig = make_subplots(rows=(len(df['Series'].unique()) + 3) // 4, cols=4, subplot_titles=df['Series'].unique(), shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.05)
    i = 0
    row_num = 1
    col_num = 1
    for sr in df['Series'].unique():
        df_sr = df[df['Series'] == sr]
        for mon in df_sr['Month'].unique():
            df1 = df_sr[df_sr['Month'] == mon]
            fig.add_trace(go.Histogram(histfunc='sum', x=df1['Ref'], y=df1['QTY'], name=mon, hovertext=df1['Month']), row=row_num, col=col_num)
        i += 1
        row_num = i // 4 + 1
        col_num = i % 4 + 1
    fig.update_layout(barmode='relative', height=1000, showlegend=False)
    fig.update_xaxes(tickangle=-65)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h6> - 모델별 지역별 물동 분포의 변동 </h6>", unsafe_allow_html=True )
    fig = make_subplots(rows=(len(df['Series'].unique()) + 3) // 4, cols=4, subplot_titles=df['Series'].unique(), shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.05)
    i = 0
    row_num = 1
    col_num = 1
    for sr in df['Series'].unique():
        df_sr = df[df['Series'] == sr]
        for reg in df_sr['Region'].unique():
            df1 = df_sr[df_sr['Region'] == reg]
            fig.add_trace(go.Histogram(histfunc='sum', x=df1['Ref'], y=df1['QTY'], name=mon, hovertext=df1['Region']), row=row_num, col=col_num)
        i += 1
        row_num = i // 4 + 1
        col_num = i % 4 + 1
    fig.update_layout(barmode='relative', height=1000, showlegend=False)
    fig.update_xaxes(tickangle=-65)
    st.plotly_chart(fig, use_container_width=True)

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

