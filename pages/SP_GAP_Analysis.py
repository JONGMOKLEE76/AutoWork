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

st.subheader('특정 두 시점간의 SP의 차이를 보여줍니다.')
vendor = st.selectbox('업체를 선택하세요.', ('Quanta', 'Pegatron', 'Wingtech'), key='vendor')
wklist = [get_weekname_from(get_weekname(datetime.date.today()), i) for i in range(0, -10, -1)]
col1, col2, col3, col4 = st.columns(4)
with col1:
    week1 = st.selectbox('기준 SP의 주차를 선택하세요.', wklist, key='wk1')
with col2:
    ver1 = st.selectbox('SP의 Version을 선택하세요.', ('Final', 'Latest', 'ODM Release'), key='ver1')
with col3:
    week2 = st.selectbox('비교할 SP의 주차를 선택하세요.', wklist, key='wk2')
with col4:
    ver2 = st.selectbox('SP의 Version을 선택하세요.', ('Final', 'Latest', 'ODM Release'), key='ver2')

if st.button('Compare', key='compare'):
    sp1 = get_sp_from_GSCP_DB(week1, ver1, vendor)
    sp2 = get_sp_from_GSCP_DB(week2, ver2, vendor)
    sp1 = sp1.set_index(['Series', 'Mapping Model.Suffix', 'Region', 'Country', 'To Site'])
    sp2 = sp2.set_index(['Series', 'Mapping Model.Suffix', 'Region', 'Country', 'To Site'])
    end_col1 = sp1.loc[:, sp1.sum() != 0].columns[-1]
    end_col2 = sp2.loc[:, sp2.sum() != 0].columns[-1]
    sp1 = sp1.loc[:, :end_col1].reset_index()
    sp2 = sp2.loc[:, :end_col2].reset_index()

    diff_table_by_month = get_difference_table(monthly_sum(sp1.copy(), ['Series', 'Region']).reset_index(),
                                                monthly_sum(sp2.copy(), ['Series', 'Region']).reset_index(),
                                                '\d-\d')[monthly_sum(sp1.copy(), ['Series', 'Region']).columns]

    for name in diff_table_by_month.columns:
        diff_table_by_month.rename(columns={name:'Gap_'+name}, inplace=True)

    diff_table_by_month['Sum_Gap'] = diff_table_by_month.sum(axis=1)
    df_month1 = monthly_sum(sp1.copy(), ['Series', 'Region'])
    df_month1['Sum'] = df_month1.sum(axis=1)

    df_summary = pd.concat([df_month1, diff_table_by_month], axis=1)
    df_summary.reset_index(inplace=True)
    df_summary.insert(1, 'Vendor_Model',df_summary['Series'].replace(supplier_model_map))
    df_summary = df_summary.set_index(['Series', 'Vendor_Model', 'Region'])
    df_summary = pd.concat([pd.DataFrame(df_summary.sum().values.reshape(1, -1), columns=df_summary.columns, index=pd.MultiIndex.from_tuples([('', '', 'Total')])), df_summary])
    st.dataframe(df_summary)
    st.download_button(
        label="Download data as Excel",
        data=get_excel_file(df_summary),
        file_name='sp_gap.xlsx',
        mime='xlsx')