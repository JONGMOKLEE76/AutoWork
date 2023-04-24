import streamlit as st
from tool import *

st.title("🐞 ODM Biz Issue Report!")

form = st.form(key="annotation")

with form:
    cols = st.columns((1, 1))
    author = cols[0].text_input("Report author:")
    issue_type = cols[1].selectbox(
        "Category", ["Pegatron", "Quanta", "Wingtech", "Common"], index=2
    )
    comment = st.text_area("Issues:")
    cols = st.columns(2)
    date = cols[0].date_input("Issue date occurrence:")
    status = cols[1].radio("Issue Status:", options=['Open', 'Closed'])
    submitted = st.form_submit_button(label="Submit")

if submitted:
    with open('D:/Data/issue_db.bin', 'rb') as f:
        issue_db = pickle.load(f)
    
    issue_data = {'Author':[author], 'Type':[issue_type], 'content':[comment], 'Date':[date], 'status':[status]}
    issue_db = pd.concat([issue_db, pd.DataFrame(issue_data)])
    
    with open('D:/Data/issue_db.bin', 'wb') as f:
        pickle.dump(issue_db, f)
    
    st.dataframe(issue_db)
    st.success("Thanks! Your Issue was registered.")
    st.balloons()