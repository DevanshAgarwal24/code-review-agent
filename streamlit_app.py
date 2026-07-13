import streamlit as st
from v4_add_linter import app, review_code

st.title("Code Review Agent")
st.write("Hello, this is working!")

code_input = st.text_area("Paste your code here:", height=300)

review_clicked = st.button("Review Code")

if review_clicked:
    if not code_input.strip():
        st.warning("Paste some code First.")
    else :
        with st.spinner("reviewing..."):
            review=review_code(code_input)
        st.markdown(review)
