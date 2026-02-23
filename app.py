import streamlit as st
from rag_pipeline import ask_question

st.title("Local RAG AI Assistant")

query = st.text_input("Ask a question")

if query:
    answer = ask_question(query)
    st.write(answer)