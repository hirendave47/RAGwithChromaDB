import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.ragwithchromadb.utils import read_file, get_table_data
import streamlit as st
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback
# from src.ragwithchromadb.RAGGenerator1 import generate_evaluate_chain
from src.ragwithchromadb.logger import logging
from src.ragwithchromadb.utils import convert_pdf_to_text
from src.ragwithchromadb.RAGGenerator import create_database, load_database

# loading json file

# with open('Response.json', 'r') as file:
#     RESPONSE_JSON = json.load(file)

# creating a title for the app
st.title("RAG - Retrieval Augmented Generation 🦜⛓️")

# Create a form using st.form
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Uplaod a PDF file")
    query = st.text_input("Insert Query", max_chars=100, value="Which animals are mentioned?")
    button = st.form_submit_button("Query")

    if button and uploaded_file is not None and query:
        with st.spinner("loading..."):
            try:
                file_name = convert_pdf_to_text(uploaded_file)
                create_database(file_name)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                st.text_area(label="Review", value=load_database(query), height=250)