# This code is a basic code to fetch LLM api and use it in our model.

import os
from api_keys import googleai_keys
from langchain.llms import GooglePalm
import streamlit as st
import google.generativeai

# Initialising the environment
os.environ["GOOGLE_PALM_API_KEY"] = googleai_keys

# streamlit will help to create UI kind of thing for our model.

st.title("Langchain model with googleai API")
input_text = st.text_input("Search the topic you want")
#############################################################



llm = GooglePalm(google_api_key = googleai_keys)

#llm = google_palm(temperature = 0.8)

if input_text:
    st.write(llm(input_text))