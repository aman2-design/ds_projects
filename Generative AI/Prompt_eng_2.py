# This code is to built a basic prompt enginering(Q and A) using Langchain.

import os
from api_keys import googleai_keys
from langchain.llms import GooglePalm
import streamlit as st
import google.generativeai
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import  SimpleSequentialChain
# this library will help to create sequence for prompts so that it can run one by one.
# but one problem in this is we will get output of only last prompt

from langchain.chains import  SequentialChain


# Initialising the environment
os.environ["GOOGLE_PALM_API_KEY"] = googleai_keys

# streamlit will help to create UI kind of thing for our model.

st.title("Langchain model with googleai API")
input_text = st.text_input("Search the topic you want")

#############################################################

llm = GooglePalm(google_api_key = googleai_keys)

# prompt  1
first_input_prompt = PromptTemplate(
    input_variables = ["name"],
    template = "Tell me about {name}"
)

chain1 = LLMChain(llm = llm,prompt=first_input_prompt,verbose=True,output_key='description')

# prompt  2

second_input_prompt = PromptTemplate(
    input_variables = ["description"],
    template = "What is the age of {description}"
)

chain2 = LLMChain(llm = llm,prompt=second_input_prompt,verbose=True,output_key='age')

# prompt  3

third_input_prompt = PromptTemplate(
    input_variables = ["age"],
    template = "What is the five most liked dishes of {description}"
)

chain3 = LLMChain(llm = llm,prompt=third_input_prompt,verbose=True,output_key='dishes')

sequence_chain = SequentialChain(chains=[chain1,chain2,chain3],input_variables=
                                 ["name"],output_variables=['description','age','dishes'],verbose=True)

if input_text:
    st.write(sequence_chain({'name':input_text}))