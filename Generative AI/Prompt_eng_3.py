# This code is for whatever done in Prompt_eng_2.py, to store that conversation/sequence 
# in some memory.

import os
from api_keys import googleai_keys
from langchain.llms import GooglePalm
import streamlit as st
import google.generativeai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import  SimpleSequentialChain
# this library will help to create sequence for prompts so that it can run one by one.
# but one problem in this is we will get output of only last prompt

from langchain.chains import  SequentialChain
from langchain.memory import ConversationBufferMemory


# Initialising the environment
os.environ["GOOGLE_PALM_API_KEY"] = googleai_keys

# streamlit will help to create UI kind of thing for our model.

st.title("Langchain model with googleai API")
input_text = st.text_input("Search the topic you want")

#############################################################

llm = GooglePalm(google_api_key = googleai_keys)

# memory

first_prompt_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
second_prompt_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
third_prompt_memory = ConversationBufferMemory(input_key='name',memory_key='description_history')


# prompt  1
first_input_prompt = PromptTemplate(
    input_variables = ["name"],
    template = "Tell me about {name}"
)


chain1 = LLMChain(llm = llm,prompt=first_input_prompt,verbose=True,output_key='description',
                  memory=first_prompt_memory)

# prompt  2

second_input_prompt = PromptTemplate(
    input_variables = ["name"],
    template = "What is the age of {name}"
)

chain2 = LLMChain(llm = llm,prompt=second_input_prompt,verbose=True,output_key='age',
                  memory=second_prompt_memory)

# prompt  3

third_input_prompt = PromptTemplate(
    input_variables = ["name"],
    template = "What is the five most liked dishes of {name}"
)

chain3 = LLMChain(llm = llm,prompt=third_input_prompt,verbose=True,output_key='dishes',
                  memory=third_prompt_memory)

sequence_chain = SequentialChain(chains=[chain1,chain2,chain3],input_variables=["name"],
                                 output_variables=['description','age','dishes'],verbose=True)



if input_text:
    st.write(sequence_chain({'name':input_text}))
    
    with st.expander("Person details"):
        st.info(first_prompt_memory.buffer)

    with st.expander("Person age"):
        st.info(second_prompt_memory.buffer)  

    with st.expander("Person dishes"):
        st.info(third_prompt_memory.buffer)     
