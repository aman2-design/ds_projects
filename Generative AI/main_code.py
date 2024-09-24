

##############   Libraries   #################################################

import os
import streamlit as st    # to create API

from langchain_groq import ChatGroq
# It provides lpu(language processing unit) which is much faster than gpu or cpu,
#Generally used to acclerate the process/work related to word processing

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # Vector database
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv  # to load environment variables

##############   Environment variables   #################################################


load_dotenv()   # It helps in importing all the environment variables
 
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

##############   For api page(streamlit)   #################################################

st.title("Q&A chatbot using groq as processor and google as llm")

llm = ChatGroq(groq_api_key = groq_api_key,model_name = "Gemma-7b-it")
# model name will be available in groq site and this model is compatabale with google gemma key

prompt= ChatPromptTemplate.from_template(
"""
Answer the questions possibly based on the provided context only. If not found give me generic answer.
<context>
{context}
<context>
Questions:{input}

"""
)
 
def vector_embeddings():
    if 'vectors' not in st.session_state:
        
        st.session_state.loader = PyPDFLoader("EnrolPolicyConditions_24.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1 = st.text_input("what you want to ask ?")

if st.button("Create vectors"):
    vector_embeddings()
    st.write("Vector store db is ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)  # google llm data as vectors
    retriever = st.session_state.vectors.as_retriever()   # pdf data as vectors
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    
    start  = time.process_time()
    response = retriever_chain.invoke({'input':prompt1})
    st.write(response["answer"])


    # with streamlit expander

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("....................................")
