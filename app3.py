# conversational chatbot using Groq, Rag pipeline and langsmith
 
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage,AIMessage

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

st.set_page_config(page_title="AI Bot", page_icon="🤖")

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vector=FAISS.from_documents(st.session_state.documents,st.session_state.embeddings)

st.title("Documemted Conversational Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-8b-8192")

prompt_template=ChatPromptTemplate.from_template(
"""
You can also have a normal conversation with the use.
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question in 50 words appx.
<context>: {context}
Chat History: {chat_history}
Question:{input}
"""
)

document_chain=create_stuff_documents_chain(llm,prompt_template)
retrieval=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retrieval,document_chain)

for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

prompt=st.chat_input("Enter your prompt here....")

if prompt:
    st.session_state.chat_history.append(HumanMessage(prompt))

    with st.chat_message("Human"):
        st.markdown(prompt)
    
    with st.chat_message("AI"):
        ai_response=retrieval_chain.invoke({"input":prompt,"chat_history": st.session_state.chat_history})
        st.write(ai_response["answer"])
    
    st.session_state.chat_history.append(AIMessage(ai_response["answer"]))