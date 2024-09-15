import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_odf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    return chunks

def get_vector_stoer(text_chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    vectore_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vectore_store.save_local("faiss_index")
    


