#import ChatOllama
import streamlit as st
import os
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sympy.physics.units import temperature
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()
st.header("Sai Charan's ChatBot")

with st.sidebar:
    st.title("My Documents")
    file = st.file_uploader("Upload a file", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever()

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=1000
    )
    llm = ChatGroq(
        temperature = 0,
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    user_question = st.text_input("Enter your question")

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
