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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


# Create FAISS vector store with the correct embedding model
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Create the conversational chain with user-defined temperature
def get_conversational_chain(temperature):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say:
    "Answer is not available in the context." Do not provide an incorrect answer.

    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Process the user input and generate the response
def user_input(user_question, temperature):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(temperature)

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])



# Main function for Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Add custom CSS to style the text input box
    st.markdown("""
        <style>
        .stTextInput > div > input {
            font-size: 16px;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #4CAF50;
            background-color: #f0f8ff;
            margin-bottom: 15px;
        }
        .custom-label {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create a label for the question input
    st.markdown('<div class="custom-label">Ask a Question from the PDF files:</div>', unsafe_allow_html=True)
    
    # Create the question input without the 'class_' argument
    user_question = st.text_input("", key="user_question", placeholder="Enter your question here...")

    if user_question:
        # Retrieve the temperature value from the sidebar
        temperature = st.session_state.get("temperature", 0.3)
        user_input(user_question, temperature)

    with st.sidebar:
        st.title("Menu:")
        
        # Replace the slider with a number input
        chunk_size = st.number_input("Set chunk size (number of characters):", min_value=100, max_value=5000, value=1000, step=100)

        pdf_docs = st.file_uploader(
            "Upload your PDF files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text, chunk_size)  # Pass user-defined chunk size
                get_vector_store(text_chunks)
                st.success("Done")

        # Add temperature explanation and number input to the sidebar
        st.write("**Temperature Control:**")
        st.write(
            """
        The temperature controls the model's creativity:
        
        - A **lower value** (closer to 0.0) makes the model more focused and deterministic.
        - A **higher value** (closer to 1.0) increases creativity but may lead to less accurate answers.
        """
        )

        # Temperature number input in the sidebar
        st.session_state["temperature"] = st.number_input(
            "Choose the temperature for the model (affects creativity)", 0.0, 1.0, 0.3
        )

# Modify get_text_chunks function to accept chunk_size parameter
def get_text_chunks(text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    main()