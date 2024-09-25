
# Multi QnA bot using Langchain

This project allows users to chat with multiple PDF documents by leveraging Langchain, Google Generative AI, and FAISS for building an efficient question-answering (QnA) bot. Users can upload PDFs, process them into chunks, and ask questions, while the chatbot searches for relevant answers within the PDFs and responds using a generative AI model. The bot’s behavior, including response word limit and creativity (temperature), can be customized via user input.
## Features

- Upload multiple PDF documents.
- Split the documents into smaller chunks for processing.
- Use FAISS to store and retrieve vector embeddings for fast similarity search.
- Interact with Google Generative AI (Gemini) for generating human-like responses.
- Adjust response creativity using a temperature setting.
- Control the response word limit as per your requirements.
- Easy-to-use interface with Streamlit.


## Seup and Installation
To get started with the project, follow the steps below:

#### 1.Clone the repository:


```bash
 git clone https://github.com/ashzad123/Multi-Docs-QnA-Bot.git
 ```


#### 2.Navigate to the project directory:
```bash
Multi-Docs-QnA-Bot
```
#### 3.Create a virtual environment and activate it.
```bash
python -m venv <environment name>
source <environment name>/bin/activate (for Linux)
<environment name>\Script\activate (for Windows)
```
#### 4.Install the requirements:
```bash
pip install -r requirements.txt
```

#### Set up your Google Generative AI API key by creating a .env file:

```bash 
touch .env
```
Inside the .env file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

#### Run the streamlit app:
```bash 
streamlit run app.py
```

## How It Works

### Function Overview

Here is an overview of the functions in the project and how they work:

**get_pdf_text(pdf_docs)**   
This function extracts the text from uploaded PDF files. It uses PyPDF2 to read and extract the text from each page of the PDF.

```bash
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

```
**get_text_chunks(text)**   
This function splits the extracted text into smaller chunks to handle large text inputs. It uses the RecursiveCharacterTextSplitter from langchain_text_splitters to break the text into manageable pieces.

```bash
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


```
**get_vector_store(text_chunks)**   
This function creates a FAISS vector store from the text chunks using Google Generative AI embeddings. It helps with fast similarity searches when the user asks a question.

```bash
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


```
**get_conversational_chain(temperature, word_limit)**   
This function defines the conversational chain using Langchain’s PromptTemplate and the Google Generative AI model. It accepts the user-defined temperature for creativity and word limit for responses.

```bash
def get_conversational_chain(temperature, word_limit):
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say: "Answer is not available in the context." 
    Do not provide an incorrect answer. Limit your response to a maximum of {word_limit} words.

    Context:
    {{context}}
    
    Question:
    {{question}}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


```
**user_input(user_question, temperature, word_limit)**   
This function handles the user input (question) and generates a response. It first loads the FAISS index, retrieves relevant documents, and uses the conversation chain to generate a response based on the user’s question, temperature, and word limit settings.

```bash
def user_input(user_question, temperature, word_limit):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(temperature, word_limit)

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


```



