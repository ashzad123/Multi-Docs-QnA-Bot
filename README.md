
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

