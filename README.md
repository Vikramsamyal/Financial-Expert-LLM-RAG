# Financial-Expert-LLM

An LLM-based Retrieval-Augmented Generation (RAG) model to enhance financial data response quality using Langchain and vector databases.

## Project Overview

This project aims to create a financial expert system using a Retrieval-Augmented Generation (RAG) model. The model uses custom financial data to improve response accuracy and provide detailed financial advice. The project leverages Langchain for text splitting, OpenAI for embeddings, and FAISS for vector database creation.

## Features

- **Custom Financial Data**: The system is trained on proprietary financial documents to enhance its knowledge base.
- **Langchain Integration**: Utilizes Langchain for document loading, splitting, and creating retrieval chains.
- **OpenAI Embeddings**: Uses OpenAI's embedding models to convert text into high-dimensional vectors.
- **Vector Database**: FAISS is used to create a vector store for efficient document retrieval.
- **Conversational Retrieval Chain**: Implements a conversational chain to interact with users and provide context-aware responses.
- **Google Search and YouTube Integration**: Fetches additional information and relevant videos from the web.

## Tools and Technologies

- **Langchain**
- **OpenAI**
- **FAISS**
- **PyPDFLoader**
- **Gradio**
- **BeautifulSoup**
- **YouTube Search**
- **MoviePy**
- **SpeechRecognition**

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install langchain openai pypdf faiss-gpu tiktoken SpeechRecognition youtube_dl moviepy pyttsx3 youtube-search-python py-espeak-ng bs4 gradio

2. **Set Up OpenAI API Key**:
   import os
   os.environ["OPENAI_API_KEY"] = 'your_openai_api_key'

## Usage
A. Document Loading: Loads financial documents from the data directory.
B. Text Splitting: Splits documents into manageable chunks for processing.
C. Vector Store Creation: Creates a vector store from the document embeddings.
D. Conversational Interface: Interacts with users to answer queries based on the document context.
E. Web Search and Video Suggestions: Provides additional information from Google and YouTube.


## Example Query

query = "What is the status of personal financial management of Indian working class?"
response = output(query)
print(response)


## Future Improvements
A. Expand Data Sources: Include more financial documents and diverse data sources.
B. Enhance Retrieval Accuracy: Improve the retrieval mechanism for more precise answers.
C. User Interface: Develop a more user-friendly interface for interacting with the system.

