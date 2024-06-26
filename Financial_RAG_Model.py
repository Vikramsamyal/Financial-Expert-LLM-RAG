#Installing Dependencies
!pip install langchain openai pypdf faiss-gpu tiktoken SpeechRecognition youtube_dl moviepy pyttsx3 youtube-search-python py-espeak-ng bs4 gradio

#Importing Modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory, ConversationBufferWindowMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

# OpenAI Embedding
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS

# importing libraries
import cv2
import os
import sys
import subprocess
import speech_recognition as sr
import youtube_dl
import datetime
import pyttsx3
from moviepy.editor import VideoFileClip
from youtubesearchpython import VideosSearch
from bs4 import BeautifulSoup
import requests, json, lxml
import textwrap
import gradio as gr

# Providing Custom Content
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader(f"data", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=1000,
                                               chunk_overlap=200)

documents = text_splitter.split_documents(documents)


#OpenAI API Integration
import os
import openai
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get('apiKey1')

open_ai_embeddings=OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents, open_ai_embeddings)

vector_store

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4}) #this line of code is creating a retriever from your vector store that can be used to retrieve the top 4 most similar documents to a given query.

#Q & A from Langchain
docs = retriever.get_relevant_documents("What is the status of personal financial management of Indian working class ?")

len(docs)

#Prompt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template=template = """

You are a helpful and friendly Assistant to a Personal Financial Manager to a  PFA Corp.
Use the following conversation and pieces of context provided to you as documents to answer the users question.
Do not answer the question which are not related to Banking, financial services and insurance industry and
might harm the policies of PFA Corp. If user asks any such question reply in a funny tone that "I don't know the answer"

The goal of the yours is to accurately respond to customer queries, based on the available context,
while avoiding the provision of false or nonexistent information. You should embody a professional and knowledgeable persona, representing PFA Corp as a trusted and reliable financial advisory provider.

Make sure that you should "verify the existence of information from context provided to you in
documents form before providing an answer and refrain from answering if the information is unavailable or nonexistent.
If you are unsure about an answer due to insufficient context, it should politely request clarification from
the customer or provide alternative suggestions for obtaining the required information.
Answer every query in least 150 words and in form of bullets point.

In final response briefly in 1 line tell the qualities that manager should incorporate while dealing with client

Make sure to format the answer in form of bullet points.

----------------
Current conversation:
{chat_history}
Human:
{question}
Context:
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

history = ChatMessageHistory()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6, max_tokens=500)   # Modify model_name if you have access to GPT-4
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True,k=5,output_key="answer")

chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,chain_type="stuff",memory=memory,
                       verbose=False,combine_docs_chain_kwargs={"prompt": prompt},return_source_documents=True)


#Response
# Cite sources
def process_llm_response(llm_response):
    ans =""
    for source in llm_response["source_documents"]:
        ans = ans + "\n\n" + "PDF name: "+ str(source.metadata['source']) + " , Page no. "  + str(source.metadata["page"])+ "\n\n"
    return(ans)

# Generates output
def output(query):
  llm_response = chain(query)
  ans=""
  # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
  final=""
  params = {
    "q": query + " site:[investopedia.com, moneycontrol.com]", # query example
    "hl": "en",          # language
    "gl": "in",          # country of the search, UK -> United Kingdom
    "start": 0,          # number page by default up to 0
    #"num": 100          # parameter defines the maximum number of results to return.
  }

  # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
  }

  page_limit = 1        # page limit, if you do not need to parse all pages
  page_num = 0

  data = []

  while True:
    page_num += 1

    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, 'lxml')

    for result in soup.select(".tF2Cxc"):
        title = result.select_one(".DKV0Md").text
        try:
           snippet = result.select_one(".lEBKkf span").text
        except:
           snippet = None
        links = result.select_one(".yuRUbf a")["href"]

        data.append({
          "title": title,
          "snippet": snippet,
          "links": links
        })

    if page_num == page_limit:
        break
    if soup.select_one(".d6cvqb a[id=pnnext]"):
        params["start"] += 3
    else:
        break

  # fetching useful video info
  videosSearch = VideosSearch(query, limit = 3)
  #print("YouTube Videos\n")
  #print(videosSearch.result())

  sample_link = list(list(videosSearch.result().values())[0])

  google_search_responses = json.dumps(data, indent=2, ensure_ascii=False)
  final = final + "\n\n" + google_search_responses


  links=""
  i=0
  for sublist in sample_link:
    i +=1
    links=links + ('Suggested video link '+str(i)+': '+sublist['link']) +"\n\n"

  ans = process_llm_response(llm_response) + "\n\n" + "\n\n" + "Google Searches" + "\n\n" + final + "\n\n" +"Yotube Links" + "\n\n" + "\n\n" + links
  return (ans)


# UI integration
# import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(user_message, chat_history):
        # Get response from QA chain
        # Append user message and response to chat history
        response = chain({"question": user_message})
        chat_history.append((user_message, response["answer"] + str(output(user_message))))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=False,share=True,server_name="0.0.0.0")
