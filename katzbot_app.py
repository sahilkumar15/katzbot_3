import streamlit as st
from dataclasses import dataclass
from typing import Literal
import base64
import pandas as pd
import time
import pickle
import os
import torch

from src.config2 import BOT_IMAGE_PATH, HUMAN_IMAGE_PATH, CSS_PATH, DATA_FILE, CHAT_HISTORY_FILE, GROQ_API_KEY, HUGGINGFACE_TOKEN

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

## Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

## load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HUGGINGFACE_TOKEN'] = os.getenv("HUGGINGFACE_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Model: Mixtral-8x7b-32768, Llama3-8b-8192
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Mixtral-8x7b-32768")

# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Use at least three sentences and keep the answer concise.
#     Please provide the most accurate response based on the question and the website.
#     if you didnot find the proper answer take help from google and try again.
#     <context>
#     {context}
#     <context>
#     Question:{input}
#     """
# )

prompt = ChatPromptTemplate.from_template(
    """
    Please answer the question using the provided context, link, and much like anything which seems best for it, ensuring the response is at least three sentences long and concise.
    If you cannot find a satisfactory answer in the context, conduct a Google search and provide the most precise information available.

    <context>
    {context}
    <context>
    Question:{input}
    """
)


@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def load_css():
    with open(CSS_PATH, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def create_vector_embedding():
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    with open(DATA_FILE, "rb") as f:
        st.session_state.docs = pickle.load(f)
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retriever = st.session_state.vectors.as_retriever()
    st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)

def handle_submit():
    user_input = st.session_state.user_input
    if user_input:
        start = time.process_time()
        chatbot_response = st.session_state.retrieval_chain.invoke({'input': user_input})
        print(f"Response time :{time.process_time()-start}")

        st.session_state.history.append(Message("human", user_input))
        st.session_state.history.append(Message("ai", chatbot_response['answer']))

        st.session_state.user_input = ""

# Initialize the vector embedding once when the script runs
if "vectors" not in st.session_state:
    create_vector_embedding()
    print("Vector created successfully")

# Initialize or load session state
if 'history' not in st.session_state:
    st.session_state.history = []

bot_image_base64 = get_image_base64(BOT_IMAGE_PATH)
human_image_base64 = get_image_base64(HUMAN_IMAGE_PATH)
load_css()  # Apply the custom CSS

# Starting divs 
st.title("Katzbot")

# Use a container for the chat history and apply the chat-container class
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.history:
        if chat.origin == 'ai':
            image_base64 = bot_image_base64
        else:
            image_base64 = human_image_base64

        div = f"""
        <div class="chat-row 
            {'' if chat.origin == 'ai' else 'row-reverse'}">
            <img class="chat-icon" src="data:image/png;base64,{image_base64}"
                 width=32 height=32>
            <div class="chat-bubble
            {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container div

# Add a fixed input area
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([4.5, 1])
user_input = col1.text_input("User Input", value="", placeholder="Ask your question here", label_visibility="collapsed", key="user_input", on_change=handle_submit)
col2.button("Submit")  # The button no longer has on_click callback
st.markdown('</div>', unsafe_allow_html=True)  # Close fixed-input-container div

# Add some space at the end
for _ in range(3):
    st.markdown("")

existing_df = pd.DataFrame()

# Read existing chat history CSV file if it exists
try:
    existing_df = pd.read_csv(CHAT_HISTORY_FILE)
except FileNotFoundError:
    pass

# Concatenate existing chat history with current session state chat history
new_df = pd.concat([existing_df, pd.DataFrame(st.session_state.history)])

# Write the combined chat history to the CSV file
new_df.to_csv(CHAT_HISTORY_FILE, index=False)
