import streamlit as st
from dataclasses import dataclass
from typing import Literal
import base64
import pandas as pd
import time
import pickle
import os
import torch

from src.config import KatzBotConfig  # Import the configuration class

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


@dataclass
class Message:
    """
    A class to represent a chat message.

    Attributes:
        origin (Literal["human", "ai"]): The origin of the message, either human or AI.
        message (str): The content of the message.
    """
    origin: Literal["human", "ai"]
    message: str


class KatzBotApp:
    """
    A class to encapsulate the KatzBot application logic.

    Attributes:
        config (KatzBotConfig): The configuration object for the application.
        llm (ChatGroq): The large language model object.
        prompt (ChatPromptTemplate): The chat prompt template.
    """

    def __init__(self, config: KatzBotConfig):
        """
        Initialize the KatzBotApp with a configuration.

        Args:
            config (KatzBotConfig): The configuration object for the application.
        """
        self.config = config
        self.llm = ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=self.config.model_name)
        self.prompt = ChatPromptTemplate.from_template(
            """
            Please answer the question using the provided context, link, and much like anything which seems best for it, 
            ensuring the response is at least three sentences long and concise.
            If you cannot find a satisfactory answer in the context, conduct a Google search and provide the most precise information available.

            <context>
            {context}
            <context>
            Question:{input}
            """
        )

    def load_css(self):
        """Load and apply custom CSS for the application."""
        with open(self.config.CSS_PATH, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

    def get_image_base64(self, image_path):
        """
        Get the base64 encoding of an image.

        Args:
            image_path (str): The file path to the image.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string

    def create_vector_embedding(self):
        """Create vector embeddings for document retrieval."""
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        with open(self.config.DATA_FILE, "rb") as f:
            st.session_state.docs = pickle.load(f)
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        st.session_state.retriever = st.session_state.vectors.as_retriever()
        st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)

    def handle_submit(self):
        """Handle user input submission and generate AI response."""
        user_input = st.session_state.user_input
        if user_input:
            start = time.process_time()
            chatbot_response = st.session_state.retrieval_chain.invoke({'input': user_input})
            print(f"Response time :{time.process_time() - start}")

            st.session_state.history.append(Message("human", user_input))
            st.session_state.history.append(Message("ai", chatbot_response['answer']))

            st.session_state.user_input = ""




# Initialize the configuration
config = KatzBotConfig()
app = KatzBotApp(config)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the GROQ API Key
os.environ['GROQ_API_KEY'] = config.GROQ_API_KEY

# If you do not have OpenAI key use the below Huggingface embedding
os.environ['HUGGINGFACE_TOKEN'] = config.HUGGINGFACE_TOKEN

# Initialize the vector embedding once when the script runs
if "vectors" not in st.session_state:
    app.create_vector_embedding()
    print("Vector created successfully")

# Initialize or load session state
if 'history' not in st.session_state:
    st.session_state.history = []

bot_image_base64 = app.get_image_base64(config.BOT_IMAGE_PATH)
human_image_base64 = app.get_image_base64(config.HUMAN_IMAGE_PATH)
app.load_css()  # Apply the custom CSS

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
user_input = col1.text_input("User Input", value="", placeholder="Ask your question here", label_visibility="collapsed", key="user_input", on_change=app.handle_submit)
col2.button("Submit")  # The button no longer has on_click callback
st.markdown('</div>', unsafe_allow_html=True)  # Close fixed-input-container div

# Add some space at the end
for _ in range(3):
    st.markdown("")

existing_df = pd.DataFrame()

# Read existing chat history CSV file if it exists
try:
    existing_df = pd.read_csv(config.CHAT_HISTORY_FILE)
except FileNotFoundError:
    pass

# Concatenate existing chat history with current session state chat history
new_df = pd.concat([existing_df, pd.DataFrame(st.session_state.history)])

# Write the combined chat history to the CSV file
new_df.to_csv(config.CHAT_HISTORY_FILE, index=False)
