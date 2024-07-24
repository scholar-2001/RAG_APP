import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
import os

os.environ["MISTRAL_API_KEY"] = st.secrets.MISTRAL_API_KEY
st.set_page_config(page_title="Chat about Apple Vision Pro", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat about Apple Vision Pro, powered by LlamaIndex and Mistral 💬🦙")
st.info("Chatbot for context based info about Apple Vison Pro", icon="📃")

llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model
if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about Apple Vision Pro!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    apple_docs = SimpleDirectoryReader(input_files=["Apple_Vision_Pro_Privacy_Overview.pdf"]).load_data()
    index = VectorStoreIndex.from_documents(apple_docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)