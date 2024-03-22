import os

#app wrapper
import streamlit as st

#not sure but maybe a best practice? i dunno what this does
from dotenv import load_dotenv

load_dotenv()

#setting API keys

os.environ["GOOGLE_API_KEY"] = st.secrets.GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = st.secrets.PINECONE_API_KEY

#UI
st.title("UX for AI bot")

#initialize chat istory
if "history" not in st.session_state:
    st.session_state.history = []

#display message history
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

#setting the llm and embeddings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

llm = Gemini()
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 768

#initializing Pinecone
from pinecone import Pinecone

pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = 'ux-for-ai'
pinecone_index = pinecone.Index(index_name)

#generating the embeddings
from llama_index.core import SimpleDirectoryReader

loader = SimpleDirectoryReader(input_dir="./data")
docs = loader.load_data()

#adding the embeddings to the vector store
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import (
    StorageContext,
    VectorStoreIndex
)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs, 
    storage_context=storage_context
)

query_engine = index.as_query_engine()

#react to user input
query = st.chat_input("Say something")
if query:
    #display user message in chat container
    with st.chat_message("user"):
        st.markdown(query)

    #add user message to chat history
    st.session_state.history.append({
        'role':'user',
        'content':query
    })

    with st.spinner('💡Thinking'):
        # Query the index
        response = query_engine.query(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)