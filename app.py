import os
import streamlit as st
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, download_loader

from llama_index.core import Settings

#setting API keys
GOOGLE_API_KEY = st.secrets.GOOGLE_API_KEY
PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

DATA_URL = "https://portfolio.danielgmason.com"

llm = Gemini()

#creating a Pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

#selecting the pinecone index
pinecone_index = pinecone_client.Index("portfolio")

#dowloading the web contents
BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=[DATA_URL])

# Define which embedding model to use "models/embedding-001"
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Create a PineconeVectorStore using the specified pinecone_index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a StorageContext using the created PineconeVectorStore
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Use the chunks of documents and the storage_context to create the index
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

#query Pinecone vector store
query_engine = index.as_query_engine()

# Query the index, send the context to Gemini, and wait for the response
gemini_response = query_engine.query("What does the author think about LlamaIndex?")

#print the response
print(gemini_response)