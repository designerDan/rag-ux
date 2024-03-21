import os
from dotenv import load_dotenv

from pinecone import Pinecone
from pinecone import PodSpec

from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#setting API keys
os.environ["GOOGLE_API_KEY"] = st.secrets.GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = st.secrets.PINECONE_API_KEY

#creating a Pinecone client
spec = PodSpec(environment="gcp-starter")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

#selecting the pinecone index
index = pc.Index("portfolio")

#setting the llm and embeddings
llm = Ollama(model="mixtral:8x7b")
embeddings = OllamaEmbeddings(model="mixtral:8x7b")

#creating the PineconeVectorStore
vectorsstore = PineconeVectorStore(
    index, embeddings
    )

#dowloading the web contents
loader = WebBaseLoader("https://portfolio.danielgmason.com")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

#using the chunks of documents to create the index
index = VectorStoreIndex.from_documents(documents)

#making the prompts
template = """
Answer the question based on the context below. If you can't  answer the question, reply "This is a tricky one. I don't have an answer. Will you ask the question in another way."

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# putting the chain together
retriever = vectorstore.as_retriever()
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser
)

#UI
import streamlit as st

st.title("UX for AI bot")

#initialize chat istory
if "history" not in st.session_state:
    st.session_state.history = []

#display message history
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

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
        response = chain.invoke(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)