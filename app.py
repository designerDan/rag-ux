import os

#app wrapper
import streamlit as st

#not sure but maybe a best practice? i dunno what this does
from dotenv import load_dotenv

load_dotenv()

#setting API keys
os.environ["GOOGLE_API_KEY"] = st.secrets.GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = st.secrets.PINECONE_API_KEY

#setting the llm and embeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

llm = Ollama(model="mixtral:8x7b")
embeddings = OllamaEmbeddings(model="mixtral:8x7b")

#initializing Pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time

spec = PodSpec(environment="gcp-starter")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# configure client
if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
else:  
    # if not using a starter index, you should specify a pod_type too  
    spec = PodSpec()

# check for and delete index if already exists  
index_name = 'ux-for-ai'  
if index_name in pc.list_indexes().names():  
    pc.delete_index(index_name)  

# create a new index  
pc.create_index(  
    index_name,  
    dimension=1536,  # dimensionality of text-embedding-ada-002  
    metric='dotproduct',  
    spec=spec  
)

# wait for index to be initialized  
while not pc.describe_index(index_name).status['ready']:  
    time.sleep(1)

pc_index = pc.Index(index_name)

#generating the embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./data/data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

#adding the embeddings to the vector store
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(pc_index=pc_index, embeddings=embeddings)

index = vectorstore.from_documents(docs)

#making the prompts
from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below. If you can't  answer the question, reply "This is a tricky one. I don't have an answer. Will you ask the question in another way."

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# putting the chain together
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = index.as_retriever()
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