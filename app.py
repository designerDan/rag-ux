import os

#app wrapper
import streamlit as st
from datetime import datetime

#not sure but maybe a best practice? i dunno what this does
from dotenv import load_dotenv

load_dotenv()

#setting API keys
GOOGLE_API_KEY = st.secrets.GOOGLE_API_KEY
PINECONE_API_KEY = st.secrets.PINECONE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

#setting llama_index defaults
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

llm = Gemini(temperature=0.7, api_key=GOOGLE_API_KEY)
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.num_output = 768
Settings.context_window = 3900

#Wrapper UII
st.title("UX for AI RAG Experiment")
st.markdown("""This bot is not about which AI tools you can use as a UX designer or how to use AI to design software. I don't care about Figma's latest AI widget or what UIzard is doing with their software.
\nThis bot is for thought leadership on the emerging challenges, principles, heuristics, and foundations for designing AI-driven user experiences.""")
st.sidebar.markdown("""If you find this bot devoid of value then send me a note indicating what you hoped it would do and how you would make it better.
If you find this bot useful then send me a note detailing what value you got out of it.
\n[Give feedback](https://forms.gle/vjkzpaa91jkpPQbT9)
\nThis AI is built off the free tiers of Gemini and Streamlit. It is not likely though still possible that your usage will hit the upper limit of those tiers. If the software keeps throwing errors then that may be the cause. Try refreshing the browser, clearing your cache, and relaunching the site. If that doesn't work then send me a message via the provided feedback form.
\nFor a list of sources powering this experiment visit the Readme on Github.
\n[View sources](https://github.com/designerDan/rag-ux)""")

#initialize chat istory
if "history" not in st.session_state:
    st.session_state.history = []

#display message history
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

#load the data
from llama_index.core import SimpleDirectoryReader

loader = SimpleDirectoryReader(input_dir="./data")
documents = loader.load_data()

# Clean up documents
import re

def clean_up_text(content: str) -> str:
    """
    Remove unwanted characters and patterns in text input.

    :param content: Text input.

    :return: Cleaned version of original text input.
    """

    # Fix hyphenated words broken by newline
    content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

    # Remove specific unwanted patterns and characters
    unwanted_patterns = [
        "\\n", "  —", "——————————", "—————————", "—————",
        r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)

    # Fix improperly spaced hyphenated words and normalize whitespace
    content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
    content = re.sub(r'\s+', ' ', content)

    return content

cleaned_docs = []
for d in documents:
    cleaned_text = clean_up_text(d.text)
    d.text = cleaned_text
    cleaned_docs.append(d)

# Iterate through `documents` and add new key:value pairs
metadata_additions = {"authors": "Various", "permission": "Used without permission"}

# Updates dict in place
for cd in cleaned_docs:
    cd.metadata.update(metadata_additions)

#initializing Pinecone
from pinecone import Pinecone
pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# create the index
index_name = 'ux-for-ai'
existing_indexes = [i.get('name') for i in pinecone.list_indexes()]

if index_name not in existing_indexes:
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
    )

# connect to the index
pinecone_index = pinecone.Index(index_name)

# select a namespace
namespace = '' # default namespace

# initializing the vector store
from llama_index.vector_stores.pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# define the ingestion pipeline
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser

pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
            ),
        embed_model,
        ],
        vector_store=vector_store
    )

pipeline.persist("./pipeline_storage")
pipeline.load("./pipeline_storage")

nodes = pipeline.run(document=cleaned_docs)

pinecone_index.describe_index_stats()

#query the data
from llama_index.core import StorageContext, VectorStoreIndex, PromptTemplate, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex(
    nodes=nodes,
    vector_store=vector_store,
    storage_context=storage_context
)

retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=5
)

response_synthesizer=get_response_synthesizer(
    response_mode="tree_summarize",
    llm=llm,
    verbose=True,
    structured_answer_filtering=True,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer)

#updating
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query using language that meets a Flesch-Kincaid score of 70 or greater.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
 
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

# Creating a file to store user queries
queries_file = "./user_queries.txt"

# Function to save the query to the file
def save_query(query):
    with open(queries_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: {query}\n")

#UI
if st.button("What are some challenges designers face when designing experiences for AI?"):
    query = "What are some challenges designers face when designing experiences for AI?"
    
    #display user message in chat container
    with st.chat_message("user"):
        st.markdown(query)

    #add user message to chat history
    st.session_state.history.append({
        'role':'user',
        'content':query
    })

    with st.spinner('💡Thinking'):
        # Query the db
        response = query_engine.query(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)

if st.button("What are some heuristics designers need to consider when designing experiences for AI?"):
    query = "What are some heuristics designers need to consider when designing experiences for AI?"
    
    #display user message in chat container
    with st.chat_message("user"):
        st.markdown(query)

    #add user message to chat history
    st.session_state.history.append({
        'role':'user',
        'content':query
    })

    with st.spinner('💡Thinking'):
        # Query the db
        response = query_engine.query(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)

if st.button("Tell me ways to establish user trust in AI."):
    query = "Tell me ways to establish user trust in AI."
    
    #display user message in chat container
    with st.chat_message("user"):
        st.markdown(query)

    #add user message to chat history
    st.session_state.history.append({
        'role':'user',
        'content':query
    })

    with st.spinner('💡Thinking'):
        # Query the db
        response = query_engine.query(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)
        
query = st.chat_input("What would you like to know?")

if query:
    #save query
    save_query(query)

    #display user message in chat container
    with st.chat_message("user"):
        st.markdown(query)

    #add user message to chat history
    st.session_state.history.append({
        'role':'user',
        'content':query
    })

    with st.spinner('💡Thinking'):
        # Query the db
        response = query_engine.query(query)

        st.session_state.history.append({
            'role':'Assistant',
            'content':response
        })

    #print the response
    with st.chat_message("Assistant"):
        st.markdown(response)