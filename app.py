# Magic fix for the sqlite3 version issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# All our library imports
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import os

st.title("✅ Test 4: Data Loading & DB Setup")

# Configure the API Key
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    st.write("API Key configured.")
except (KeyError, AttributeError):
    st.error("ERROR: The GOOGLE_API_KEY was not found. Please re-check your Streamlit Secrets.", icon="🚨")

# Define the functions to load data and setup the DB
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('Data_set - Data.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        return df, embeddings, ids
    except FileNotFoundError as e:
        st.error(f"ERROR: A data file was not found. Make sure all files are in your GitHub repo. Details: {e}", icon="🚨")
        return None, None, None
    except Exception as e:
        st.error(f"ERROR: An error occurred while loading data files. Details: {e}", icon="🚨")
        return None, None, None

@st.cache_resource
def setup_vector_db(_embeddings, _ids):
    try:
        client = chromadb.Client()
        collection_name = "memes"
        if collection_name in [c.name for c in client.list_collections()]:
            collection = client.get_collection(name=collection_name)
        else:
            collection = client.create_collection(name=collection_name)
        if collection.count() == 0:
            collection.add(
                embeddings=_embeddings.tolist(),
                ids=[str(i) for i in _ids]
            )
        return collection
    except Exception as e:
        st.error(f"ERROR: An error occurred during ChromaDB setup. Details: {e}", icon="🚨")
        return None

# --- Run the test ---
st.write("Attempting to load data files...")
meme_df, embeddings, ids = load_data()

if meme_df is not None:
    st.write("Data files loaded successfully.")
    st.write("Attempting to set up the vector database...")
    collection = setup_vector_db(embeddings, ids)
    if collection is not None:
        st.success("If you can see this, the Data was Loaded and the DB was Setup Successfully!")
        st.write(f"The database contains {collection.count()} meme entries.")
