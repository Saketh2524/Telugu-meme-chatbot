# This is the magic fix for the sqlite3 version issue on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now the rest of your imports can follow
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import os

# --- 1. CONFIGURATION ---
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except AttributeError:
    st.warning("Please set your GOOGLE_API_KEY as a Streamlit Secret!", icon="⚠️")

# --- 2. CACHED FUNCTIONS ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('Data_set - Data.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        return df, embeddings, ids
    except FileNotFoundError:
        st.error("Data files not found! Ensure all necessary files are in the GitHub repo.")
        return None, None, None

@st.cache_resource
def setup_vector_db(_embeddings, _ids):
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

# --- 3. RAG CORE FUNCTION ---
def get_bot_response(user_query, df, collection):
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_ids = results['ids'][0]

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, arrogant, and high-attitude personality. Your knowledge is only Telugu memes. Your replies must be very short and dismissive, and directly use or reference the provided memes.
    
    USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE:
    1. {retrieved_contexts[0]}
    2. {retrieved_contexts[1]}
    3. {retrieved_contexts[2]}

    Generate a short, high-attitude reply that cleverly uses ONE of these memes to respond.
    """
    
    # --- THIS IS THE CORRECTED LINE ---
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')