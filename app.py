# This is the magic fix for the sqlite3 version issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Now the rest of your imports can follow
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# --- 1. CONFIGURATION ---
# Configure your Google API Key. Get your key from https://aistudio.google.com/
# It's recommended to set this as a Streamlit Secret.
# For local testing, you can uncomment the line below and paste your key.
# os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"

try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except AttributeError:
    st.warning("Please set your GOOGLE_API_KEY as a Streamlit Secret to use the bot!", icon="⚠️")


# --- 2. CACHED FUNCTIONS TO LOAD DATA AND MODELS ---
# Using caching is crucial for performance in Streamlit.

@st.cache_resource
def load_data():
    """Loads the dataset, embeddings, and IDs from files."""
    try:
        df = pd.read_csv('Data_set - Data.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        return df, embeddings, ids
    except FileNotFoundError:
        st.error("Data files not found! Please make sure 'Data_set - Data.csv', 'embeddings.npy', and 'ids.npy' are in the same folder as app.py.")
        return None, None, None

@st.cache_resource
def setup_vector_db(_embeddings, _ids):
    """Sets up and populates the ChromaDB vector database."""
    client = chromadb.Client()
    collection_name = "memes"
    
    # Check if the collection exists to avoid errors
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)
    
    # Add data to the collection
    collection.add(
        embeddings=_embeddings.tolist(),
        ids=[str(i) for i in _ids]
    )
    return collection

@st.cache_resource
def load_embedding_model():
    """Loads the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. THE RAG (RETRIEVAL-AUGMENTED GENERATION) CORE ---

def get_bot_response(user_query, df, collection, model):
    """
    Takes a user query and returns the bot's witty response.
    """
    # 1. Create embedding for the user's query
    query_embedding = model.encode(user_query).tolist()

    # 2. Query the vector database to find relevant memes
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3 # Get the top 3 most relevant memes
    )
    
    retrieved_ids = results['ids'][0]
    
    # 3. Retrieve the context for the found memes
    retrieved_contexts = []
    for meme_id in retrieved_ids:
        # Find the row in the DataFrame that matches the meme_id
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)
        
    # 4. Construct the Master Prompt for the Generative Model
    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, arrogant, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    Your replies must be very short, dismissive, and directly use or reference the provided memes.
    
    USER'S QUERY: "{user_query}"

    Based on the user's query, here are the three most relevant memes from your knowledge base:
    1. {retrieved_contexts[0]}
    2. {retrieved_contexts[1]}
    3. {retrieved_contexts[2]}

    Your task is to now generate a short, high-attitude reply that cleverly uses ONE of these memes to respond to the user. Do not be helpful. Be sarcastic and dismissive.
    """
    
    # 5. Call the Generative AI
    generative_model = genai.GenerativeModel('gemini-pro')
    response = generative_model.generate_content(prompt)
    
    return response.text


# --- 4. STREAMLIT USER INTERFACE ---

# Load everything once at the start
meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)
    embedding_model = load_embedding_model()

st.title("🗣️ Meme Mowa Chat")
st.markdown("Nenu chaala planned ga untaa nandi... adagandi.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Em sangathulu?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if meme_df is not None:
        # Get bot response
        bot_response = get_bot_response(prompt, meme_df, collection, embedding_model)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        st.error("Bot is not available due to data loading issues.")