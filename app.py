import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import os

# --- 1. CONFIGURATION ---
# Configures the Google API key from Streamlit's Secrets.
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except AttributeError:
    st.warning("Please set your GOOGLE_API_KEY as a Streamlit Secret!", icon="⚠️")

# --- 2. CACHED FUNCTIONS ---
# These functions will run only once to load the data and models.

@st.cache_resource
def load_data():
    """Loads the dataset, embeddings, and IDs from files."""
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
    """Sets up the in-memory ChromaDB vector database."""
    client = chromadb.Client()
    collection_name = "memes"
    
    # Get or create the collection
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)
    
    # Add data to the collection only if it's empty
    if collection.count() == 0:
        collection.add(
            embeddings=_embeddings.tolist(),
            ids=[str(i) for i in _ids]
        )
    return collection

# --- 3. RAG CORE FUNCTION ---
def get_bot_response(user_query, df, collection):
    """
    Takes a user query, retrieves relevant memes, and generates a bot response.
    """
    # Create an embedding for the user's query using the Google API
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    # Query the vector database to find the top 3 most similar memes
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    retrieved_ids = results['ids'][0]
    
    # Get the context for the retrieved memes
    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)
        
    # Construct the master prompt for the Gemini model
    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, arrogant, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    Your replies must be very short and dismissive, and directly use or reference the provided memes.
    
    USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE:
    1. {retrieved_contexts[0]}
    2. {retrieved_contexts[1]}
    3. {retrieved_contexts[2]}

    Generate a short, high-attitude reply that cleverly uses ONE of these memes to respond.
    """
    
    # Call the Generative AI model to get the final response
    generative_model = genai.GenerativeModel('gemini-pro')
    response = generative_model.generate_content(prompt)
    
    return response.text

# --- 4. STREAMLIT USER INTERFACE ---

st.title("🗣️ Meme Mowa Chat")
st.markdown("Nenu chaala planned ga untaa nandi... adagandi.")

# Load the data and set up the database
meme_df, embeddings, ids = load_data()

# Only run the app logic if the data was loaded successfully
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input from the chat box
    if prompt := st.chat_input("Em sangathulu?"):
        # Display the user's message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get and display the bot's response
        bot_response = get_bot_response(prompt, meme_df, collection)
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})