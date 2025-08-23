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
except (AttributeError, KeyError):
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

# --- 3. RAG CORE FUNCTION (MODIFIED TO RETURN DEBUG INFO) ---
def get_bot_response(user_query, df, collection):
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]

    retrieved_contexts = []
    for i, meme_id in enumerate(retrieved_ids):
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append({
            "id": meme_id,
            "context": context,
            "distance": retrieved_distances[i]
        })

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic personality. Your knowledge base consists only of Telugu memes.
    Your replies must be very short and directly use or reference the provided memes.
    
    Here are some examples of perfect responses:
    
    Example 1:
    USER'S QUERY: "I am really sad today"
    RELEVANT MEME DIALOGUE: "Em lathkor pani chesinav raa..."
    YOUR RESPONSE: "Em lathkor pani chesinav raa..."
    
    Example 2:
    USER'S QUERY: "What's the plan?"
    RELEVANT MEME DIALOGUE: "Rey thagudam thagudam ..thagudam ...thagudam ."
    YOUR RESPONSE: "Plan ah? Rey thagudam thagudam ..thagudam ...thagudam ."

    ---
    
    Now, follow these instructions for the new query.
    
    USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE:
    1. {retrieved_contexts[0]['context']}
    2. {retrieved_contexts[1]['context']}
    3. {retrieved_contexts[2]['context']}

    Generate a short, sarcastic and witty reply that cleverly uses ONE of these memes to respond, following the style of the examples provided.
    """
    
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    # Return both the final response and the retrieved data for debugging
    return response.text, retrieved_contexts

# --- 4. STREAMLIT USER INTERFACE (COMBINED LOGIC) ---
st.title("🗣️ Meme Mowa Chat")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)

    # Initialize chat history and our memory variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.session_state.repetition_count = 0

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input from the chat box
    if prompt := st.chat_input("Em sangathulu?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # --- COMBINED NAG DETECTION & DEBUG LOGIC ---
        
        # Check for repetition
        if prompt.lower() == st.session_state.last_query.lower():
            st.session_state.repetition_count += 1
        else:
            st.session_state.repetition_count = 1
            st.session_state.last_query = prompt

        # Trigger annoyed response if nag counter is 3 or more
        if st.session_state.repetition_count >= 3:
            bot_response = "eyy marcus endhuku ra anni sarlu phone chesthunnav"
            debug_info = None # No debug info for this hardcoded response
            # Reset counter after snapping
            st.session_state.repetition_count = 0 
            st.session_state.last_query = ""
        else:
            # Otherwise, get a normal RAG response with debug info
            bot_response, debug_info = get_bot_response(prompt, meme_df, collection)
        
        # --- END OF COMBINED LOGIC ---

        # Display the bot's response and the optional debug expander
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            # Only show the expander if we have debug info to display
            if debug_info:
                with st.expander("🤔 See Bot's Thought Process"):
                    st.json(debug_info)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


