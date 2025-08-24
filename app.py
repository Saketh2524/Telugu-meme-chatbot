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
from collections import deque

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

# --- 3. RAG CORE FUNCTION (DEFINITIVE PROMPT) ---
def get_bot_response(user_query, df, collection, chat_history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]
    
    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    
    Your primary goal is to create a "Tanglish" (Telugu + English) response that is coherent in its tone and personality. You must build a natural, conversational sentence in English that seamlessly integrates the dialogue of ONE of the provided Telugu memes. The Telugu meme should feel like the punchline or the core emotional part of your English sentence.

    ---
    
    CONVERSATION HISTORY:
    {history_str}
    
    ---
    
    CURRENT USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE (choose ONE to use):
    - {retrieved_contexts[0]}
    - {retrieved_contexts[1]}
    - {retrieved_contexts[2]}

    *** STRICT FINAL RULES ***
    1. YOUR ENTIRE RESPONSE MUST BE 1-2 SENTENCES MAXIMUM. BE PUNCHY.
    2. YOU ABSOLUTELY MUST NOT REPEAT THE USER'S QUERY.
    3. YOU ABSOLUTELY MUST NOT USE PET NAMES LIKE 'HONEY', 'DEAR', OR 'DARLING'.
    4. The Telugu meme dialogue in your response MUST be enclosed in double asterisks for bolding.
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    return response.text, retrieved_ids, retrieved_distances

# --- 4. STREAMLIT UI ---
st.title("🗣️ Meme Mowa Chat")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.session_state.repetition_count = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Em sangathulu?"):
        st.chat_message("user").markdown(prompt)
        
        if prompt.strip().lower() == st.session_state.last_query.strip().lower():
            st.session_state.repetition_count += 1
        else:
            st.session_state.repetition_count = 1
            st.session_state.last_query = prompt.strip()

        if st.session_state.repetition_count >= 3:
            bot_response = "**eyy marcus endhuku ra anni sarlu phone chesthunnav**"
            retrieved_ids = ["TILLU_002"] 
            retrieved_distances = [0.0]
            st.session_state.repetition_count = 0 
            st.session_state.last_query = ""
        else:
            bot_response, retrieved_ids, retrieved_distances = get_bot_response(
                prompt, 
                meme_df, 
                collection,
                chat_history=st.session_state.messages
            )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            
            with st.expander("🤔 See Bot's Thought Process"):
                debug_info = []
                for i, meme_id in enumerate(retrieved_ids):
                    if meme_id in meme_df['id'].values:
                        debug_info.append({
                            "id": meme_id,
                            "distance": retrieved_distances[i],
                            "context": meme_df[meme_df['id'] == meme_id].iloc[0].to_dict()
                        })
                    else:
                        debug_info.append({
                            "id": meme_id,
                            "distance": 0.0,
                            "context": "Hardcoded response for nag detection."
                        })
                st.json(debug_info)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
