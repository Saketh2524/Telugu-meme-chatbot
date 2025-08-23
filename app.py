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

# --- 3. RAG CORE FUNCTION (UPGRADED WITH MEMORY) ---
def get_bot_response(user_query, df, collection, chat_history, used_memes):
    # Format the last 4 turns of conversation history
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_ids = results['ids'][0]

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    Your goal is to be an engaging and funny conversational partner. After delivering your meme-based reply, your goal is to encourage the user to reply. You can do this by either asking a short, witty, open-ended question or by making a sharp, provocative statement that invites a response.

    ---
    HERE ARE SOME EXAMPLES OF YOUR PERFECT RESPONSES:

    Example 1 (Ending with a question):
    USER'S QUERY: "I am really sad today"
    YOUR RESPONSE: "Chala Delicate mind naadhi.. cheppu emaindi?"
    
    Example 2 (Ending with a statement):
    USER'S QUERY: "You are the best chatbot"
    YOUR RESPONSE: "Atluntadhi mana thoni."
    
    Example 3 (Ending with a question):
    USER'S QUERY: "What's the plan?"
    YOUR RESPONSE: "Plan ah? Rey thagudam thagudam ..thagudam ...thagudam. Party ekkada?"
    ---

    CONVERSATION HISTORY:
    {history_str}

    RECENTLY USED MEMES (try to avoid these unless they are a perfect fit):
    {', '.join(used_memes)}
    
    ---
    
    CURRENT USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE (choose one):
    1. {retrieved_contexts[0]}
    2. {retrieved_contexts[1]}
    3. {retrieved_contexts[2]}
    4. {retrieved_contexts[3]}
    5. {retrieved_contexts[4]}

    Generate a short, witty, and in-character reply that cleverly uses ONE of the retrieved memes. Remember to encourage a response from the user with a natural follow-up, following the style of the examples provided.
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)

    top_meme_id = retrieved_ids[0]
    
    return response.text, top_meme_id

# --- 4. STREAMLIT UI ---
st.title("🗣️ Meme Mowa Chat")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.used_memes = deque(maxlen=5) 

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Em sangathulu?"):
        st.chat_message("user").markdown(prompt)
        
        bot_response, used_id = get_bot_response(
            prompt, 
            meme_df, 
            collection,
            chat_history=st.session_state.messages,
            used_memes=list(st.session_state.used_memes)
        )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.used_memes.append(used_id)
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
