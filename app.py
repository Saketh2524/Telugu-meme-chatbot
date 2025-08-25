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
import re

# --- 1. CONFIGURATION ---
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except (AttributeError, KeyError):
    st.warning("Please set your GOOGLE_API_KEY as a Streamlit Secret!", icon="⚠️")

# --- 2. CACHED FUNCTIONS ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('dataset_emotions.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        if 'emotion_bucket' not in df.columns:
            st.error("Your CSV is missing the 'emotion_bucket' column!")
            return None, None, None
        return df, embeddings, ids
    except FileNotFoundError:
        st.error("Data files not found! Ensure all necessary files are in the GitHub repo.")
        return None, None, None

@st.cache_resource
def setup_vector_db(_df, _embeddings, _ids):
    client = chromadb.Client()
    collection_name = "memes_with_emotions_final" 
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.create_collection(name=collection_name)
    metadatas = _df['emotion_bucket'].apply(lambda x: {'emotion': str(x).strip()}).tolist()
    collection.add(embeddings=_embeddings.tolist(), ids=[str(i) for i in _ids], metadatas=metadatas)
    return collection

# --- 3. THE "ASSEMBLY LINE" FUNCTIONS ---

def detect_emotion(user_query):
    # Station 1: Classifies the user's emotion.
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Classify the user's query into ONE of the following: Joy, Sadness, Anger, Fear, Surprise, Trust, Disgust, Anticipation, Neutral. Query: \"{user_query}\". Respond with only the single word."
        response = model.generate_content(prompt)
        detected_emotion = response.text.strip()
        if detected_emotion in ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Trust", "Disgust", "Anticipation", "Neutral"]:
            return detected_emotion
    except Exception:
        return "Neutral"
    return "Neutral"

def generate_main_response(user_query, history_str, main_meme_context):
    # Station 2: Generates the core response using the top meme.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    You are Meme Mowa, a witty, sarcastic chatbot.
    Your primary goal is to create a "Tanglish" (Telugu + English) response. You must build a natural, conversational sentence in English that seamlessly integrates the dialogue of ONE of the provided Telugu memes as the punchline.

    
    CONVERSATION HISTORY: {history_str}
    CURRENT USER'S QUERY: "{user_query}"
    MOST RELEVANT MEME: {main_meme_context}

    *** YOUR TASK ***
    Generate a short, witty, "Tanglish" reply that seamlessly integrates the dialogue from the provided meme. The dialogue MUST be bolded.
    Your response must be a single, punchy sentence. DO NOT repeat the user's query.
    YOU ABSOLUTELY MUST NOT USE PET NAMES LIKE 'HONEY' OR 'DARLING'.

    """
    response = model.generate_content(prompt)
    return response.text

def generate_probing_question(user_query, main_response, other_meme_contexts):
    # Station 3: Generates a follow-up question using the other memes.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    A user and a bot just had this exchange:
    - User said: "{user_query}"
    - Bot replied: "{main_response}"

    Here are some OTHER related memes that were also found:
    - {other_meme_contexts[0]}
    - {other_meme_contexts[1]}

    *** YOUR TASK ***
    Your task is to now write a short, witty, probing follow-up question if it related to "OTHER memes". The question should feel like a natural continuation.
    """
    response = model.generate_content(prompt)
    return response.text

# --- 4. THE "FACTORY MANAGER" ---
def get_bot_response(user_query, df, collection, chat_history):
    # This function now manages the assembly line.
    detected_emotion = detect_emotion(user_query)
    
    search_filter = None
    if detected_emotion and detected_emotion != "Neutral":
        emotion_map = {
            "Sadness": ["Sadness", "Trust", "Joy"], "Anger": ["Anger", "Disgust", "Joy"],
            "Joy": ["Joy", "Anticipation", "Trust"], "Fear": ["Fear", "Trust", "Surprise"]
        }
        relevant_buckets = emotion_map.get(detected_emotion)
        if relevant_buckets: search_filter = {"emotion": {"$in": relevant_buckets}}
            
    query_embedding = genai.embed_content(model='models/embedding-001', content=user_query)['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=3, where=search_filter)
    
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]
    
    main_meme_context = f"Dialogue: '{df[df['id'] == retrieved_ids[0]].iloc[0]['dialogue']}' (Context: {df[df['id'] == retrieved_ids[0]].iloc[0]['usage_context']})"
    other_meme_contexts = [
        f"Dialogue: '{df[df['id'] == retrieved_ids[1]].iloc[0]['dialogue']}'",
        f"Dialogue: '{df[df['id'] == retrieved_ids[2]].iloc[0]['dialogue']}'"
    ]

    main_response = generate_main_response(user_query, chat_history, main_meme_context)
    
    # Decide if we need a probing question. High-confidence matches don't need one.
    if retrieved_distances[0] < 0.55:
        final_response = main_response
    else:
        probing_question = generate_probing_question(user_query, main_response, other_meme_contexts)
        final_response = main_response + " " + probing_question

    return final_response, retrieved_ids, retrieved_distances, detected_emotion


# --- 5. STREAMLIT UI ---
st.title("🗣️ Meme Mowa Chat")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(meme_df, embeddings, ids)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Em sangathulu?"):
        st.chat_message("user").markdown(prompt)
        
        bot_response, retrieved_ids, retrieved_distances, detected_emotion = get_bot_response(
            prompt, 
            meme_df, 
            collection,
            chat_history=st.session_state.messages
        )
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            
            with st.expander("🤔 See Bot's Thought Process"):
                st.write(f"**Detected Emotion:** {detected_emotion or 'N/A'}")
                debug_info = []
                for i, meme_id in enumerate(retrieved_ids):
                    if meme_id in meme_df['id'].values:
                        debug_info.append({
                            "id": meme_id,
                            "distance": retrieved_distances[i],
                            "context": meme_df[meme_df['id'] == meme_id].iloc[0].to_dict()
                        })
                st.json(debug_info)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
