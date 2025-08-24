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
        # --- THIS IS THE UPDATED FILENAME ---
        df = pd.read_csv('dataset_emotions.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        if 'emotion_bucket' not in df.columns:
            st.error("Your CSV is missing the 'emotion_bucket' column! Please add it.")
            return None, None, None
        return df, embeddings, ids
    except FileNotFoundError:
        st.error("Data files not found! Ensure 'dataset_emotions.csv', 'embeddings.npy', and 'ids.npy' are in the GitHub repo.")
        return None, None, None

@st.cache_resource
def setup_vector_db(_df, _embeddings, _ids):
    client = chromadb.Client()
    collection_name = "memes_with_emotions_v2" 
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name)
    
    metadatas = _df['emotion_bucket'].apply(lambda x: {'emotion': str(x).strip()}).tolist()
    
    collection.add(
        embeddings=_embeddings.tolist(),
        ids=[str(i) for i in _ids],
        metadatas=metadatas
    )
    return collection

# --- 3. RAG CORE FUNCTION ---
def get_bot_response(user_query, df, collection, chat_history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    detected_emotion = detect_emotion(user_query)
    search_filter = None
    if detected_emotion:
        emotion_map = {
            "Sadness": ["Sadness", "Trust", "Joy"],
            "Anger": ["Anger", "Disgust", "Joy"],
            "Joy": ["Joy", "Anticipation", "Trust"],
            "Fear": ["Fear", "Trust", "Surprise"]
        }
        relevant_buckets = emotion_map.get(detected_emotion)
        if relevant_buckets:
            search_filter = {"emotion": {"$in": relevant_buckets}}
            
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=search_filter
    )
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    Your task is to generate a short, punchy, "Tanglish" response that seamlessly integrates the dialogue from ONE of the provided memes into a natural sentence.
    The user's query has been analyzed and their emotion is likely: {detected_emotion or 'Neutral'}. Use this to inform your tone.
    
    CONVERSATION HISTORY: {history_str}
    CURRENT USER'S QUERY: "{user_query}"
    RELEVANT MEMES (choose ONE):
    - {retrieved_contexts[0]}
    - {retrieved_contexts[1]}
    - {retrieved_contexts[2]}

    *** STRICT FINAL RULES ***
    1. YOUR ENTIRE RESPONSE MUST BE 1-2 SENTENCES MAXIMUM.
    2. YOU ABSOLUTELY MUST NOT REPEAT THE USER'S QUERY.
    3. The Telugu meme dialogue in your response MUST be bolded.
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    return response.text, retrieved_ids, retrieved_distances, detected_emotion

def detect_emotion(user_query):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Classify the user's query into ONE of the following 8 emotions: 
        Joy, Sadness, Anger, Fear, Surprise, Trust, Disgust, Anticipation.
        If the query is neutral or a simple question, respond with "Neutral".
        User query: "{user_query}"
        Respond with only the single emotion word.
        """
        response = model.generate_content(prompt)
        detected_emotion = response.text.strip()
        if detected_emotion in ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Trust", "Disgust", "Anticipation", "Neutral"]:
            return detected_emotion
    except Exception:
        return None 
    return None

# --- 4. STREAMLIT UI ---
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
