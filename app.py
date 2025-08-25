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
import random


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

# --- 3. RAG CORE FUNCTIONS ---

def detect_emotion(user_query):
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

def get_probing_candidates(retrieved_contexts, retrieved_distances, threshold_meme=0.45, threshold_general=0.65):
    """
    Decide if probing is allowed and whether it should use a meme line.
    Returns: (probe_allowed, probe_with_meme, probing_contexts)
    """
    probe_allowed = False
    probe_with_meme = False
    probing_contexts = []

    for i, d in enumerate(retrieved_distances[1:], start=1):  # skip punchline meme at index 0
        if d < threshold_meme:
            probe_allowed = True
            probe_with_meme = True
            probing_contexts.append(retrieved_contexts[i])
        elif d < threshold_general:
            probe_allowed = True
            # only general English probing allowed here

    return probe_allowed, probe_with_meme, probing_contexts

def get_bot_response(user_query, df, collection, chat_history, used_memes):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
    detected_emotion = detect_emotion(user_query)
    
    search_filter = None
    if detected_emotion and detected_emotion != "Neutral":
        emotion_map = {
            "Sadness": ["Sadness", "Trust", "Joy"], 
            "Anger": ["Anger", "Disgust", "Joy"],
            "Joy": ["Joy", "Anticipation", "Trust"], 
            "Fear": ["Fear", "Trust", "Surprise"]
        }
        relevant_buckets = emotion_map.get(detected_emotion)
        if relevant_buckets: 
            search_filter = {"emotion": {"$in": relevant_buckets}}
            
    query_embedding = genai.embed_content(model='models/embedding-001', content=user_query)['embedding']
    # Query Chroma with more results for variety
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=7,
    where=search_filter
    )

retrieved_ids = results['ids'][0]
retrieved_distances = results['distances'][0]

# Randomize selection of top memes
if len(retrieved_ids) > 3:
    indices = list(range(len(retrieved_ids)))
    random.shuffle(indices)
    selected_indices = indices[:3]
    retrieved_ids = [retrieved_ids[i] for i in selected_indices]
    retrieved_distances = [retrieved_distances[i] for i in selected_indices]

# Gather the meme texts
memes = [df.iloc[int(idx)]['dialogue'] for idx in retrieved_ids]
    
    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    # --- Probing Decision ---
    probe_allowed, probe_with_meme, probing_contexts = get_probing_candidates(retrieved_contexts, retrieved_distances)

    if probe_with_meme:
        probe_instruction = f"You may ask ONE probing follow-up using ONLY these meme dialogues if they fit naturally:\n- {'; '.join(probing_contexts)}"
    elif probe_allowed:
        probe_instruction = "You may ask ONE probing follow-up in plain English if it feels relevant."
    else:
        probe_instruction = "Do NOT ask a probing follow-up. Only respond with the meme punchline."

    prompt = f"""
    You are Meme Mowa, a chatbot that responds using Telugu memes.
    Your personality is witty, sarcastic, and high-attitude.
    Your primary goal is to create a "Tanglish" (Telugu + English) response. 
    You must build a natural, conversational sentence in English that seamlessly integrates 
    the dialogue of ONE of the provided Telugu memes as the punchline.

    *** CRUCIAL FORMATTING RULE ***
    When you include a Telugu meme dialogue in your response, you MUST enclose it in special tags: ||MEME||dialogue||/MEME||.

    CONVERSATION HISTORY: {history_str}
    RECENTLY USED MEMES (AVOID THESE): {', '.join(used_memes)}
    CURRENT USER'S QUERY: "{user_query}"
    RELEVANT MEMES:
    - {retrieved_contexts[0]}
    - {retrieved_contexts[1]}
    - {retrieved_contexts[2]}

    *** STRICT BEHAVIORAL RULES ***
    1. YOUR ENTIRE RESPONSE MUST BE 1-2 SENTENCES MAXIMUM.
    2. DO NOT repeat the user's query.
    3. DO NOT use pet names like 'honey' or 'darling'.
    4. {probe_instruction}
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    return response.text, retrieved_ids, retrieved_distances, detected_emotion

# --- 4. STREAMLIT UI ---
st.title("🗣️ Meme Mowa")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(meme_df, embeddings, ids)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.used_memes = deque(maxlen=5)
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
            bot_response = "||MEME||eyy marcus endhuku ra anni sarlu phone chesthunnav||/MEME||"
            retrieved_ids = ["TILLU_002"]; retrieved_distances = [0.0]; detected_emotion = "Anger"
            st.session_state.repetition_count = 0; st.session_state.last_query = ""
        else:
            bot_response, retrieved_ids, retrieved_distances, detected_emotion = get_bot_response(
                prompt, meme_df, collection, st.session_state.messages, list(st.session_state.used_memes)
            )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.used_memes.append(retrieved_ids[0])
        
        with st.chat_message("assistant"):
            # --- CODE-ENFORCED RULES ---
            interim_response = re.sub(r'(?i)\b(honey|darling|sweetie)\b[, ]*', '', bot_response).strip()
            formatted_response = re.sub(r'\|\|MEME\|\|(.*?)\|\|/MEME\|\|', r'**\1**', interim_response)
            
            st.markdown(formatted_response)
            
            with st.expander("🤔 See Bot's Thought Process"):
                st.write(f"**Detected Emotion:** {detected_emotion or 'N/A'}")
                st.write("**Raw Bot Response (from AI):**"); st.text(bot_response)
                debug_info = []
                for i, meme_id in enumerate(retrieved_ids):
                    if meme_id in meme_df['id'].values:
                        debug_info.append({
                            "id": meme_id,
                            "distance": retrieved_distances[i],
                            "context": meme_df[meme_df['id'] == meme_id].iloc[0].to_dict()
                        })
                st.json(debug_info)
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})





