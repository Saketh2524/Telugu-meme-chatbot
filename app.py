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
        df = pd.read_csv('dataset_emotions.csv')
        embeddings = np.load('embeddings.npy')
        ids = np.load('ids.npy', allow_pickle=True)
        if 'emotion_bucket' not in df.columns:
            st.error("Your CSV is missing the 'emotion_bucket' column! Please add it.")
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
    
    collection.add(
        embeddings=_embeddings.tolist(),
        ids=[str(i) for i in _ids],
        metadatas=metadatas
    )
    return collection

# --- 3. RAG CORE FUNCTION ---
def get_bot_response(user_query, df, collection, chat_history, used_memes):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    detected_emotion = detect_emotion(user_query)
    search_filter = None
    if detected_emotion:
        emotion_map = {
            "Sadness": ["Sadness", "Trust", "Joy"], "Anger": ["Anger", "Disgust", "Joy"],
            "Joy": ["Joy", "Anticipation", "Trust"], "Fear": ["Fear", "Trust", "Surprise"]
        }
        relevant_buckets = emotion_map.get(detected_emotion)
        if relevant_buckets:
            search_filter = {"emotion": {"$in": relevant_buckets}}
            
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(
        query_embeddings=[query_embedding], n_results=5, where=search_filter
    )
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]
    best_match_distance = retrieved_distances[0]
    confidence_threshold = 0.55

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    # --- DYNAMIC PROMPT LOGIC ---
    if best_match_distance < confidence_threshold:
        # High-confidence "Mic Drop" prompt
        prompt = f"""
        You are Meme Mowa. Your task is to respond using ONLY the dialogue from the most relevant meme provided.
        The user said: "{user_query}"
        The most relevant meme is: {retrieved_contexts[0]}
        Your response must be the Telugu dialogue, enclosed in double asterisks for bolding. Nothing else.
        """
    else:
        # Lower-confidence "Conversational" prompt with multi-meme logic
        prompt = f"""
        You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality.
        Your primary goal is to create a punchy "Tanglish" response that integrates a Telugu meme.
        A CRUCIAL SKILL is to generate probing follow-up questions. After your main response (based on the first meme), look at the OTHER retrieved memes for inspiration to ask a clever follow-up question.

        ---
        EXAMPLE of a compound response:
        USER'S QUERY: "My friend is ignoring my calls."
        YOUR RESPONSE: "Being ignored is the worst, it makes you feel like **'Naa paatiki nenu maadipoyina masala dosa tintunte..'**. But is this a 'my friend is busy' problem or a **'Chedagetthera yedava'** kind of problem?"
        ---

        CONVERSATION HISTORY: {history_str}
        RECENTLY USED MEMES (AVOID THESE): {', '.join(used_memes)}
        CURRENT USER'S QUERY: "{user_query}"
        RELEVANT MEMES (use #1 for response, others for follow-up):
        - {retrieved_contexts[0]}
        - {retrieved_contexts[1]}
        - {retrieved_contexts[2]}

        *** STRICT BEHAVIORAL RULES ***
        1. YOUR ENTIRE RESPONSE MUST BE 1-2 SENTENCES MAXIMUM.
        2. DO NOT repeat the user's query.
        3. DO NOT use pet names like 'honey' or 'darling'.
        """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    return response.text, retrieved_ids, retrieved_distances, detected_emotion

def detect_emotion(user_query):
    # This function remains the same
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Classify the user's query into ONE of the following: 
        Joy, Sadness, Anger, Fear, Surprise, Trust, Disgust, Anticipation, Neutral.
        Query: "{user_query}"
        Respond with only the single word.
        """
        response = model.generate_content(prompt)
        detected_emotion = response.text.strip()
        if detected_emotion in ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Trust", "Disgust", "Anticipation", "Neutral"]:
            return detected_emotion
    except Exception:
        return None 
    return None

# --- 4. STREAMLIT UI ---
# This section remains the same
st.title("🗣️ Meme Mowa Chat")
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
            bot_response = "**eyy marcus endhuku ra anni sarlu phone chesthunnav**"
            retrieved_ids = ["TILLU_002"] 
            retrieved_distances = [0.0]
            detected_emotion = "Anger"
            st.session_state.repetition_count = 0 
            st.session_state.last_query = ""
        else:
            bot_response, retrieved_ids, retrieved_distances, detected_emotion = get_bot_response(
                prompt, 
                meme_df, 
                collection,
                chat_history=st.session_state.messages,
                used_memes=list(st.session_state.used_memes)
            )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.used_memes.append(retrieved_ids[0])
        
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
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
