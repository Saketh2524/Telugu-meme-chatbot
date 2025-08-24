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
import re # Import the regular expression library

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

# --- 3. RAG CORE FUNCTION (DEFINITIVE VERSION) ---
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

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality. Your knowledge base consists only of Telugu memes.

    Your primary goal is to create a "Tanglish" (Telugu + English) response. You must build a natural, conversational sentence in English that seamlessly integrates the dialogue of ONE of the provided Telugu memes as the punchline.

    A CRUCIAL SKILL is to generate probing follow-up questions. After your main response (based on the first meme), you can look at the OTHER retrieved memes for inspiration to ask a clever follow-up question that connects the ideas.

    *** CRUCIAL FORMATTING RULE ***
    When you include a Telugu meme dialogue in your response, you ABSOLUTELY MUST enclose it in special tags: ||MEME||dialogue||/MEME||.

    CONVERSATION HISTORY:
    {history_str}

    RECENTLY USED MEMES (AVOID THESE):
    {', '.join(used_memes)}
    
    CURRENT USER'S QUERY: "{user_query}"

    RELEVANT MEMES (use #1 for response, others for follow-up inspiration):
    - {retrieved_contexts[0]}
    - {retrieved_contexts[1]}
    - {retrieved_contexts[2]}

    *** STRICT BEHAVIORAL RULES ***
    1. YOUR ENTIRE RESPONSE MUST BE 1-2 SENTENCES MAXIMUM.
    2. YOU ABSOLUTELY MUST NOT REPEAT THE USER'S QUERY.
    3. YOU ABSOLUTELY MUST NOT USE PET NAMES LIKE 'HONEY' OR 'DARLING'.
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)
    
    top_meme_id = retrieved_ids[0]
    
    return response.text, retrieved_ids, retrieved_distances, detected_emotion

def detect_emotion(user_query):
    # This function remains the same
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
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
            bot_response = "||MEME||eyy marcus endhuku ra anni sarlu phone chesthunnav||/MEME||"
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
            formatted_response = re.sub(r'\|\|MEME\|\|(.*?)\|\|/MEME\|\|', r'**\1**', bot_response)
            st.markdown(formatted_response)
            
            with st.expander("🤔 See Bot's Thought Process"):
                st.write(f"**Detected Emotion:** {detected_emotion or 'N/A'}")
                st.write("**Raw Bot Response (before formatting):**")
                st.text(bot_response)
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
