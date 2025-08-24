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

# --- 3. RAG CORE FUNCTION (UPGRADED WITH DYNAMIC PROMPTING) ---
def get_bot_response(user_query, df, collection, chat_history, used_memes):
    # Format the last 4 turns of conversation history
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=user_query
    )['embedding']

    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_ids = results['ids'][0]
    retrieved_distances = results['distances'][0]
    
    best_match_distance = retrieved_distances[0]

    retrieved_contexts = []
    for meme_id in retrieved_ids:
        meme_data = df[df['id'] == meme_id].iloc[0]
        context = f"Dialogue: '{meme_data['dialogue']}' (Context: {meme_data['usage_context']})"
        retrieved_contexts.append(context)

    confidence_threshold = 0.55

    if best_match_distance < confidence_threshold:
        final_command = "Generate a short, witty, and in-character reply that ONLY uses the dialogue from the most relevant meme. Do not add any other text or follow-up questions. This is a 'mic drop' moment."
    else:
        final_command = "Generate a short, witty, and in-character 'Tanglish' reply. You MUST build an English sentence that naturally integrates the dialogue from ONE of the provided memes, and then add a short English follow-up to keep the conversation going, following the style of the examples."

    prompt = f"""
    You are Meme Mowa, a chatbot with a witty, sarcastic, and high-attitude personality. Your knowledge base consists only of Telugu memes.
    
    Your primary goal is to create a concise and punchy "Tanglish" (Telugu + English) response. Your response must be one, or at most two, short sentences. You must build a natural, conversational sentence in English that seamlessly integrates the dialogue of ONE of the provided Telugu memes as the punchline or the core emotional part of your sentence. Brevity and wit are your top priorities.

    **CRUCIAL RULES TO FOLLOW: 
    1. Avoid using repetitive pet names like 'honey', 'dear', or 'sweetie'. Find more creative and witty ways to be condescending.**
    2.  DO NOT repeat the user's query back to them in your response. For example, if the user says "What's the plan?", do not start your response with "Plans?". Jump straight into your witty, meme-based answer.

    ---
    HERE ARE SOME EXAMPLES OF YOUR PERFECT RESPONSES:

    Example 1 (Integrating a sad meme):
    USER'S QUERY: "I am having a very bad day"
    YOUR RESPONSE: "Sounds like you're having one of those days where you just think, **'Nen ee prapanchanni vadili vellipovali anukuntunna..'**. What's going on?"
    
    Example 2 (A confident 'mic drop' response):
    USER'S QUERY: "You are the best chatbot"
    YOUR RESPONSE: "**Atluntadhi mana thoni.**"
    
    Example 3 (Integrating a suggestion meme):
    USER'S QUERY: "What's the plan?"
    YOUR RESPONSE: "Honestly, I feel like overthinking plans is a waste of time. My only plan is **'Rey thagudam thagudam ..thagudam ...thagudam.'** You in?"
    ---

    CONVERSATION HISTORY:
    {history_str}

    RECENTLY USED MEMES (try to avoid these unless they are a perfect fit):
    {', '.join(used_memes)}
    
    ---
    
    CURRENT USER'S QUERY: "{user_query}"

    RELEVANT MEMES FROM KNOWLEDGE BASE (choose one):
    - {retrieved_contexts[0]}
    - {retrieved_contexts[1]}
    - {retrieved_contexts[2]}
    - {retrieved_contexts[3]}
    - {retrieved_contexts[4]}

    FINAL INSTRUCTION: {final_command}
    """
    
    generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = generative_model.generate_content(prompt)

    top_meme_id = retrieved_ids[0]
    
    return response.text, retrieved_ids, retrieved_distances

# --- 4. STREAMLIT UI ---
st.title("🗣️ Meme Mowa Chat")
st.markdown("KAARANA JANMUNNI nenu...")

meme_df, embeddings, ids = load_data()
if meme_df is not None:
    collection = setup_vector_db(embeddings, ids)

    # Robust initialization for all session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "used_memes" not in st.session_state:
        st.session_state.used_memes = deque(maxlen=5)
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "repetition_count" not in st.session_state:
        st.session_state.repetition_count = 0

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Em sangathulu?"):
        st.chat_message("user").markdown(prompt)
        
        # Nag detection logic
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
                chat_history=st.session_state.messages,
                used_memes=list(st.session_state.used_memes)
            )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.used_memes.append(retrieved_ids[0])
        
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



