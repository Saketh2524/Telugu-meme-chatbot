# Magic fix for the sqlite3 version issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# All our library imports
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import os

st.title("✅ Test 3: API Key")

# This block tests the API Key configuration
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    st.success("If you can see this, the API Key was found and configured successfully!")
except (KeyError, AttributeError):
    st.error("ERROR: The GOOGLE_API_KEY was not found. Please make sure you have set it correctly in the Streamlit Secrets manager.", icon="🚨")
