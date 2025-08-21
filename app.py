# This is the magic fix for the sqlite3 version issue on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now we test the rest of the imports
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import os

st.title("✅ Test 2: Imports")
st.success("If you can see this, all Python libraries were imported successfully!")