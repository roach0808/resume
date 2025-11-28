# Minimal Streamlit test to isolate segmentation fault
import streamlit as st
import os
from config import get_openai_api_key

def main():
    st.set_page_config(page_title="Minimal Test", layout="wide")
    st.title("🧪 Minimal Streamlit Test")
    
    # Test basic functionality
    st.success("✅ Streamlit is working!")
    
    # Test API key retrieval
    try:
        api_key = get_openai_api_key()
        st.success("✅ OpenAI API Key found")
    except ValueError as e:
        st.warning(f"⚠️ OpenAI API Key not found: {str(e)}")
    
    # Test basic imports
    try:
        import sys
        st.info(f"🐍 Python version: {sys.version}")
    except Exception as e:
        st.error(f"❌ System error: {e}")
    
    # Test if we can import heavy libraries
    st.subheader("Testing Heavy Imports")
    
    if st.button("Test ChromaDB Import"):
        try:
            import chromadb
            st.success("✅ ChromaDB import successful")
        except Exception as e:
            st.error(f"❌ ChromaDB import failed: {e}")
    
    if st.button("Test Sentence Transformers Import"):
        try:
            from sentence_transformers import SentenceTransformer
            st.success("✅ Sentence Transformers import successful")
        except Exception as e:
            st.error(f"❌ Sentence Transformers import failed: {e}")
    
    if st.button("Test LangChain Import"):
        try:
            from langchain_community.document_loaders import PyPDFLoader
            st.success("✅ LangChain import successful")
        except Exception as e:
            st.error(f"❌ LangChain import failed: {e}")

if __name__ == "__main__":
    main()
