import os
from openai import OpenAI
import streamlit as st

def get_openai_api_key():
    """
    Get OpenAI API key from Streamlit secrets (for cloud deployment) 
    with fallback to environment variable (for local development).
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If API key is not found in secrets or environment
    """
    try:
        # Try to get from Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    # Fallback to environment variable (for local development)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # If neither is available, raise an error
    raise ValueError(
        "OpenAI API key not found. Please set it in Streamlit secrets (for cloud) "
        "or as OPENAI_API_KEY environment variable (for local development)."
    )

def get_openai_llm_response(messages: list[dict], temperature: float = 0.1, max_tokens = 512) -> str:
    """
    Get the response from OpenAI LLM using the provided prompt.
    :param prompt: The prompt to send to the LLM.
    :return: The response from the LLM.
    """

    # Get the OpenAI API key
    api_key = get_openai_api_key()

    # Initialize the OpenAI client and get the response
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()