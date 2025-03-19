import os
import streamlit as st
import json
from langchain_groq import ChatGroq

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key
DEFAULT_GROQ_API_KEY = "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"

def initialize_rag_system(groq_api_key, groq_model, temperature, max_tokens):
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def process_transaction_message(message, llm):
    system_prompt = (
        "Check if this message is a valid transaction message or not. "
        "If valid, extract the following data: Amount, Transaction Type, Bank Name, Card Type, "
        "Merchant, Transaction Mode, Transaction Date, Reference Number, and tag. "
        "Return only valid JSON output without any extra text."
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    
    response = llm.invoke(input_prompt)

    if response is None:
        return None  # Handle missing response

    try:
        json_output = json.loads(response.content)  # Ensure valid JSON
        return json_output
    except (AttributeError, json.JSONDecodeError):
        return None  # Handle invalid JSON response

st.title("Transaction Message Extractor")

llm = initialize_rag_system(DEFAULT_GROQ_API_KEY, "llama3-70b-8192", 0.5, 1024)
if llm:
    st.success("RAG system initialized successfully!")
    user_input = st.text_area("Enter transaction message:")
    
    if st.button("Extract Details"):
        if user_input:
            result = process_transaction_message(user_input, llm)
            if result:
                st.json(result)  # Display JSON response
            else:
                st.error("Invalid or non-transactional message. Please try again.")
        else:
            st.warning("Please enter a transaction message.")
