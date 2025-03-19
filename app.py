import os
import streamlit as st
from langchain_groq import ChatGroq
import json

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
        "If valid, extract the following data: Amount, Transaction Type, Bank Name, Card Type, paied to whom,marchent, Transaction Mode, Transaction Date, Reference Number, and tag."
        """example 1: messsage - Rs.105.00 spent on your SBI Credit Card ending with 5775 at Auto Fuel Station on 18-03-25 via UPI (Ref No. 507775912830). Trxn. not done by you? Report at https://sbicard.com/Dispute , output :{
        "Amount":105,
        "Transaction Type":"Debit",
        "Bank Name":"SBI",
        "Card Type":"Credit Card",
        "marchent":"Auto Fuel Station",
        "paied to whom":"Auto Fuel Station",
        "Transaction Mode":"Credit Card",
        "Transaction Date":"19-03-25",
        "Reference Number":"507775912830",
        "tag":["Transport"]
        } """

        """example 2 : ICICI Bank Acct XX337 debited for Rs 500.00 on 17-Jan-25; BPCL Ufill 2 credited. UPI:501714256060. Call 18002662 for dispute. SMS BLOCK 337 to 9215676766.
        output :{
        "Amount":500,
        "Transaction Type":"Debit",
        "Bank Name":"ICICI",
        "Card Type":NULL,
        "Merchant":"BPCL",
        "paied to whom":"BPCL", 
        "Transaction Mode":"UPI",
        "Transaction Date":"19-03-25",
        "Reference Number":NULL,
        "tag":["Transport"]}""" 
        "Tag meaning which category of spending, if amazon then shopping etc, if zomato then eating"
        "return null if it is a personal messege, bill payment reminder, ads, or anything non transactional"  
        "Just give the json output, Don't say anything else , if there is no output then don't predict, say it is null" 
        "$3000 will be credited in your bank account - this looks like personal messege , so ignore it"
        "ignore all messege like personal messege, ads, loan ads, bill alert, spam and all, just focus on valid transaction messege")
        
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)

    return response

st.title("Transaction Message Extractor")

llm = initialize_rag_system(DEFAULT_GROQ_API_KEY, "llama3-70b-8192", 0.5, 1024)
if llm:
    st.success("RAG system initialized successfully!")
    user_input = st.text_area("Enter transaction message:")
    if st.button("Extract Details"):
        if user_input:
            result = process_transaction_message(user_input, llm)
            st.json(result.content)

        else:
            st.warning("Please enter a transaction message.")
