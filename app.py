import os
import streamlit as st
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
        "ignore all messege like personal messege, ads, loan ads, bill alert, spam and all, just focus on valid transaction messege"
        "If amount is credited to marchent account means it is debited from my account , so transaction type will be debited"
    
    )
        
    input_prompt = f"{system_prompt}\nMessage: {message}"
    input_prompt = f"{system_prompt}\nMessage: {message}"
    # The invoke method expects a string, PromptValue or a list of BaseMessages.
    # Pass the input_prompt directly as a string.
    response = llm.invoke(input_prompt)
    return response

# Streamlit UI
st.title("Transaction Message Processor")

# API Key Input
api_key = st.text_input("GROQ API Key", value=DEFAULT_GROQ_API_KEY, type="password")

# Model Selection
model = st.selectbox(
    "Select GROQ Model",
    ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
    index=0
)

# Temperature and Max Tokens
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
with col2:
    max_tokens = st.number_input("Max Tokens", min_value=10, max_value=4096, value=1024, step=10)

# Initialize the RAG system
if st.button("Initialize RAG System"):
    llm = initialize_rag_system(api_key, model, temperature, max_tokens)
    if llm:
        st.session_state.llm = llm
        st.success("RAG system initialized successfully!")

# Message Input
message = st.text_area("Enter Transaction Message")

# Process Message
if st.button("Process Message") and message:
    if 'llm' not in st.session_state:
        st.warning("Please initialize the RAG system first.")
    else:
        with st.spinner("Processing transaction message..."):
            result = process_transaction_message(message, st.session_state.llm)
            st.subheader("Extracted Transaction Details:")
            st.code(result.content, language="json")

# Sample Transactions
st.sidebar.header("Sample Transactions")
sample_transactions = [
    "Rs.105.00 spent on your SBI Credit Card ending with 5775 at Auto Fuel Station on 18-03-25 via UPI (Ref No. 507775912830). Trxn. not done by you? Report at https://sbicard.com/Dispute",
    "Rs.2,500.00 debited from your HDFC Bank Account ending with 6789 on 19-03-25 for Amazon purchase (Ref No. 987654321).",
    "Alert: Rs.1,450.00 withdrawn from your ICICI ATM Card ending with 1234 at ICICI ATM on 17-03-25 (Ref No. 123456789)."
]

st.sidebar.subheader("Click to use sample transaction")
for sample in sample_transactions:
    if st.sidebar.button(sample[:50] + "..."):
        st.session_state.message = sample
        # This will refresh the page and populate the message text area
        st.experimental_rerun()

if 'message' in st.session_state:
    message = st.session_state.message
