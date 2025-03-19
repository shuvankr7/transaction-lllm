import os
import streamlit as st
from langchain_groq import ChatGroq

# Set environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit Page Config
st.set_page_config(page_title="Transaction Processor", page_icon="📊", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        body {font-family: 'Arial', sans-serif; background-color: #f5f7fa;}
        .stApp {background-color: #ffffff; padding: 30px;}
        .main {background-color: #ffffff;}
        .stTextInput, .stTextArea, .stNumberInput, .stSelectbox, .stButton {
            border-radius: 8px; border: 1px solid #d1d5db;
        }
        .stButton>button {background-color: #007BFF; color: white; border-radius: 8px; font-weight: bold;}
        .stButton>button:hover {background-color: #0056b3;}
        .stSidebar {background-color: #2c3e50; color: white;}
        .stSidebar .sidebar-content {padding: 20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Transaction Message Processor")
st.write("Extract structured transaction details from bank messages.")

# Sidebar Styling
st.sidebar.header("Sample Transactions")
st.sidebar.write("Use these for testing:")

sample_transactions = [
    "Rs.105.00 spent on your SBI Credit Card ending with 5775 at Auto Fuel Station on 18-03-25 via UPI (Ref No. 507775912830).",
    "Rs.2,500.00 debited from your HDFC Bank Account ending with 6789 on 19-03-25 for Amazon purchase (Ref No. 987654321).",
    "Alert: Rs.1,450.00 withdrawn from your ICICI ATM Card ending with 1234 at ICICI ATM on 17-03-25 (Ref No. 123456789)."
]

for sample in sample_transactions:
    if st.sidebar.button(sample[:50] + "..."):
        st.session_state.message = sample
        st.experimental_rerun()

# API Key Input
api_key = st.text_input("GROQ API Key", type="password")
model = st.selectbox("Select GROQ Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0)

# Temperature and Max Tokens
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
with col2:
    max_tokens = st.number_input("Max Tokens", min_value=10, max_value=4096, value=1024, step=10)

# Initialize LLM
if st.button("Initialize System"):
    if not api_key:
        st.error("Please enter a valid API Key.")
    else:
        llm = ChatGroq(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        st.session_state.llm = llm
        st.success("System Initialized Successfully!")

# Message Input
message = st.text_area("Enter Transaction Message")

if st.button("Process Message") and message:
    if 'llm' not in st.session_state:
        st.warning("Please initialize the system first.")
    else:
        with st.spinner("Processing..."):
            response = st.session_state.llm.invoke(message)
            st.subheader("Extracted Transaction Details")
            st.code(response.content, language="json")
