import streamlit as st
import openai
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import tempfile
import os
import requests

# Page configuration
st.set_page_config(
    page_title="Procurement Opportunity Filter",
    page_icon="LaTronics Solutions",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        margin-right: auto;
        border-left: 4px solid #9c27b0;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metrics-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None

# Databricks job trigger function
def trigger_databricks_job():
    instance = os.getenv("DATABRICKS_INSTANCE")
    token = os.getenv("DATABRICKS_TOKEN")
    job_id = os.getenv("DATABRICKS_JOB_ID")

    if not all([instance, token, job_id]):
        st.error("❌ Databricks credentials not set in environment variables.")
        return

    response = requests.post(
        f"{instance}/api/2.1/jobs/run-now",
        headers={"Authorization": f"Bearer {token}"},
        json={"job_id": job_id}
    )

    if response.status_code == 200:
        run_id = response.json().get("run_id")
        st.success(f"✅ Triggered Databricks Job! Run ID: {run_id}")
    else:
        st.error(f"❌ Failed to trigger job: {response.text}")

# Process uploaded document and return QA chain
def process_document(uploaded_file, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        csv_reader = CSVLoader(tmp_file_path)
        documents = csv_reader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(api_key=api_key)
        db = FAISS.from_documents(documents=chunks, embedding=embeddings)

        llm = OpenAI(openai_api_key=api_key, temperature=0.1)

        condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}
            Follow up Input: {question}
            Standalone question:"""
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=False
        )

        return qa_chain

    finally:
        os.unlink(tmp_file_path)

# Main UI logic
def main():
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Procurement Opportunity Filter</h1>
        <p style="font-size: 1.2em; margin-top: 0.5rem;">
            RAG-Powered Business Development System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        api_key = st.text_input("🔑 OpenAI API Key", type="password")
        st.markdown("### ⚙️ Run RAG Agent")
        if st.button("🚀 Run Databricks Job"):
            trigger_databricks_job()

        st.markdown("---")
        st.markdown("### About This Tool")
        st.info("Upload procurement forecasts and ask AI questions to identify strategic contract opportunities.")

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Document Upload and Chat UI
    col1, col2 = st.columns([2, 1])

    with col1:
        if not st.session_state.document_processed:
            st.subheader("📁 Upload Procurement Forecasts")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file and api_key:
                if st.button("Process Document"):
                    with st.spinner("Processing..."):
                        try:
                            qa_chain = process_document(uploaded_file, api_key)
                            st.session_state.qa_chain = qa_chain
                            st.session_state.document_processed = True
                            st.session_state.document_name = uploaded_file.name
                            st.success("✅ Document ready!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

        else:
            st.subheader(f"💬 Chat with: {st.session_state.document_name}")
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Response:</strong> {message["content"]}</div>', unsafe_allow_html=True)

            user_question = st.chat_input("Ask a question...")
            if user_question and st.session_state.qa_chain:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.qa_chain({
                            "question": user_question,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = result["answer"]
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.chat_history.append((user_question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    with col2:
        if st.session_state.document_processed:
            st.markdown("""<div class="feature-card"><h4>💡 Sample Questions</h4></div>""", unsafe_allow_html=True)
            for q in [
                "What contract opportunities are related to cloud services or AI?",
                "Which listings involve Program or Project Management?",
                "What NAICS codes are most common in this dataset?",
            ]:
                if st.button(f"❓ {q}", key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Analyzing..."):
                        try:
                            result = st.session_state.qa_chain({
                                "question": q,
                                "chat_history": st.session_state.chat_history
                            })
                            answer = result["answer"]
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.session_state.chat_history.append((q, answer))
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            st.markdown("""<div class="feature-card"><h4>📄 Document Info</h4></div>""", unsafe_allow_html=True)
            st.info(f"**Document:** {st.session_state.document_name}")
            st.info(f"**Status:** ✅ Ready")
            st.info(f"**Chat Messages:** {len(st.session_state.messages)}")

            if st.button("🔄 Process New Document"):
                st.session_state.document_processed = False
                st.session_state.qa_chain = None
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.document_name = None
                st.rerun()

if __name__ == "__main__":
    main()
