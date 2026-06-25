import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1400px;
}

.header-box {
    background: linear-gradient(
        90deg,
        #1f2937,
        #374151
    );
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.header-title {
    color: white;
    font-size: 34px;
    font-weight: bold;
}

.header-sub {
    color: #d1d5db;
}

.source-box {
    border-left: 5px solid #4CAF50;
    padding: 10px;
    margin-bottom: 10px;
    background-color: #f8f9fa;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="header-box">
    <div class="header-title">
        📚 Smart PDF RAG Assistant
    </div>
    <div class="header-sub">
        Upload a PDF and chat with it using Mistral AI + ChromaDB
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --------------------------------------------------
# MODELS
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatMistralAI(
        model="mistral-small-2506"
    )

embedding_model = load_embedding_model()
llm = load_llm()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Upload Book")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    st.divider()

    search_type = st.selectbox(
        "Retrieval Type",
        ["mmr", "similarity"]
    )

    k_value = st.slider(
        "Top K Results",
        1,
        10,
        4
    )

    process_btn = st.button(
        "🚀 Process PDF",
        use_container_width=True
    )

# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file and process_btn:

    with st.spinner("Processing PDF..."):

        temp_dir = tempfile.mkdtemp()

        pdf_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        if os.path.exists("chroma-db"):
            shutil.rmtree("chroma-db")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma-db"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_name = uploaded_file.name

        st.success(
            f"PDF Processed Successfully! ({len(chunks)} chunks)"
        )

# --------------------------------------------------
# PDF DETAILS
# --------------------------------------------------

if st.session_state.pdf_name:

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"📄 Current Book: {st.session_state.pdf_name}"
        )

    with col2:
        st.success(
            "✅ Vector Database Ready"
        )

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If answer not found say:

'I could not find the answer in the document.'
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for role, content in st.session_state.messages:

    with st.chat_message(role):
        st.markdown(content)

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask anything about the uploaded PDF..."
)

if question:

    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        st.session_state.messages.append(
            ("user", question)
        )

        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k_value,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.spinner("Searching document..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            response = llm.invoke(final_prompt)

            answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("📖 Retrieved Sources"):

                for i, doc in enumerate(docs):

                    page = doc.metadata.get(
                        "page",
                        "Unknown"
                    )

                    st.markdown(
                        f"""
<div class="source-box">
<b>Chunk {i+1}</b><br>
Page: {page}<br><br>
{doc.page_content[:400]}...
</div>
""",
                        unsafe_allow_html=True
                    )

        st.session_state.messages.append(
            ("assistant", answer)
        )