import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page Config
st.set_page_config(
    page_title="Saurabh AI Representative",
    page_icon="🤖"
)

st.title("🤖 Saurabh AI Representative")
with st.sidebar:
    st.header("About Saurabh")

    st.write("""
    M.Tech Candidate with interests in:

    - Machine Learning
    - Computer Vision
    - Generative AI
    - Data Analytics
    - Software Development
    """)

    st.header("Projects")

    st.write("""
    🚦 Smart Traffic Monitoring System

    🖼️ Image Search Engine

    📊 MLFlow Experiment Tracking
    """)

    with open("data/resume.pdf", "rb") as pdf_file:
        st.download_button(
            label="📄 Download Resume",
            data=pdf_file,
            file_name="Saurabh_Resume.pdf",
            mime="application/pdf"
        )

st.write(
    "Ask questions about Saurabh's experience, projects, skills, and background."
)

# Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@st.cache_resource
def load_retriever():

    files = [
        "data/about_saurabh.md",
        "data/traffic.md",
        "data/image_search.md",
        "data/mlflow.md"
    ]

    docs = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append(
            Document(
                page_content=text,
                metadata={"source": file}
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
user_query = st.chat_input("Ask anything about Saurabh...")

st.write("### 🎤 Voice Input")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True
)

if user_query:

    # Show User Message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    # Retrieve Relevant Documents
    docs = retriever.invoke(user_query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Generate Response
    answer = f"""
Based on Saurabh's profile and project knowledge base:

{context}

---
Source: Resume / Project Documents
"""

    # Show Assistant Message
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )