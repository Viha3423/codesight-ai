import os
import shutil
import streamlit as st
from google import genai

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Page Setup
st.set_page_config(
    page_title="CodeSight AI — Codebase Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("CodeSight AI — Codebase Search")

# Retrieve API Key from Streamlit Secrets or Environment
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY missing! Please add it to Streamlit Secrets or your .env file.")
    st.stop()

# Initialize Gemini Client & Embeddings
client = genai.Client(api_key=api_key)
DB_DIR = "./chroma_db"
UPLOAD_DIR = "./user_uploaded_code"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- SECTION 1: CODEBASE FILE UPLOAD ---
st.header("1. Upload Your Code File")
uploaded_file = st.file_uploader(
    "Upload a source code file (.py, .js, .txt, .json, etc.)", 
    accept_multiple_files=False  # Restricted to single-file upload
)

if st.button("Upload & Index File", type="secondary"):
    if not uploaded_file:
        st.warning("Please select a file to upload.")
    else:
        with st.spinner("Indexing code file into vector store..."):
            try:
                # 1. Clean upload directory
                if os.path.exists(UPLOAD_DIR):
                    for filename in os.listdir(UPLOAD_DIR):
                        file_path = os.path.join(UPLOAD_DIR, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            pass

                # 2. Save single uploaded file
                saved_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(saved_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # 3. Load & Chunk File
                loader = DirectoryLoader(
                    UPLOAD_DIR, 
                    glob="**/*", 
                    loader_cls=TextLoader,
                    loader_kwargs={"autodetect_encoding": True}
                )
                docs = loader.load()

                if not docs:
                    st.error("Could not read text/code from uploaded file.")
                else:
                    python_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.PYTHON, 
                        chunk_size=300, 
                        chunk_overlap=40
                    )
                    chunks = python_splitter.split_documents(docs)

                    # 4. Overwrite ChromaDB records
                    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                    try:
                        existing_ids = vector_db.get()['ids']
                        if existing_ids:
                            vector_db.delete(ids=existing_ids)
                    except Exception:
                        pass

                    vector_db.add_documents(chunks)
                    st.success(f"Successfully indexed `{uploaded_file.name}` into ChromaDB!")

            except Exception as e:
                st.error(f"Error during indexing: {e}")

# --- SECTION 2: QUERY CODEBASE ---
st.header("2. Ask Questions About Your Code")
user_query = st.text_area(
    "Query uploaded codebase:",
    placeholder="e.g., How does user authentication work in this uploaded codebase?",
    height=80
)

if st.button("Search Codebase", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif not os.path.exists(DB_DIR):
        st.warning("Please upload and index code files first.")
    else:
        with st.spinner("Searching uploaded context..."):
            try:
                vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                search_results = vector_db.similarity_search(user_query, k=2)

                if not search_results:
                    st.error("No relevant code snippets found in vector store.")
                else:
                    context_blocks = []
                    for idx, doc in enumerate(search_results, 1):
                        file_path = doc.metadata.get("source", "Unknown File")
                        snippet_text = doc.page_content
                        context_blocks.append(f"--- Snippet #{idx} (File: {file_path}) ---\n{snippet_text}\n")

                    context_str = "\n".join(context_blocks)

                    rag_prompt = f"""
You are an expert AI Software Engineer assisting with codebase exploration.
Answer the user's question accurately using ONLY the code snippets provided in the context below.

=== RETRIEVED CODE CONTEXT ===
{context_str}

=== USER QUESTION ===
{user_query}
"""

                    # Fallback loop across models
                    fallback_models = ["gemini-flash-latest", "gemini-2.0-flash-lite"]
                    response = None

                    for model_name in fallback_models:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=rag_prompt
                            )
                            if response and response.text:
                                break
                        except Exception:
                            continue

                    if response and response.text:
                        st.subheader("Answer")
                        st.write(response.text.strip())

                        st.divider()

                        st.subheader("Source Code Context")
                        for idx, doc in enumerate(search_results, 1):
                            st.write(f"**Snippet #{idx} — `{doc.metadata.get('source', 'Unknown')}`**")
                            st.code(doc.page_content, language="python")
                    else:
                        st.error("Rate limit reached. Please wait a minute and try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")