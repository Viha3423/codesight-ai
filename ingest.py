import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./chroma_db"
SAMPLE_CODE_DIR = "./sample_code"

def run_ingestion():
    print("📁 Loading code files from directory...")
    # Load all python files from sample_code folder
    loader = DirectoryLoader(
        SAMPLE_CODE_DIR, 
        glob="**/*.py", 
        loader_cls=TextLoader
    )
    docs = loader.load()
    print(f"Found {len(docs)} files.")

    print("✂️ Chunking code files using Python syntax splitter...")
    # Split code structurally using language syntax aware rules
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=300, 
        chunk_overlap=40
    )
    chunks = python_splitter.split_documents(docs)
    print(f"Created {len(chunks)} code chunks.")

    print("🧠 Initializing free HuggingFace Embedding Model (all-MiniLM-L6-v2)...")
    # Open-source embedding model running locally on your CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Storing vectors into local ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"✅ Ingestion complete! Vector database saved at '{DB_DIR}'")

if __name__ == "__main__":
    run_ingestion()