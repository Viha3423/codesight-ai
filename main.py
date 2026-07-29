import os
import shutil
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from google import genai

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")

client = genai.Client(api_key=api_key)

DB_DIR = "./chroma_db"
UPLOAD_DIR = "./user_uploaded_code"

os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

app = FastAPI(title="CodeSight AI API", version="0.3.2")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

class SourceSnippet(BaseModel):
    file_source: str
    code_content: str

class RAGResponse(BaseModel):
    status: str
    question: str
    retrieved_sources: List[SourceSnippet]
    ai_answer: str

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "CodeSight AI RAG Backend"}

@app.post("/upload-codebase")
async def upload_codebase(files: List[UploadFile] = File(...)):
    """Accepts multiple user code files, clears existing DB collections, and re-indexes."""
    try:
        # 1. Clear upload folder safely
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        # 2. Save newly uploaded files
        saved_file_paths = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_file_paths.append(file_path)

        # 3. Load & Chunk Files (autodetect_encoding handles .js, .json, .py cleanly)
        loader = DirectoryLoader(
            UPLOAD_DIR, 
            glob="**/*", 
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        docs = loader.load()

        if not docs:
            raise HTTPException(status_code=400, detail="No readable text/code files found in upload.")

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=300, 
            chunk_overlap=40
        )
        chunks = python_splitter.split_documents(docs)

        # 4. Safely overwrite ChromaDB collection
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        try:
            existing_ids = vector_db.get()['ids']
            if existing_ids:
                vector_db.delete(ids=existing_ids)
        except Exception:
            pass

        vector_db.add_documents(chunks)

        return {
            "status": "success", 
            "message": f"Successfully uploaded {len(saved_file_paths)} file(s)",
            "processed_files": [f.filename for f in files]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-codebase", response_model=RAGResponse)
def query_codebase(request: QueryRequest):
    try:
        if not os.path.exists(DB_DIR):
            raise HTTPException(status_code=400, detail="No code vector database found. Please upload code files first.")

        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        search_results = vector_db.similarity_search(request.question, k=request.top_k)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant code snippets found in database.")

        sources = []
        context_blocks = []

        for idx, doc in enumerate(search_results, 1):
            file_path = doc.metadata.get("source", "Unknown File")
            snippet_text = doc.page_content
            
            sources.append(SourceSnippet(file_source=file_path, code_content=snippet_text))
            context_blocks.append(f"--- Snippet #{idx} (File: {file_path}) ---\n{snippet_text}\n")

        context_str = "\n".join(context_blocks)

        rag_prompt = f"""
You are an expert AI Software Engineer assisting with codebase exploration.
Answer the user's question accurately using ONLY the code snippets provided in the context below.

=== RETRIEVED CODE CONTEXT ===
{context_str}

=== USER QUESTION ===
{request.question}
"""

        # Model Fallback Mechanism to prevent 429 quota locks
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
            except Exception as model_err:
                print(f"Attempt with {model_name} failed: {model_err}")
                continue

        if not response or not response.text:
            raise HTTPException(status_code=429, detail="API rate limit reached across available models. Please wait a minute and retry.")

        return RAGResponse(
            status="success",
            question=request.question,
            retrieved_sources=sources,
            ai_answer=response.text.strip()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))