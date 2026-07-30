# CodeSight AI — Codebase Intelligence Engine

A full-stack Retrieval-Augmented Generation (RAG) platform that enables natural-language Q&A over source code files. **CodeSight AI** extracts precise source logic and prevents LLM hallucinations by dynamically embedding, indexing, and retrieving localized code context.

---

## Problem Solved

Pasting raw code files directly into standard LLMs leads to strict token limits, high costs, and missing subtle logic in long files (*"Lost in the Middle"* effect). 

**CodeSight AI** solves this by:
* Chunking code into semantically rich blocks ($1000$ characters, $200$ overlap).
* Storing embeddings in **ChromaDB** using `all-MiniLM-L6-v2`.
* Retrieving only the top-$4$ relevant snippets for the query.
* Feeding precise context to **Google Gemini API** to generate accurate, zero-hallucination explanations.

---

## Tech Stack

* **Frontend & Deployment:** Streamlit Community Cloud
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **LLM Engine:** Google Gemini API (`google-genai`)
* **Orchestration:** LangChain

---

## Key Features

* **Grounded Responses:** Strict prompt constraints enforce answers based *only* on retrieved code context.
* **Code-Aware Chunking:** Keeps full class definitions and function bodies intact during vector indexing.
* **Source Attribution:** Displays exact code snippet blocks used to answer each query.
* **Auto-Purge Storage:** Automatically resets local storage on new file uploads.

