# 📘 NotebookLM-style AI Assistant (RAG-based)

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask grounded questions with context-aware answers.

---

## 🚀 Features

- 📂 Upload PDF/TXT documents
- 🧠 Semantic search using embeddings
- 🔍 FAISS-based vector retrieval
- 🤖 LLM-powered answer generation (Groq - Llama 3.1)
- 📌 Context-grounded answers (reduces hallucination)
- 📝 Document summarization
- 💡 Auto-generated questions
- 💬 Conversation history

---

## 🧠 Tech Stack

- Python
- Streamlit (UI)
- Sentence Transformers (Embeddings)
- FAISS (Vector DB)
- Groq API (LLM - Llama 3.1)
- PyPDF / Text Processing

---

## ⚙️ Architecture (RAG Pipeline)


Upload → Loader → Preprocessing → Chunking
→ Embedding → FAISS Storage

Query → Rewrite → Retrieve → Generate → Answer





---

## 🧠 Models Used

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`

### LLM Model
- `llama-3.1-8b-instant` (Groq)

---

## 📁 Project Structure


Upload → Loader → Preprocessing → Chunking
→ Embedding → FAISS Storage

Query → Rewrite → Retrieve → Generate → Answer





├── ingestion/ # Loading, cleaning, chunking
├── embeddings/ # Embedding generation
├── vectorstore/ # FAISS storage
├── rag/ # Retriever + Generator
├── utils/ # Helpers + Logger
├── data/uploads/ # Uploaded files
├── app.py # Main Streamlit app
├── config.py # Configurations
├── requirements.txt



---

## ⚙️ Setup Instructions

### 1. Clone Repo

git clone https://github.com/mohitsk26/NOTEBOOKLMPROJECT.git

cd NOTEBOOKLMPROJECT


---

### 2. Create Virtual Environment

python -m venv myvenv
.\myvenv\Scripts\Activate.ps1


---

### 3. Install Dependencies

pip install -r requirements.txt


---

### 4. Add API Key

Create `.env` file:


GROQ_API_KEY=your_api_key_here


---

### 5. Run App

streamlit run app.py



---

## 🧠 Key Concepts Implemented

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases (FAISS)
- Prompt Engineering
- Query Rewriting
- Chunking Strategy
- Context-Aware LLM Responses

---

## ⚠️ Challenges & Fixes

- Fixed API key loading using `.env`
- Resolved FAISS persistence issue
- Fixed missing method errors
- Updated deprecated LLM model
- Handled dependency issues (torchvision)

---

## Future Improvements

- Hybrid search (BM25 + embeddings)
- Re-ranking models
- FastAPI backend
- Multi-document querying

---

## 

Mohit Singh Kashyap
