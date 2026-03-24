# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, FAISS, and FLAN-T5.

## 📁 Project Structure
rag-chatbot/
├── data/              # Put your PDF files here
├── faiss_index/       # Auto-generated after running vector_store.py
├── vector_store.py    # Ingests PDF and builds FAISS index
├── chatbot.py         # Run this to chat
├── requirements.txt   # Dependencies
└── README.md

## 🚀 Setup & Run

### 1. Clone the repo
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your PDF
Place your PDF inside the `data/` folder and rename it `sample.pdf`

### 5. Build the index
python vector_store.py

### 6. Run the chatbot
python chatbot.py

## 🛠️ Tech Stack
- Python 3.14
- LangChain
- FAISS (Vector Store)
- Hugging Face Transformers (FLAN-T5-Large)
- Sentence Transformers (all-MiniLM-L6-v2)