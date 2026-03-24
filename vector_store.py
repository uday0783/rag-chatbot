from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ fixed import
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

PDF_PATH = "data/sample.pdf"  # ✅ your filename

if not os.path.exists(PDF_PATH):
    print(f"❌ ERROR: '{PDF_PATH}' not found!")
    print(f"   Files in data/ folder: {os.listdir('data') if os.path.exists('data') else 'data/ folder missing!'}")
    exit()

# 1. Load PDF
print(f"📄 Loading: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)
print(f"✅ Created {len(docs)} chunks")

# Verify content
print("\n--- Sample chunk ---")
print(docs[0].page_content[:300] if docs else "NO CONTENT EXTRACTED!")
print("--------------------\n")

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ PDF processed and stored in FAISS")