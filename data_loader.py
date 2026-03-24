from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Load Text
def load_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Combine all documents
def load_all_documents():
    all_docs = []

    text_docs = load_text("data/sample.txt")

    all_docs.extend(text_docs)

    return all_docs

# Test loader
if __name__ == "__main__":
    docs = load_all_documents()

    print("Total Documents:", len(docs))
    print(docs[0].page_content)