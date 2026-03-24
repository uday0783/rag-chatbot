from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import load_all_documents

def split_documents():
    docs = load_all_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":
    chunks = split_documents()

    print("Total Chunks:", len(chunks))
    print(chunks[0].page_content)