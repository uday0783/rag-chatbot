from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load FAISS
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Load FLAN-T5 large
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

def generate_answer(context, question):
    prompt = f"""You are a detailed and helpful AI assistant.
Using the context below, give a thorough and complete answer to the question.
List all relevant points. Do not stop after one sentence.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Detailed Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            min_new_tokens=80,        # ✅ forces at least 80 tokens output
            max_new_tokens=300,       # ✅ allows up to 300 tokens
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,       # ✅ higher = longer answers
            repetition_penalty=1.2    # ✅ avoids repeating same words
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("🤖 PDF Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()

    if not query:
        continue

    if query.lower() == "exit":
        print("👋 Exiting chatbot.")
        break

    # ✅ k=6 for more context coverage
    docs = db.similarity_search(query, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])

    answer = generate_answer(context, query)
    print(f"\nBot: {answer}\n")
    print("-" * 60)