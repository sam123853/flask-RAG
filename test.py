from flask import Flask, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
app = Flask(__name__)
 
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vector_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/")
def home():
    return '''
    <h2>Flask RAG App</h2>
    <form action="/upload_file" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Upload File</button>
    </form>
    <br>
    <form action="/query" method="post">
        <input type="text" name="query" placeholder="Enter your question">
        <button type="submit">Ask</button>
    </form>
    '''
 
 
# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
def load_vector_store():
    """Load existing FAISS vector store if present."""
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None
 
def save_vector_store(db):
    """Persist FAISS vector store."""
    db.save_local(VECTOR_DB_PATH)
 
# ---------- 1. FILE UPLOAD ENDPOINT ----------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
 
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
 
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
 
    # Detect file type
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(filepath)
    elif filepath.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
 
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
 
    # Load or create vector store
    db = load_vector_store()
    if db is None:
        db = FAISS.from_documents(split_docs, embeddings)
    else:
        db.add_documents(split_docs)
 
    save_vector_store(db)
 
    return jsonify({"message": f"File '{file.filename}' uploaded and processed successfully."})
 
# ---------- 2. QUERY ENDPOINT ----------
@app.route("/query", methods=["POST"])
def query():
    # ✅ Accept both JSON and form submissions
    if request.is_json:
        data = request.get_json()
        query_text = data.get("query", "")
    else:
        query_text = request.form.get("query", "")
 
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
 
    # ✅ Check vector store
    if not os.path.exists(VECTOR_DB_PATH):
        return jsonify({"error": "No vector store found. Please upload a file first."}), 400
 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
 
    docs = retriever.invoke(query_text)
    context = " ".join([d.page_content for d in docs])
 
    # ✅ Use FLAN-T5 for response generation
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query_text}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    return jsonify({"query": query_text, "answer": answer})
# ---------- RUN APP ----------
if __name__ == "__main__":
    app.run(debug=True)
 