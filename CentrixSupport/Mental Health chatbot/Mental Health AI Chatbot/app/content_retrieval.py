import os
import hashlib
import logging
import mimetypes
import pdfplumber
import pytesseract
import cv2
import tempfile
import re
import nltk
import json
import csv
import zipfile
import whisper
import docx
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

nltk.download("punkt")

# Config
CHROMA_DIR = "chroma_index"
HASH_CACHE_FILE = "text_hash.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma2:2b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== TEXT EXTRACTION ==========

def extract_text_from_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text(x_tolerance=2, y_tolerance=2) or "" for p in pdf.pages)
    except Exception as e:
        logging.error(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_image(path):
    try:
        image = cv2.imread(path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"Image OCR failed: {e}")
        return ""

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logging.error(f"DOCX extraction failed: {e}")
        return ""

def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"TXT extraction failed: {e}")
        return ""

def extract_text_from_csv(path):
    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            return "\n".join([" | ".join(row) for row in csv.reader(csvfile)])
    except Exception as e:
        logging.error(f"CSV extraction failed: {e}")
        return ""

def extract_text_from_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        logging.error(f"JSON extraction failed: {e}")
        return ""

def extract_text_from_audio(path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(path)
        return result["text"]
    except Exception as e:
        logging.error(f"Audio transcription failed: {e}")
        return ""

def extract_text_from_zip(path):
    text = ""
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                for file in os.listdir(temp_dir):
                    full_path = os.path.join(temp_dir, file)
                    text += extract_text(full_path) + "\n"
    except Exception as e:
        logging.error(f"ZIP extraction failed: {e}")
    return text

# Dispatcher
def extract_text(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".csv":
        return extract_text_from_csv(file_path)
    elif ext == ".json":
        return extract_text_from_json(file_path)
    elif ext in [".mp3", ".wav", ".mp4"]:
        return extract_text_from_audio(file_path)
    elif ext == ".zip":
        return extract_text_from_zip(file_path)
    elif ext == ".exe":
        logging.warning("EXE files are not supported for text extraction.")
        return ""
    else:
        logging.warning("Unsupported file type.")
        return ""

# ========== UTILS ==========

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.replace("•", "-").replace("–", "-").strip()

def semantic_chunk_text(text, chunk_size=500, overlap=100):
    sentences = sent_tokenize(text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-(overlap // len(sent)):] if sent else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))
    return chunks

def compute_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def has_file_changed(text):
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, "r") as f:
            return compute_hash(text) != f.read().strip()
    return True

def cache_hash(text):
    with open(HASH_CACHE_FILE, "w") as f:
        f.write(compute_hash(text))

def embed_text_and_store(text, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(CHROMA_DIR):
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        vs.add_documents(docs)
    else:
        vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DIR)

    cache_hash(text)
    return vs

def answer_query(query, vectorstore):
    try:
        results = vectorstore.similarity_search_with_score(query, k=5)
        relevant_docs = [doc for doc, score in results if score <= 1.4]

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = (
            "You are an AI assistant. You must rephase the answer ONLY using the provided context. "
            "If the answer is not found in the context, you must respond with: 'Not available in the documentation.' "
            "Do NOT make assumptions. Do NOT generate answers from general knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        ollama = OllamaLLM(model=OLLAMA_MODEL)
        answer = ollama.invoke(prompt)

        source = "📚 Documents" if context.strip() else "💡 No relevant document match"
        print(f"\n{source}\n✅ Answer: {answer.strip()}")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")

# ========== FUNCTION FOR CHATBOT RAG ==========

def get_rag_context(file_paths, query):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_texts = []
    for file_path in file_paths:
        raw_text = extract_text(file_path)
        cleaned = clean_text(raw_text)
        if cleaned:
            all_texts.append(cleaned)

    if not all_texts:
        return ""

    combined_text = "\n".join(all_texts)
    chunks = semantic_chunk_text(combined_text)
    full_text = "\n".join(chunks)

    if has_file_changed(full_text):
        file_names = ", ".join([os.path.basename(p) for p in file_paths])
        vectorstore = embed_text_and_store(full_text, file_names)
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

    try:
        results = vectorstore.similarity_search_with_score(query, k=5)
        relevant_docs = [doc for doc, score in results if score <= 1.4]
        return "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
    except Exception as e:
        logging.error(f"RAG query error: {e}")
        return ""

# ========== OPTIONAL STANDALONE USAGE ==========

def main():
    file_path = input("📁 Enter the path to your file: ").strip()

    if not os.path.exists(file_path):
        print("❌ File does not exist.")
        return

    raw_text = extract_text(file_path)
    cleaned = clean_text(raw_text)

    if not cleaned:
        return

    chunks = semantic_chunk_text(cleaned)
    full_text = "\n".join(chunks)
    file_name = os.path.basename(file_path)

    if has_file_changed(full_text):
        vectorstore = embed_text_and_store(full_text, file_name)
    else:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    print("\n💬 Ask your questions (type 'exit' to quit):")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ['exit', 'quit']:
            break
        answer_query(q, vectorstore)

if __name__ == "__main__":
    main()