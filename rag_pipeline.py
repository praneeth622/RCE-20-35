import os
import glob
import json
import argparse
import pickle
import re
from pathlib import Path
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai

INDEX_DIR = "./index"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 220
OVERLAP = 40
WEIGHT_DENSE = 0.6
WEIGHT_SPARSE = 0.4
TOP_K = 6
ABSTAIN_THRESHOLD = 0.18

def chunk_text(text, doc_id, chunk_size=220, overlap=40):
    words = text.split()
    chunks = []
    if not words:
        return chunks
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({"doc_id": doc_id, "text": chunk_text})
        i += chunk_size - overlap
    return chunks

def min_max_normalize(scores):
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val == 0:
        return np.ones_like(scores) if max_val > 0 else np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def cmd_build(docs_dir):
    os.makedirs(INDEX_DIR, exist_ok=True)
    all_chunks = []
    
    docs_path = Path(docs_dir)
    for ext in ["*.txt", "*.md"]:
        for file_path in docs_path.rglob(ext):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    doc_id = file_path.name
                    all_chunks.extend(chunk_text(content, doc_id, CHUNK_SIZE, OVERLAP))
            except Exception as e:
                print(f"Failed reading {file_path}: {e}")
    
    if not all_chunks:
        print("No documents found or all empty.")
        return

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "faiss.index"))
    
    print("Building BM25 index...")
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
        
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
        
    print(f"Successfully built index for {len(all_chunks)} chunks in {INDEX_DIR}.")

def retrieve(query):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        all_chunks = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
        
    k_search = len(all_chunks)
    dense_scores_sub, I = faiss_index.search(q_emb, k_search)
    dense_scores = np.zeros(len(all_chunks))
    for i, idx in enumerate(I[0]):
        dense_scores[idx] = dense_scores_sub[0][i]
        
    tokenized_query = query.lower().split()
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    norm_dense = min_max_normalize(dense_scores)
    norm_sparse = min_max_normalize(sparse_scores)
    
    fused_scores = WEIGHT_DENSE * norm_dense + WEIGHT_SPARSE * norm_sparse
    
    top_indices = np.argsort(fused_scores)[::-1][:TOP_K]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": all_chunks[idx],
            "score": fused_scores[idx]
        })
        
    return results

def process_ask(question):
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
    try:
        results = retrieve(question)
    except Exception as e:
        raise ValueError(f"Failed to retrieve documents (has the index been built?): {e}")
    
    if not results:
        return {
            "answer": "I cannot answer this from the provided documents.",
            "citations": [],
            "abstained_flag": True,
            "retrieved_doc_ids": [],
            "fused_scores": []
        }
        
    best_score = results[0]["score"]
    retrieved_doc_ids = list(set([r["chunk"]["doc_id"] for r in results]))
    fused_scores = [float(r["score"]) for r in results]
    
    if best_score < ABSTAIN_THRESHOLD:
        return {
            "answer": "I cannot answer this from the provided documents.",
            "citations": [],
            "abstained_flag": True,
            "retrieved_doc_ids": retrieved_doc_ids,
            "fused_scores": fused_scores
        }
        
    context = "\n\n".join([f"Document: {r['chunk']['doc_id']}\nText: {r['chunk']['text']}" for r in results])
    
    system_prompt = (
        "You are an assistant. Answer ONLY from the retrieved passages.\n"
        "Cite every claim as [SOURCE: filename].\n"
        "If the passages don't contain the answer, output exactly 'I cannot answer this from the provided documents.' instead of guessing."
    )
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt
    )
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    answer = response.text
    
    abstained = "I cannot answer this from the provided documents." in answer
        
    citations_found = list(set(re.findall(r'\[SOURCE: (.*?)\]', answer)))
    valid_citations = [c for c in citations_found if c in retrieved_doc_ids]
            
    return {
        "answer": answer,
        "citations": valid_citations,
        "abstained_flag": abstained,
        "retrieved_doc_ids": retrieved_doc_ids,
        "fused_scores": fused_scores
    }

def cmd_ask(question):
    try:
        res = process_ask(question)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")

def cmd_batch(questions_file, out_file):
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    results = []
    for q in questions:
        print(f"Processing question: {q}")
        try:
            res = process_ask(q)
        except Exception as e:
            res = {"error": str(e)}
        results.append({"question": q, **res})
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vanilla RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    build_parser = subparsers.add_parser("build", help="Build the FAISS and BM25 indexes")
    build_parser.add_argument("--docs_dir", required=True, help="Directory containing .txt and .md files")
    
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("--q", required=True, help="The question to ask")
    
    batch_parser = subparsers.add_parser("batch", help="Batch process a JSON array of questions")
    batch_parser.add_argument("--questions", required=True, help="JSON file containing list of query strings")
    batch_parser.add_argument("--out", required=True, help="Path for JSON output")
    
    args = parser.parse_args()
    
    if args.command == "build":
        cmd_build(args.docs_dir)
    elif args.command == "ask":
        cmd_ask(args.q)
    elif args.command == "batch":
        cmd_batch(args.questions, args.out)
