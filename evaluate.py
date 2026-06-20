import argparse
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import google.generativeai as genai
import rag_pipeline

ABSTAIN_TEXT = "I cannot answer this from the provided documents."


def load_env_file(env_path):
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_text(text):
    text = text or ""
    text = re.sub(r"\[SOURCE:\s*.*?\]", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def token_f1(prediction, gold):
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def dedupe_doc_ids(results):
    seen = set()
    ranked = []
    for item in results:
        doc_id = item["chunk"]["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def reciprocal_rank(ranked_doc_ids, gold_doc_ids):
    gold_set = set(gold_doc_ids)
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranked_doc_ids, gold_doc_ids, k):
    if not gold_doc_ids:
        return None
    hits = len(set(ranked_doc_ids[:k]) & set(gold_doc_ids))
    return hits / len(set(gold_doc_ids))


def precision_at_k(ranked_doc_ids, gold_doc_ids, k):
    if k <= 0:
        return 0.0
    hits = len(set(ranked_doc_ids[:k]) & set(gold_doc_ids))
    return hits / k


def extract_citations(answer_text, citations_field):
    if citations_field:
        return list(dict.fromkeys(citations_field))
    matches = re.findall(r"\[SOURCE:\s*(.*?)\]", answer_text or "", flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


def load_sparse_index():
    with open(Path(rag_pipeline.INDEX_DIR) / "chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    with open(Path(rag_pipeline.INDEX_DIR) / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return all_chunks, bm25


def sparse_retrieve(query, all_chunks, bm25, top_k):
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    return [{"chunk": all_chunks[idx], "score": float(scores[idx])} for idx in top_indices]


def generate_with_gemini(question, retrieval_results):
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    retrieved_doc_ids = list(dict.fromkeys(item["chunk"]["doc_id"] for item in retrieval_results))
    fused_scores = [float(item["score"]) for item in retrieval_results]

    if not retrieval_results:
        return {
            "answer": ABSTAIN_TEXT,
            "citations": [],
            "abstained_flag": True,
            "retrieved_doc_ids": [],
            "fused_scores": [],
        }

    context = "\n\n".join(
        f"Document: {item['chunk']['doc_id']}\nText: {item['chunk']['text']}" for item in retrieval_results
    )

    system_prompt = (
        "You are an assistant. Answer ONLY from the retrieved passages.\n"
        "Cite every claim as [SOURCE: filename].\n"
        "If the passages don't contain the answer, output exactly "
        "'I cannot answer this from the provided documents.' instead of guessing."
    )

    model = genai.GenerativeModel(
        model_name=rag_pipeline.GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = model.generate_content(prompt)
    try:
        answer = response.text
    except ValueError:
        answer = ABSTAIN_TEXT

    citations_found = extract_citations(answer, [])
    valid_citations = [citation for citation in citations_found if citation in retrieved_doc_ids]

    return {
        "answer": answer,
        "citations": valid_citations,
        "abstained_flag": normalize_text(answer) == normalize_text(ABSTAIN_TEXT),
        "retrieved_doc_ids": retrieved_doc_ids,
        "fused_scores": fused_scores,
    }


def is_abstained(result):
    if result.get("error"):
        return True
    answer = (result.get("answer") or "").strip()
    if not answer:
        return True
    if result.get("abstained_flag"):
        return True
    if normalize_text(answer) == normalize_text(ABSTAIN_TEXT):
        return True
    if "quota" in answer.lower() and "failed" in answer.lower():
        return True
    return False


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def evaluate(gold_items, k, run_generation, retrieval_mode):
    detailed = []
    recall_scores = []
    precision_scores = []
    rr_scores = []
    f1_scores = []
    false_answer_count = 0
    false_abstain_count = 0
    citation_hallucination_count = 0
    answered_count = 0
    unanswerable_count = 0
    answerable_count = 0
    generation_error_count = 0
    all_chunks = None
    bm25 = None

    if retrieval_mode == "bm25":
        all_chunks, bm25 = load_sparse_index()

    for item in gold_items:
        question = item["question"]
        if retrieval_mode == "bm25":
            retrieval_results = sparse_retrieve(question, all_chunks, bm25, k)
        else:
            retrieval_results = rag_pipeline.retrieve(question)
        ranked_doc_ids = dedupe_doc_ids(retrieval_results)

        result = {
            "answer": "",
            "citations": [],
            "retrieved_doc_ids": ranked_doc_ids,
            "fused_scores": [float(r["score"]) for r in retrieval_results],
            "abstained_flag": True,
        }

        if run_generation:
            try:
                if retrieval_mode == "bm25":
                    result = generate_with_gemini(question, retrieval_results)
                else:
                    result = rag_pipeline.process_ask(question)
            except Exception as exc:
                result["error"] = str(exc)
                generation_error_count += 1

        gold_doc_ids = item["gold_doc_ids"]
        rec = recall_at_k(ranked_doc_ids, gold_doc_ids, k)
        prec = precision_at_k(ranked_doc_ids, gold_doc_ids, k)
        rr = reciprocal_rank(ranked_doc_ids, gold_doc_ids)

        recall_scores.append(rec)
        precision_scores.append(prec)
        rr_scores.append(rr)

        abstained = is_abstained(result)
        answer_text = result.get("answer", "")

        if item["answerable"]:
            answerable_count += 1
            f1_scores.append(token_f1(answer_text, item["gold_answer"]))
            if abstained:
                false_abstain_count += 1
        else:
            unanswerable_count += 1
            if not abstained:
                false_answer_count += 1

        citations = extract_citations(answer_text, result.get("citations"))
        answered = not abstained
        if answered:
            answered_count += 1
            if any(citation not in set(gold_doc_ids) for citation in citations):
                citation_hallucination_count += 1

        detailed.append(
            {
                "question": question,
                "gold_doc_ids": gold_doc_ids,
                "ranked_doc_ids": ranked_doc_ids,
                "answerable": item["answerable"],
                "gold_answer": item["gold_answer"],
                "prediction": result,
                "recall_at_k": rec,
                "precision_at_k": prec,
                "reciprocal_rank": rr,
                "token_f1": token_f1(answer_text, item["gold_answer"]) if item["answerable"] else None,
                "answered": answered,
                "citations_used": citations,
            }
        )

    summary = {
        "num_questions": len(gold_items),
        "k": k,
        "run_generation": run_generation,
        "retrieval_mode": retrieval_mode,
        "recall_at_k": mean(recall_scores),
        "precision_at_k": mean(precision_scores),
        "mrr": mean(rr_scores),
        "answer_token_f1": mean(f1_scores),
        "false_answer_rate": (false_answer_count / unanswerable_count) if unanswerable_count else 0.0,
        "false_abstain_rate": (false_abstain_count / answerable_count) if answerable_count else 0.0,
        "citation_hallucination_rate": (citation_hallucination_count / answered_count) if answered_count else 0.0,
        "answered_count": answered_count,
        "generation_error_count": generation_error_count,
    }

    return {"summary": summary, "details": detailed}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and QA performance against gold.json.")
    parser.add_argument("--gold", default="gold.json", help="Path to the gold dataset.")
    parser.add_argument("--k", type=int, default=6, help="k for Recall@k and Precision@k.")
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25", help="Retrieval mode to evaluate.")
    parser.add_argument("--skip-generation", action="store_true", help="Only run retrieval metrics.")
    parser.add_argument("--out", default="eval_results.json", help="Path to write detailed evaluation results.")
    parser.add_argument("--env-file", default=".env", help="Optional env file for GOOGLE_API_KEY.")
    args = parser.parse_args()

    load_env_file(args.env_file)

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_items = json.load(f)

    evaluation = evaluate(gold_items, args.k, run_generation=not args.skip_generation, retrieval_mode=args.retrieval)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    print(json.dumps(evaluation["summary"], indent=2))
    print(f"Saved detailed results to {args.out}")


if __name__ == "__main__":
    main()
