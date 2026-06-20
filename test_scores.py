import rag_pipeline

questions = [
    "What is the capital of France?",
    "What is the policy for medical examination of a person found fit for grant of commission?",
    "asdkjaslkdj random gibberish question"
]

for q in questions:
    results = rag_pipeline.retrieve(q)
    top_score = results[0]["score"] if results else 0
    print(f"Q: {q}\nTop Fused Score: {top_score}\n---")
