# app/evaluation.py
def recall_at_k(relevant_ids, retrieved_ids, k):
    retrieved_k = retrieved_ids[:k]
    return len(set(relevant_ids) & set(retrieved_k)) / len(relevant_ids)


def mrr(relevant_ids, retrieved_ids):
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1 / (i + 1)
    return 0.0
