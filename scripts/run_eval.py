# scripts/run_eval.py
from app.evaluation import recall_at_k, mrr

def run():
    relevant = ["doc1_0", "doc2_1"]
    retrieved = ["doc3_0", "doc2_1", "doc1_0"]

    print("Recall@2:", recall_at_k(relevant, retrieved, 2))
    print("MRR:", mrr(relevant, retrieved))


if __name__ == "__main__":
    run()
