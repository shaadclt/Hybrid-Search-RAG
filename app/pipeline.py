# app/pipeline.py
from generation import build_prompt, generate_answer


class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, k: int):
        docs = self.retriever.retrieve(query, k)
        context = "\n\n".join(doc["text"] for doc in docs)

        prompt = build_prompt(context, query)
        return generate_answer(prompt)
