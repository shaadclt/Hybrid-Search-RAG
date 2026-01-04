# app/generation.py
from groq import Groq
from config import LLM_MODEL, TEMPERATURE, GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def build_prompt(context: str, query: str) -> str:
    return f"""
You are an AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""


def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()
