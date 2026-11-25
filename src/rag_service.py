
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

from src.mongodb_service import medical_chunks_collection
from src.embedding_service import generate_embedding


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    raise ValueError("SARVAM_API_KEY is missing from .env")

client = SarvamAI(api_subscription_key=API_KEY)


# ==========================================
# 2. COSINE SIMILARITY
# ==========================================

def calculate_similarity(vector1, vector2):

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    denominator = (
        np.linalg.norm(vector1) *
        np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    similarity = np.dot(
        vector1,
        vector2
    ) / denominator

    return float(similarity)


# ==========================================
# 3. RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve_relevant_chunks(question, top_k=3):

    question_embedding = generate_embedding(question)

    documents = medical_chunks_collection.find(
        {
            "embedding": {"$exists": True}
        },
        {
            "file_name": 1,
            "text": 1,
            "embedding": 1,
            "_id": 0
        }
    )

    results = []

    for document in documents:

        similarity = calculate_similarity(
            question_embedding,
            document["embedding"]
        )

        results.append({
            "file_name": document["file_name"],
            "text": document["text"],
            "similarity": similarity
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]


# ==========================================
# 4. GENERATE ANSWER USING SARVAM AI
# ==========================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:

        return "No relevant medical information was found."

    context = "\n\n".join(
        [
            f"Source: {item['file_name']}\n{item['text']}"
            for item in retrieved_chunks
        ]
    )

    prompt = f"""
You are a medical information assistant.

Answer the user's question using only the retrieved context.

IMPORTANT RULES:

1. Do not invent medical facts.
2. Do not invent medicine names or dosages.
3. If information is missing, clearly say it is not available.
4. Mention the audio filename supporting the answer.
5. Clearly distinguish an example from a real prescription.
6. All five recordings are fictional test samples.
7. Never recommend that a person take a mentioned medicine.
8. Keep answers simple and accurate.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = client.chat.completions(
        model="sarvam-105b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        reasoning_effort="low",
        max_tokens=1500
    )

    answer = response.choices[0].message.content

    if not answer:

        return "Sarvam AI could not generate an answer."

    return answer.strip()


# ==========================================
# 5. COMPLETE RAG PIPELINE
# ==========================================

def answer_medical_question(question, top_k=3):

    print("\nSearching medical database...")

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        top_k
    )

    print("\nRetrieved Sources:")

    if not retrieved_chunks:

        print("No relevant sources found.")

        return "No relevant medical information was found."

    for item in retrieved_chunks:

        print(
            f"{item['file_name']} | "
            f"Similarity: {item['similarity']:.4f}"
        )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer


# ==========================================
# 6. TEST FIVE MEDICAL QUESTIONS
# ==========================================

def test_multiple_questions():

    questions = [

        "What medicine was mentioned for fever?",

        "Which medicine was mentioned for seasonal allergies?",

        "What symptoms were reported in the cough recording?",

        "What was mentioned for stomach discomfort?",

        "What symptoms were reported in audio2?"

    ]

    print("\n======================================")
    print("MEDICAL AUDIO RAG TESTING")
    print("======================================")

    successful = 0
    failed = 0

    for index, question in enumerate(questions, start=1):

        print("\n\n======================================")
        print(f"QUESTION {index}")
        print("======================================")

        print("Question:", question)

        try:

            answer = answer_medical_question(
                question,
                top_k=3
            )

            print("\nRAG ANSWER")
            print("--------------------------------------")
            print(answer)

            successful += 1

        except Exception as error:

            print("\nQuestion failed:", str(error))

            failed += 1

    print("\n======================================")
    print("RAG TEST REPORT")
    print("======================================")
    print("Total questions:", len(questions))
    print("Successful:", successful)
    print("Failed:", failed)
    print("======================================")


# ==========================================
# 7. INTERACTIVE QUESTION MODE
# ==========================================

def interactive_question_mode():

    print("\n======================================")
    print("MEDICAL AUDIO RAG ASSISTANT")
    print("======================================")

    print("Ask questions about your five audio files.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() in ["exit", "quit"]:

            print("Exiting Medical RAG Assistant.")
            break

        if not question:

            print("Please enter a question.")
            continue

        try:

            answer = answer_medical_question(question)

            print("\nANSWER")
            print("--------------------------------------")
            print(answer)

        except Exception as error:

            print("Error:", str(error))


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Run five predefined questions first.

    test_multiple_questions()

    # After testing, uncomment this line
    # to enable interactive question answering.

    # interactive_question_mode()
