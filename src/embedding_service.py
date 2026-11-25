
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. LOAD EMBEDDING MODEL
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")


# ==================================================
# 2. GENERATE SINGLE EMBEDDING
# ==================================================

def generate_embedding(text):

    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


# ==================================================
# 3. GENERATE MULTIPLE EMBEDDINGS
# ==================================================

def generate_embeddings(text_list):

    if not text_list:
        return []

    cleaned_texts = []

    for text in text_list:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("All input texts must be non-empty strings")

        cleaned_texts.append(text.strip())

    embeddings = model.encode(
        cleaned_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


# ==================================================
# 4. TEST EMBEDDING GENERATION
# ==================================================

if __name__ == "__main__":

    sample_text = (
        "Patient reports headache, tiredness and dehydration. "
        "The conversation mentions oral rehydration solution."
    )

    print("\nGenerating sample embedding...")

    vector = generate_embedding(sample_text)

    print("Embedding generated successfully!")

    print("Vector dimensions:", len(vector))

    print("First 10 values:", vector[:10])
