
from datetime import datetime, timezone

from src.mongodb_service import (
    transcriptions_collection,
    medical_chunks_collection
)

from src.embedding_service import generate_embeddings


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================
# 2. CREATE TEXT CHUNKS
# ==========================================

def create_chunks(text):

    if not text or not text.strip():
        return []

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 3. PREPARE MEDICAL TEXT
# ==========================================

def prepare_medical_text(record):

    medical_data = record.get("medical_information") or {}

    if not isinstance(medical_data, dict):
        medical_data = {}

    symptoms = medical_data.get("patient_symptoms", [])
    advice = medical_data.get("doctor_advice", [])

    if not isinstance(symptoms, list):
        symptoms = []

    if not isinstance(advice, list):
        advice = []

    parts = [
        "Audio File: " + record.get("file_name", ""),
        "Transcript: " + record.get("transcript", ""),
        "Medical Summary: " + str(
            medical_data.get("summary", "")
        ),
        "Symptoms: " + ", ".join(str(x) for x in symptoms),
        "Diagnosis: " + str(
            medical_data.get("diagnosis") or ""
        ),
        "Doctor Advice: " + ", ".join(str(x) for x in advice)
    ]

    prescriptions = medical_data.get("prescription", [])

    if isinstance(prescriptions, list):

        for medicine in prescriptions:

            if not isinstance(medicine, dict):
                continue

            medicine_text = (
                "Medicine: " +
                str(medicine.get("medicine_name") or "") +

                ", Dosage: " +
                str(medicine.get("dosage") or "") +

                ", Frequency: " +
                str(medicine.get("frequency") or "") +

                ", Duration: " +
                str(medicine.get("duration") or "") +

                ", Instructions: " +
                str(medicine.get("instructions") or "") +

                ", Mention Type: " +
                str(medicine.get("mention_type") or "")
            )

            parts.append(medicine_text)

    return "\n".join(parts)


# ==========================================
# 4. SAVE CHUNKS AND EMBEDDINGS
# ==========================================

def save_chunks(file_name, chunks):

    if not chunks:
        print("No chunks available.")
        return 0

    embeddings = generate_embeddings(chunks)

    # Remove outdated chunks for this audio file.
    medical_chunks_collection.delete_many({
        "file_name": file_name
    })

    total_chunks = 0

    for index, chunk in enumerate(chunks):

        document = {
            "file_name": file_name,
            "chunk_index": index,
            "text": chunk,
            "embedding": embeddings[index],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": len(embeddings[index]),
            "created_at": datetime.now(timezone.utc)
        }

        medical_chunks_collection.insert_one(document)

        total_chunks += 1

    return total_chunks


# ==========================================
# 5. PROCESS ALL COMPLETED MEDICAL RECORDS
# ==========================================

def process_medical_records():

    records = transcriptions_collection.find({
        "status": "completed",
        "extraction_status": "completed"
    })

    total_chunks = 0
    processed_files = 0
    failed_files = 0

    for record in records:

        file_name = record.get("file_name", "Unknown")

        print("\n================================")
        print("Processing:", file_name)
        print("================================")

        try:

            medical_text = prepare_medical_text(record)

            chunks = create_chunks(medical_text)

            print("Chunks created:", len(chunks))

            if not chunks:

                print("No text available. Skipping.")
                failed_files += 1
                continue

            saved_count = save_chunks(
                file_name,
                chunks
            )

            total_chunks += saved_count
            processed_files += 1

            print("Chunks saved:", saved_count)
            print("Embeddings saved successfully!")

        except Exception as error:

            print("Processing failed:", str(error))
            failed_files += 1

    print("\n================================")
    print("MEDICAL CHUNKING REPORT")
    print("================================")
    print("Files processed:", processed_files)
    print("Failed files:", failed_files)
    print("Total chunks:", total_chunks)
    print("Embedding dimensions: 384")
    print("MongoDB storage completed.")
    print("================================")


# ==========================================
# 6. REPROCESS A SINGLE AUDIO FILE
# ==========================================

def reprocess_single_chunk(file_name):

    record = transcriptions_collection.find_one({
        "file_name": file_name,
        "status": "completed",
        "extraction_status": "completed"
    })

    if not record:

        print("Completed medical record not found:", file_name)

        return

    print("\nUpdating chunks for:", file_name)

    medical_text = prepare_medical_text(record)

    chunks = create_chunks(medical_text)

    print("New chunks created:", len(chunks))

    if not chunks:

        print("No medical text available.")
        return

    saved_count = save_chunks(
        file_name,
        chunks
    )

    print("\nUpdated chunks:", saved_count)
    print("Embeddings updated successfully!")


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Process all five completed audio records.

    process_medical_records()

    # To update only one audio file later:
    # reprocess_single_chunk("audio5.wav")
