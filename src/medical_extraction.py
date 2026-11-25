
import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from src.mongodb_service import transcriptions_collection


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
# 2. MEDICAL EXTRACTION PROMPT
# ==========================================

def create_medical_prompt(transcript):

    return f"""
You are a medical information extraction assistant.

Extract medical information from the following audio transcript.

Return ONLY valid JSON. Do not include Markdown or explanations.

Use exactly this JSON structure:

{{
    "summary": "",
    "patient_symptoms": [],
    "observations": [],
    "diagnosis": "",
    "prescription": [
        {{
            "medicine_name": "",
            "dosage": "",
            "frequency": "",
            "duration": "",
            "instructions": "",
            "mention_type": ""
        }}
    ],
    "medicine_names": [],
    "doctor_advice": [],
    "follow_up": "",
    "uncertain_information": [],
    "fictional_sample": false
}}

IMPORTANT RULES:

1. Extract only information explicitly present in the transcript.
2. Never invent medicine names or dosages.
3. If a medicine name is unclear, use an empty string.
4. Preserve the medicine name exactly when clearly mentioned.
5. Extract dosage and frequency separately.
6. Identify whether a medicine is a prescription, example,
   general mention, or unclear mention.
7. Include medicines mentioned as fictional examples.
8. Set fictional_sample to true if the transcript identifies
   itself as fictional or synthetic.
9. If no medicine is mentioned, return an empty prescription list.
10. Use empty strings or empty lists for missing information.
11. Do not convert medicine mentions into medical advice.
12. medicine_names must contain only clearly identified names.
13. Preserve the original meaning of the transcript.

TRANSCRIPT:

{transcript}
"""


# ==========================================
# 3. EXTRACT RESPONSE TEXT
# ==========================================

def get_response_text(response):

    try:

        content = response.choices[0].message.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))

            return "\n".join(text_parts).strip()

        return str(content).strip() if content else ""

    except (AttributeError, IndexError, TypeError):

        return ""


# ==========================================
# 4. CLEAN JSON RESPONSE
# ==========================================

def clean_json_response(response_text):

    response_text = response_text.strip()

    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    return response_text.strip()


# ==========================================
# 5. PARSE JSON
# ==========================================

def parse_medical_json(response_text):

    cleaned_text = clean_json_response(response_text)

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found")

    json_text = cleaned_text[start:end + 1]

    medical_data = json.loads(json_text)

    if not isinstance(medical_data, dict):
        raise ValueError("Expected a JSON object")

    return medical_data


# ==========================================
# 6. VALIDATE MEDICAL DATA
# ==========================================

def validate_medical_data(medical_data):

    required_fields = {
        "summary": "",
        "patient_symptoms": [],
        "observations": [],
        "diagnosis": "",
        "prescription": [],
        "medicine_names": [],
        "doctor_advice": [],
        "follow_up": "",
        "uncertain_information": [],
        "fictional_sample": False
    }

    for field, default_value in required_fields.items():

        if field not in medical_data:
            medical_data[field] = default_value

    list_fields = [
        "patient_symptoms",
        "observations",
        "prescription",
        "medicine_names",
        "doctor_advice",
        "uncertain_information"
    ]

    for field in list_fields:

        if not isinstance(medical_data[field], list):
            medical_data[field] = []

    # Normalize medicine information.
    for medicine in medical_data["prescription"]:

        if not isinstance(medicine, dict):
            continue

        medicine_fields = [
            "medicine_name",
            "dosage",
            "frequency",
            "duration",
            "instructions",
            "mention_type"
        ]

        for field in medicine_fields:

            if field not in medicine:
                medicine[field] = ""

    return medical_data


# ==========================================
# 7. EXTRACT MEDICAL INFORMATION
# ==========================================

def extract_medical_information(transcript):

    prompt = create_medical_prompt(transcript)

    for attempt in range(1, 4):

        print(f"\nExtraction attempt: {attempt}")

        try:

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
                max_tokens=3500
            )

            response_text = get_response_text(response)

            if not response_text:

                raise ValueError(
                    "Empty response received from Sarvam AI"
                )

            medical_data = parse_medical_json(response_text)

            medical_data = validate_medical_data(
                medical_data
            )

            return medical_data

        except ApiError:
            raise

        except Exception as error:

            print("Extraction error:", str(error))

            if attempt == 3:
                raise

    raise RuntimeError("Medical extraction failed")


# ==========================================
# 8. SAVE MEDICAL INFORMATION
# ==========================================

def save_medical_information(file_name, medical_data):

    result = transcriptions_collection.update_one(
        {"file_name": file_name},
        {
            "$set": {
                "medical_information": medical_data,
                "extraction_status": "completed"
            }
        }
    )

    if result.matched_count == 0:

        raise ValueError(
            f"Transcription record not found: {file_name}"
        )

    print("Medical information saved in MongoDB!")


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

def display_results(file_name, medical_data):

    print("\n===================================")
    print("MEDICAL EXTRACTION RESULTS")
    print("===================================")

    print("Audio File:", file_name)

    print("\nSummary:")
    print(medical_data.get("summary", ""))

    print("\nPatient Symptoms:")

    for symptom in medical_data.get("patient_symptoms", []):
        print("-", symptom)

    print("\nObservations:")

    for observation in medical_data.get("observations", []):
        print("-", observation)

    print("\nDiagnosis:")
    print(medical_data.get("diagnosis", ""))

    print("\nMedicines:")

    prescriptions = medical_data.get("prescription", [])

    if not prescriptions:
        print("No medicines identified.")

    for medicine in prescriptions:

        if not isinstance(medicine, dict):
            continue

        print("\nMedicine:", medicine.get("medicine_name") or "Unclear")
        print("Dosage:", medicine.get("dosage") or "Not specified")
        print("Frequency:", medicine.get("frequency") or "Not specified")
        print("Duration:", medicine.get("duration") or "Not specified")
        print("Instructions:", medicine.get("instructions") or "Not specified")
        print("Mention Type:", medicine.get("mention_type") or "Not specified")

    print("\nDoctor Advice:")

    for advice in medical_data.get("doctor_advice", []):
        print("-", advice)

    print("\nFollow-up:")
    print(medical_data.get("follow_up", "") or "Not specified")

    print("\nFictional Sample:")
    print(medical_data.get("fictional_sample", False))

    print("\n===================================")


# ==========================================
# 10. PROCESS ONE AUDIO FILE
# ==========================================

def process_single_audio(record):

    file_name = record["file_name"]

    print("\nProcessing:", file_name)

    try:

        medical_data = extract_medical_information(
            record["transcript"]
        )

        save_medical_information(
            file_name,
            medical_data
        )

        display_results(
            file_name,
            medical_data
        )

        print("\nExtraction completed successfully!")

        return True

    except ApiError as error:

        print("\nSarvam API error:", error.status_code)
        print("Details:", error.body)

        return False

    except Exception as error:

        print("\nProcessing failed:", str(error))

        return False


# ==========================================
# 11. PROCESS ALL PENDING TRANSCRIPTIONS
# ==========================================

def process_transcriptions():

    records = transcriptions_collection.find({
        "status": "completed",
        "transcript": {
            "$exists": True,
            "$ne": ""
        }
    })

    successful = 0
    failed = 0
    skipped = 0

    for record in records:

        file_name = record.get("file_name", "Unknown")

        if record.get("extraction_status") == "completed":

            print(f"\nAlready completed: {file_name}")

            skipped += 1

            continue

        result = process_single_audio(record)

        if result:
            successful += 1
        else:
            failed += 1

    print("\n===================================")
    print("MEDICAL EXTRACTION REPORT")
    print("===================================")
    print("Newly successful:", successful)
    print("Failed:", failed)
    print("Already completed:", skipped)
    print("===================================")


# ==========================================
# 12. REPROCESS ONE AUDIO FILE
# ==========================================

def reprocess_single_audio(file_name):

    record = transcriptions_collection.find_one({
        "file_name": file_name,
        "status": "completed",
        "transcript": {
            "$exists": True,
            "$ne": ""
        }
    })

    if not record:

        print("Transcription not found:", file_name)

        return

    process_single_audio(record)


# ==========================================
# 13. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Process only audio files that have not
    # completed medical extraction.

    process_transcriptions()

    # To reprocess one specific file later,
    # use:
    #
    # reprocess_single_audio("audio5.wav")
