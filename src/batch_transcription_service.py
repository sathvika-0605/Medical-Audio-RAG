
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from sarvamai import SarvamAI

from src.mongodb_service import (
    audio_collection,
    transcriptions_collection
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    raise ValueError("SARVAM_API_KEY is missing from .env")

client = SarvamAI(api_subscription_key=API_KEY)

AUDIO_DIR = BASE_DIR / "audio_files"

OUTPUT_DIR = BASE_DIR / "batch_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. SAVE TRANSCRIPTION TO MONGODB
# ==========================================

def save_batch_transcription(file_name, result):

    transcript = result.get("transcript", "")

    if not transcript:
        raise ValueError("Empty transcript received")

    transcription_document = {
        "file_name": file_name,
        "transcript": transcript,
        "language_code": result.get("language_code"),
        "request_id": result.get("request_id"),
        "status": "completed",
        "transcription_method": "batch",
        "transcribed_at": datetime.now(timezone.utc)
    }

    transcriptions_collection.update_one(
        {"file_name": file_name},
        {"$set": transcription_document},
        upsert=True
    )

    audio_collection.update_one(
        {"file_name": file_name},
        {
            "$set": {
                "status": "transcribed",
                "transcription_method": "batch",
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    print("Transcription saved to MongoDB!")


# ==========================================
# 3. READ DOWNLOADED TRANSCRIPT
# ==========================================

def read_transcript(output_folder):

    json_files = list(output_folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            "No JSON transcript files were downloaded"
        )

    if len(json_files) != 1:
        raise ValueError(
            "Expected exactly one output JSON for this audio job"
        )

    with open(json_files[0], "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Unexpected transcript JSON format")

    return data


# ==========================================
# 4. PROCESS ONE AUDIO USING BATCH API
# ==========================================

def process_batch_audio(file_name):

    audio_path = AUDIO_DIR / file_name

    print("\n======================================")
    print("BATCH TRANSCRIPTION")
    print("======================================")
    print("Audio file:", file_name)

    if not audio_path.exists():
        print("Audio file not found:", audio_path)
        return False

    if audio_path.stat().st_size == 0:
        print("Audio file is empty.")
        return False

    output_folder = OUTPUT_DIR / audio_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    try:

        # Step 1: Create batch job
        print("\nStep 1: Creating Sarvam batch job...")

        job = client.speech_to_text_job.create_job(
            model="saaras:v4",
            mode="translate"
        )

        print("Batch job created successfully!")

        # Step 2: Upload audio
        print("\nStep 2: Uploading audio...")

        job.upload_files(
            file_paths=[str(audio_path)]
        )

        print("Audio uploaded successfully!")

        # Step 3: Start processing
        print("\nStep 3: Starting batch transcription...")

        job.start()

        print("Batch processing started!")

        # Step 4: Wait for completion
        print("\nStep 4: Waiting for Sarvam AI...")

        job.wait_until_complete(
            poll_interval=5,
            timeout=1800
        )

        print("Batch processing finished!")

        # Step 5: Check file results
        file_results = job.get_file_results()

        successful = file_results.get("successful", [])
        failed = file_results.get("failed", [])

        if failed:
            for item in failed:
                print(
                    "Transcription failed:",
                    item.get("file_name"),
                    item.get("error_message")
                )

        if not successful:
            print("No successful transcription results.")
            return False

        # Step 6: Download result
        print("\nStep 5: Downloading transcript...")

        job.download_outputs(
            output_dir=str(output_folder)
        )

        print("Transcript downloaded!")

        # Step 7: Read transcript JSON
        result = read_transcript(output_folder)

        transcript = result.get("transcript", "")

        if not transcript:
            raise ValueError("Transcript field is missing")

        print("\nTRANSCRIPTION RESULT")
        print("--------------------------------------")
        print(transcript)
        print("--------------------------------------")

        # Step 8: Save in MongoDB
        save_batch_transcription(
            file_name,
            result
        )

        print("\nSuccessfully processed:", file_name)

        return True

    except Exception as error:

        print("\nBatch transcription error:", str(error))

        return False


# ==========================================
# 5. PROCESS BOTH LONG AUDIO FILES
# ==========================================

def process_long_audio_files():

    files = [
        "audio1.wav",
        "audio4.wav"
    ]

    successful = 0
    failed = 0

    for file_name in files:

        result = process_batch_audio(file_name)

        if result:
            successful += 1
        else:
            failed += 1

    print("\n======================================")
    print("BATCH TRANSCRIPTION REPORT")
    print("======================================")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Total files:", len(files))
    print("======================================")


# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":

    process_long_audio_files()
