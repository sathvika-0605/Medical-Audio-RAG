
import os
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from src.mongodb_service import (
    audio_collection,
    transcriptions_collection
)


# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio_files"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

if not API_KEY:
    raise ValueError("SARVAM_API_KEY is missing from .env")


# --------------------------------------------------
# 2. Initialize Sarvam client
# --------------------------------------------------

client = SarvamAI(api_subscription_key=API_KEY)


# --------------------------------------------------
# 3. Transcribe audio
# --------------------------------------------------

def transcribe_audio(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty")

    with open(audio_path, "rb") as audio_file:

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v4",
            mode="translate"
        )

    return {
        "transcript": response.transcript,
        "language_code": response.language_code,
        "request_id": response.request_id
    }


# --------------------------------------------------
# 4. Save transcription in MongoDB
# --------------------------------------------------

def save_transcription(file_name, result):

    transcriptions_collection.update_one(
        {"file_name": file_name},
        {
            "$set": {
                "file_name": file_name,
                "transcript": result["transcript"],
                "language_code": result["language_code"],
                "request_id": result["request_id"],
                "status": "completed"
            }
        },
        upsert=True
    )

    audio_collection.update_one(
        {"file_name": file_name},
        {"$set": {"status": "transcribed"}}
    )


# --------------------------------------------------
# 5. Process one audio file
# --------------------------------------------------

def process_audio(audio_path):

    print(f"\nProcessing: {audio_path.name}")

    result = transcribe_audio(audio_path)

    save_transcription(audio_path.name, result)

    print("Transcription successful!")
    print("Detected language:", result["language_code"])
    print("Transcript:", result["transcript"])

    return result


# --------------------------------------------------
# 6. Run transcription
# --------------------------------------------------

if __name__ == "__main__":

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    print("Total audio files:", len(audio_files))

    for audio_path in audio_files:

        try:
            process_audio(audio_path)

        except ApiError as error:
            print("Sarvam API error:", error.status_code)
            print("Details:", error.body)

            if error.status_code == 403:
                print("Authentication failed. Check Sarvam API credentials.")

        except Exception as error:
            print(f"Error processing {audio_path.name}: {error}")
