
from pathlib import Path
from datetime import datetime, timezone

from src.mongodb_service import audio_collection

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio_files"

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg"}


def get_audio_files():
    return [
        file for file in AUDIO_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS
    ]


def validate_audio(audio_path):
    if not audio_path.exists():
        return False, "File does not exist"

    if audio_path.stat().st_size == 0:
        return False, "Audio file is empty"

    return True, "Valid audio file"


def save_audio_record(audio_path):
    audio_path = Path(audio_path)

    audio_collection.update_one(
        {"file_name": audio_path.name},
        {
            "$set": {
                "file_name": audio_path.name,
                "file_path": str(audio_path),
                "file_size": audio_path.stat().st_size,
                "status": "pending",
                "updated_at": datetime.now(timezone.utc)
            },
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )


if __name__ == "__main__":
    files = get_audio_files()
    print("Total audio files found:", len(files))

    for file in files:
        valid, message = validate_audio(file)
        print(f"{file.name}: {message}")

        if valid:
            save_audio_record(file)
            print("MongoDB record updated successfully")
