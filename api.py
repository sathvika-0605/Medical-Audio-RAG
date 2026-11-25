
import uuid
import time
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.response_builder import build_response
from src.sarvam_service import (
    transcribe_audio,
    save_transcription
)
from src.medical_extraction import (
    extract_medical_information,
    save_medical_information
)
from src.medicine_match_service import (
    match_extracted_medicines
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multilingual Healthcare Voice Assistant API",
    version="1.1.0"
)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploaded_audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

MAX_UPLOAD_SIZE = 20 * 1024 * 1024


@app.get("/")
def home():
    return {
        "message": "Healthcare Voice Assistant API is running",
        "status": "success",
        "medicineCatalogue": "enabled"
    }


def process_saved_audio(audio_path, file_name):
    """
    Process uploaded audio through:
    1. Sarvam AI transcription
    2. Medical information extraction
    3. MongoDB medicine catalogue matching
    4. Structured JSON response
    """

    start_total = time.perf_counter()

    # ----------------------------------
    # STEP 1: TRANSCRIBE AUDIO
    # ----------------------------------

    start = time.perf_counter()

    transcription = transcribe_audio(audio_path)

    transcription_ms = round(
        (time.perf_counter() - start) * 1000
    )

    transcript = transcription.get("transcript", "")

    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(
            "Sarvam returned an empty transcript."
        )

    save_transcription(file_name, transcription)

    logger.info("Audio transcription completed.")

    # ----------------------------------
    # STEP 2: EXTRACT MEDICAL INFORMATION
    # ----------------------------------

    start = time.perf_counter()

    medical_data = extract_medical_information(
        transcript
    )

    medical_ms = round(
        (time.perf_counter() - start) * 1000
    )

    if not isinstance(medical_data, dict):
        raise ValueError(
            "Medical extraction must return a dictionary."
        )

    save_medical_information(
        file_name,
        medical_data
    )

    logger.info("Medical extraction completed.")

    # ----------------------------------
    # STEP 3: MATCH MEDICINES IN MONGODB
    # ----------------------------------

    start = time.perf_counter()

    medicine_matches = []
    medicine_match_status = "completed"
    medicine_match_error = None

    try:
        medicine_matches = match_extracted_medicines(
            medical_data
        )

        logger.info(
            "Medicine catalogue matching completed."
        )

    except Exception as error:
        # A medicine search failure should not
        # discard successful transcription and
        # medical extraction results.

        medicine_match_status = "failed"
        medicine_match_error = type(error).__name__

        logger.exception(
            "Medicine catalogue matching failed."
        )

    medicine_match_ms = round(
        (time.perf_counter() - start) * 1000
    )

    # ----------------------------------
    # STEP 4: PREPARE TRANSCRIPTION TURNS
    # ----------------------------------

    # Sarvam is currently used in translate mode.
    # Original-language text, diarization and
    # timestamps are not available from this call.

    turns = [
        {
            "turnId": "turn_1",
            "speakerId": None,
            "role": "unknown",
            "startTimeMs": None,
            "endTimeMs": None,
            "originalText": None,
            "translations": {
                "en-IN": transcript
            }
        }
    ]

    # ----------------------------------
    # STEP 5: BUILD JSON RESPONSE
    # ----------------------------------

    total_ms = round(
        (time.perf_counter() - start_total) * 1000
    )

    result = build_response(
        source_language=(
            transcription.get("language_code") or "und"
        ),
        target_languages=["en-IN"],
        stt_model="saaras:v4",
        translation_model="not_run_separately",
        diarized=False,
        num_speakers=0,
        turns=turns,
        medical=medical_data,
        latency={
            "transcription": transcription_ms,
            "diarization": 0,
            "translation": 0,
            "medical": medical_ms,
            "medicineMatching": medicine_match_ms,
            "total": total_ms
        }
    )

    # ----------------------------------
    # STEP 6: ADD MEDICINE CATALOGUE DATA
    # ----------------------------------

    result["data"]["medicineCatalogueMatches"] = (
        medicine_matches
    )

    result["data"]["medicineCatalogueStatus"] = (
        medicine_match_status
    )

    if medicine_match_error is not None:
        result["data"]["medicineCatalogueErrorType"] = (
            medicine_match_error
        )

    result["data"]["processingStatus"] = "completed"

    result["data"]["uploadedFile"] = file_name

    return result


@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...)
):

    extension = Path(
        audio.filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only WAV, MP3 and M4A "
                "files are supported."
            )
        )

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    audio_path = UPLOAD_DIR / unique_filename

    try:
        # ----------------------------------
        # SAVE UPLOADED AUDIO
        # ----------------------------------

        total_size = 0

        with open(audio_path, "wb") as output:

            while True:

                chunk = await audio.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Audio file exceeds 20 MB."
                        )
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty."
            )

        # ----------------------------------
        # PROCESS AUDIO
        # ----------------------------------

        result = await run_in_threadpool(
            process_saved_audio,
            audio_path,
            unique_filename
        )

        return JSONResponse(
            content=result,
            status_code=200
        )

    except HTTPException:

        audio_path.unlink(
            missing_ok=True
        )

        raise

    except Exception as error:

        logger.exception(
            "Audio processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": (
                    "Audio processing failed."
                ),
                "errorType": (
                    type(error).__name__
                ),
                "error": str(error)
            }
        )

    finally:
        await audio.close()
