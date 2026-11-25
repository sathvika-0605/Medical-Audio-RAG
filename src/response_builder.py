
import uuid
from datetime import datetime


def build_response(
    source_language="te-IN",
    target_languages=None,
    stt_model="saaras:v4",
    translation_model="mayura",
    diarized=False,
    num_speakers=0,
    turns=None,
    medical=None,
    latency=None,
    success=True,
    status=200
):
    """
    Builds the structured JSON response for the
    Healthcare Voice Assistant.
    """

    if target_languages is None:
        target_languages = ["en-IN"]

    if turns is None:
        turns = []

    if medical is None:
        medical = {}

    if latency is None:
        latency = {}

    response = {
        "success": success,
        "requestId": f"medical_audio_{uuid.uuid4().hex[:12]}",
        "data": {
            "sourceLanguageCode": source_language,
            "targetLanguageCodes": target_languages,
            "sttModel": stt_model,
            "translationModel": translation_model,
            "diarized": diarized,
            "numSpeakers": num_speakers,
            "turns": turns,
            "medical": medical,
            "latencyMs": {
                "transcription": latency.get("transcription", 0),
                "diarization": latency.get("diarization", 0),
                "translation": latency.get("translation", 0),
                "medical": latency.get("medical", 0),
                "total": latency.get("total", 0)
            }
        },
        "status": status
    }

    return response
