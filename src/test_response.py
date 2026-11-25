
import json

from src.response_builder import build_response


result = build_response(
    source_language="te-IN",
    target_languages=["en-IN", "te-IN"],
    diarized=True,
    num_speakers=2,
    turns=[
        {
            "turnId": "turn_1",
            "speakerId": "speaker_0",
            "role": "provider",
            "startTimeMs": 1000,
            "endTimeMs": 4000,
            "originalText": "మీకు ఎన్ని రోజులుగా జ్వరం ఉంది?",
            "translations": {
                "en-IN": "How many days have you had a fever?",
                "te-IN": "మీకు ఎన్ని రోజులుగా జ్వరం ఉంది?"
            }
        }
    ],
    medical={
        "symptoms": ["Fever"],
        "medicines": [],
        "diagnosis": [],
        "doctorAdvice": [],
        "followUp": []
    },
    latency={
        "transcription": 1200,
        "diarization": 300,
        "translation": 900,
        "medical": 1500,
        "total": 3900
    }
)

print(json.dumps(result, indent=4, ensure_ascii=False))
