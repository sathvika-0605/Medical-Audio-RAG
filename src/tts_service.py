
import os
import base64
import io
import wave
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "generated_audio"

# Load API key from the project root .env file
load_dotenv(PROJECT_DIR / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

MODEL = "bulbul:v3"
LANGUAGE_CODE = "te-IN"

# Sample Telugu doctor-patient conversation
DEFAULT_CONVERSATION = [
    (
        "patient",
        "నమస్కారం డాక్టర్ గారు. నాకు మూడు రోజులుగా జ్వరం, "
        "తలనొప్పి, ఒళ్ళు నొప్పులు ఉన్నాయి."
    ),
    (
        "doctor",
        "నమస్కారం. మీకు దగ్గు, జలుబు లేదా గొంతు నొప్పి కూడా ఉన్నాయా?"
    ),
    (
        "patient",
        "కొంచెం జలుబు ఉంది డాక్టర్ గారు. దగ్గు లేదు. "
        "రాత్రి సమయంలో జ్వరం ఎక్కువగా వస్తోంది."
    ),
    (
        "doctor",
        "సరే. మీకు ఏమైనా మందుల అలర్జీలు ఉన్నాయా? "
        "ప్రస్తుతం వేరే మందులు ఏమైనా వాడుతున్నారా?"
    ),
    (
        "patient",
        "నాకు తెలిసిన మందుల అలర్జీలు లేవు. "
        "ప్రస్తుతం వేరే మందులు ఏమీ వాడటం లేదు."
    ),
    (
        "doctor",
        "మీరు చెప్పిన లక్షణాలను నమోదు చేస్తున్నాను. "
        "జ్వరం కోసం పారాసిటమాల్ ఐదు వందల మిల్లీగ్రాముల మాత్ర "
        "గురించి సూచిస్తున్నాను."
    ),
    (
        "doctor",
        "మందు ఎంత మోతాదులో, రోజుకు ఎన్నిసార్లు, ఎన్ని రోజులు "
        "వాడాలో మీ వైద్యుడు ఇచ్చిన సూచనలను మాత్రమే పాటించండి. "
        "మీకు జ్వరం కొనసాగితే మళ్లీ వైద్యుడిని సంప్రదించండి."
    ),
    (
        "patient",
        "సరే డాక్టర్ గారు. మీరు చెప్పిన సూచనలు పాటిస్తాను. ధన్యవాదాలు."
    ),
]


def get_client():
    if not SARVAM_API_KEY:
        raise ValueError(
            "SARVAM_API_KEY is missing. Check your project .env file."
        )

    return SarvamAI(api_subscription_key=SARVAM_API_KEY)


def generate_telugu_audio(
    conversation=None,
    output_filename="telugu_healthcare_conversation.wav",
    silence_duration=0.65
):
    """
    Generate a Telugu doctor-patient conversation using Sarvam Bulbul v3.

    conversation format:
        [
            ("doctor", "Telugu sentence"),
            ("patient", "Telugu sentence")
        ]
    """

    if conversation is None:
        conversation = DEFAULT_CONVERSATION

    if not conversation:
        raise ValueError("Conversation cannot be empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / output_filename
    client = get_client()

    audio_frames = []
    audio_params = None

    for index, (role, sentence) in enumerate(conversation, start=1):

        if role not in ("doctor", "patient"):
            raise ValueError(
                f"Invalid role '{role}'. Use doctor or patient."
            )

        if not sentence.strip():
            continue

        # Speaker mapping from the supplied working example
        speaker = "shubh" if role == "doctor" else "neha"

        print(f"Generating line {index}/{len(conversation)}: {role}")

        response = client.text_to_speech.convert(
            model=MODEL,
            text=sentence,
            language_code=LANGUAGE_CODE,
            speaker=speaker
        )

        if not response.audios:
            raise RuntimeError(
                f"No audio returned for conversation line {index}."
            )

        audio_bytes = base64.b64decode(response.audios[0])

        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            current_params = wav_file.getparams()
            frames = wav_file.readframes(wav_file.getnframes())

        if audio_params is None:
            audio_params = current_params

        elif (
            current_params.nchannels != audio_params.nchannels
            or current_params.sampwidth != audio_params.sampwidth
            or current_params.framerate != audio_params.framerate
            or current_params.comptype != audio_params.comptype
        ):
            raise ValueError(
                "Audio segments have different WAV formats. "
                "They cannot safely be combined."
            )

        audio_frames.append(frames)

        # Add silence between speakers
        silence_frames = int(
            audio_params.framerate * silence_duration
        )

        silence = (
            b"\x00" * audio_params.nchannels
            * audio_params.sampwidth
            * silence_frames
        )

        audio_frames.append(silence)

    if not audio_frames or audio_params is None:
        raise RuntimeError("No audio segments were generated.")

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setparams(audio_params)
        output_wav.writeframes(b"".join(audio_frames))

    print("\nTelugu audio generated successfully.")
    print(f"Saved at: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_telugu_audio()
