
import os
import io
import wave
import base64
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "generated_audio"

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY missing from .env")

client = SarvamAI(api_subscription_key=api_key)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONVERSATIONS = {
    "english_medical_test.wav": {
        "language": "en-IN",
        "lines": [
            ("patient", "Doctor, I have a fever and headache."),
            ("doctor", "How many days have you had these symptoms?"),
            ("patient", "For two days. I also have a mild cold."),
            ("doctor", "In this fictional example, the medicine is paracetamol.")
        ]
    },
    "telugu_medical_test.wav": {
        "language": "te-IN",
        "lines": [
            ("patient", "డాక్టర్ గారు, నాకు జ్వరం మరియు తలనొప్పి ఉన్నాయి."),
            ("doctor", "ఎన్ని రోజులుగా ఈ సమస్య ఉంది?"),
            ("patient", "రెండు రోజులుగా ఉంది. కొంచెం జలుబు కూడా ఉంది."),
            ("doctor", "ఇది కేవలం పరీక్ష కోసం. మందు పేరు పారాసిటమాల్. మోతాదు ఐదు వందల మిల్లీగ్రాములు.")
        ]
    }
}


def generate_audio(filename, config):
    frames = []
    audio_params = None

    for role, sentence in config["lines"]:

        speaker = "shubh" if role == "doctor" else "neha"

        print(f"Generating {role}: {sentence}")

        response = client.text_to_speech.convert(
            model="bulbul:v3",
            text=sentence,
            language_code=config["language"],
            speaker=speaker
        )

        audio_bytes = base64.b64decode(response.audios[0])

        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            params = wav.getparams()
            audio_data = wav.readframes(wav.getnframes())

        if audio_params is None:
            audio_params = params
        elif (
            params.nchannels != audio_params.nchannels
            or params.sampwidth != audio_params.sampwidth
            or params.framerate != audio_params.framerate
            or params.comptype != audio_params.comptype
        ):
            raise ValueError(
                "Generated audio formats do not match."
            )

        frames.append(audio_data)

        silence = (
            b"\x00"
            * int(audio_params.framerate * 0.4)
            * audio_params.nchannels
            * audio_params.sampwidth
        )

        frames.append(silence)

    output_path = OUTPUT_DIR / filename

    with wave.open(str(output_path), "wb") as output:
        output.setparams(audio_params)
        output.writeframes(b"".join(frames))

    with wave.open(str(output_path), "rb") as output:
        duration = (
            output.getnframes() / output.getframerate()
        )

    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.2f} seconds\n")


if __name__ == "__main__":

    for filename, config in CONVERSATIONS.items():
        generate_audio(filename, config)

    print("Both test recordings generated!")
