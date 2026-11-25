
from src.tts_service import generate_telugu_audio


if __name__ == "__main__":
    audio_path = generate_telugu_audio()

    print("\nGenerated audio file:")
    print(audio_path)
