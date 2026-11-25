
import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    raise ValueError("SARVAM_API_KEY is missing from .env")

client = SarvamAI(api_subscription_key=API_KEY)

MAX_TEXT_LENGTH = 1000


def chunk_text(text, max_length=MAX_TEXT_LENGTH):
    """Split long text into manageable chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []

    while len(text) > max_length:
        split_at = text.rfind(" ", 0, max_length)

        if split_at < max_length // 2:
            split_at = max_length

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        chunks.append(text)

    return chunks


def translate_text(text, target_language="en-IN"):
    """
    Translate text using Sarvam Mayura.
    Returns translated text.
    """

    if not text or not text.strip():
        return ""

    text_chunks = chunk_text(text)
    translated_parts = []

    for chunk in text_chunks:

        response = client.text.translate(
            input=chunk,
            source_language_code="auto",
            target_language_code=target_language,
            model="mayura:v1"
        )

        translated_parts.append(response.translated_text)

    return " ".join(translated_parts)


def translate_to_english(text):
    return translate_text(text, "en-IN")


def translate_to_telugu(text):
    return translate_text(text, "te-IN")


if __name__ == "__main__":

    sample_text = "నాకు రెండు రోజులుగా జ్వరం ఉంది"

    result = translate_to_english(sample_text)

    print("Original:", sample_text)
    print("English translation:", result)
