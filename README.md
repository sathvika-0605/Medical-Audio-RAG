
# Medical Audio RAG

## Project Overview

Medical Audio RAG is an AI-powered multilingual healthcare
audio processing system that converts medical conversations
into structured and searchable medical information.

The system processes audio recordings, extracts medical
information, matches medicine names against a MongoDB
medicine catalogue, and generates structured JSON responses.

## Key Features

- Multilingual audio transcription
- English translation using Sarvam AI
- Medical information extraction
- Symptoms and medicine identification
- Dosage and prescription information extraction
- MongoDB medicine catalogue matching
- RAG-based medical question answering
- Streamlit interactive dashboard
- FastAPI REST API
- Structured JSON response and download

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend API |
| Streamlit | Frontend dashboard |
| Sarvam AI | Audio transcription and translation |
| MongoDB Atlas | Database |
| Sentence Transformers | Text embeddings |
| RAG | Context-based question answering |
| Uvicorn | Backend server |

## System Architecture

Audio Upload
    |
    v
Streamlit Frontend
    |
    v
FastAPI Backend
    |
    v
Sarvam AI Transcription and Translation
    |
    v
Medical Information Extraction
    |
    v
MongoDB Medicine Catalogue Matching
    |
    v
Structured JSON Response
    |
    v
Streamlit Results Dashboard

## Project Structure

    Medical-Audio-RAG/
    |
    |-- api.py
    |-- app.py
    |-- requirements.txt
    |-- README.md
    |-- .gitignore
    |-- .env.example
    |
    |-- src/
        |-- audio_pipeline.py
        |-- batch_transcription_service.py
        |-- chunking_service.py
        |-- embedding_service.py
        |-- generate_test_audio.py
        |-- medical_extraction.py
        |-- medicine_catalogue_service.py
        |-- medicine_match_service.py
        |-- medicine_report.py
        |-- mongodb_service.py
        |-- rag_service.py
        |-- response_builder.py
        |-- sarvam_service.py
        |-- translation_service.py
        |-- tts_service.py

## Installation

Clone the repository and open the project directory.

Create a Python virtual environment:

    python -m venv venv

Activate the virtual environment on Windows:

    .\venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r requirements.txt

## Environment Configuration

Create a .env file using .env.example as a reference.

Configure valid API credentials and the MongoDB connection
string.

Never commit actual API keys or database credentials.

## Running the Backend

Open a terminal and activate the virtual environment.

    python -m uvicorn api:app --reload

API documentation:

http://127.0.0.1:8000/docs

## Running the Frontend

Open another terminal and activate the virtual environment.

    python -m streamlit run app.py

Frontend URL:

http://localhost:8501

## API Endpoint

POST /process-audio

Request type: multipart/form-data

Audio field name: audio

Supported formats:

- WAV
- MP3
- M4A

Maximum file size: 20 MB

## Streamlit Dashboard

The application contains five main pages:

1. Overview
2. Process New Audio
3. Audio Records
4. Medicine Report
5. Ask Medical Assistant

## Database Collections

Database: medical_audio_rag

- audio_records
- transcriptions
- medical_chunks
- medicines_catalogue

## Medicine Catalogue

The system searches imported medicine records using
medicine names, compositions, and available dosage strengths.

Catalogue matches are informational and do not represent
clinical recommendations or approved medicine substitutions.

The original medicine dataset is not included in this
repository. Catalogue matching requires a populated MongoDB
medicine collection.

## Testing

The prototype has been tested using fictional English
and Telugu medical audio recordings.

## Current Limitations

- New audio processing requires available Sarvam AI credits.
- The medicine catalogue is partially imported.
- Transcription accuracy depends on audio quality.
- Actual speaker diarization is not implemented.
- Medical extraction and RAG responses require human review.

## Future Enhancements

- Offline speech recognition
- Speaker diarization
- Improved medicine matching
- Complete medicine catalogue integration
- Improved multilingual support
- Production deployment
- Authentication and access control

## Disclaimer

This project is a healthcare information processing
prototype intended for development and demonstration.

It is not a substitute for professional medical advice,
diagnosis, or treatment.
