
import json
from pathlib import Path
from collections import defaultdict

import requests
import streamlit as st

from src.mongodb_service import (
    audio_collection,
    transcriptions_collection,
    medical_chunks_collection,
)

from src.rag_service import (
    answer_medical_question,
    retrieve_relevant_chunks,
)


# ======================================
# CONFIGURATION
# ======================================

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_files"

API_URL = "http://127.0.0.1:8000/process-audio"
MAX_AUDIO_BYTES = 20 * 1024 * 1024

st.set_page_config(
    page_title="Medical Audio RAG Assistant",
    page_icon="🏥",
    layout="wide",
)


# ======================================
# DATABASE FUNCTIONS
# ======================================

def get_transcriptions():
    return list(transcriptions_collection.find({}))


def get_audio_records():
    return list(audio_collection.find({}))


def get_medicine_report():
    records = transcriptions_collection.find({
        "extraction_status": "completed"
    })

    medicines = defaultdict(lambda: {
        "name": "",
        "dosage": set(),
        "frequency": set(),
        "source_files": set(),
        "mention_types": set(),
        "fictional": False,
    })

    for record in records:
        medical_data = record.get("medical_information", {}) or {}

        if not isinstance(medical_data, dict):
            continue

        for medicine in medical_data.get("prescription", []):
            if not isinstance(medicine, dict):
                continue

            name = medicine.get("medicine_name")

            if not name or not str(name).strip():
                continue

            name = str(name).strip()
            key = name.lower()

            item = medicines[key]
            item["name"] = name
            item["source_files"].add(record.get("file_name", ""))

            if medicine.get("dosage"):
                item["dosage"].add(str(medicine["dosage"]))

            if medicine.get("frequency"):
                item["frequency"].add(str(medicine["frequency"]))

            if medicine.get("mention_type"):
                item["mention_types"].add(
                    str(medicine["mention_type"])
                )

            item["fictional"] = (
                item["fictional"]
                or bool(medical_data.get("fictional_sample", False))
            )

    return list(medicines.values())


# ======================================
# OVERVIEW PAGE
# ======================================

def show_overview():
    st.title("🏥 Medical Audio RAG Assistant")

    st.write(
        "AI-powered audio transcription, medical information "
        "extraction, medicine matching, and question answering."
    )

    st.divider()

    total_audio = audio_collection.count_documents({})

    total_transcriptions = transcriptions_collection.count_documents({
        "status": "completed"
    })

    total_extractions = transcriptions_collection.count_documents({
        "extraction_status": "completed"
    })

    total_chunks = medical_chunks_collection.count_documents({})

    medicines = get_medicine_report()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Audio Files", total_audio)
    col2.metric("Transcriptions", total_transcriptions)
    col3.metric("Medical Extractions", total_extractions)
    col4.metric("Stored Chunks", total_chunks)

    st.divider()

    st.subheader("Project Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "Audio transcription and translation are "
            "performed using Sarvam AI."
        )

    with col2:
        st.success(
            f"{len(medicines)} unique medicine names identified."
        )

    st.subheader("Audio Processing Status")

    for record in get_transcriptions():
        file_name = record.get("file_name", "Unknown")

        st.write(
            f"🎵 **{file_name}** | "
            f"Transcription: {record.get('status', 'Unknown')} | "
            f"Extraction: "
            f"{record.get('extraction_status', 'Pending')}"
        )


# ======================================
# AUDIO RECORDS PAGE
# ======================================

def show_audio_records():
    st.title("🎵 Audio Records")

    records = get_transcriptions()

    if not records:
        st.warning("No audio records found.")
        return

    file_names = [
        record["file_name"]
        for record in records
        if record.get("file_name")
    ]

    if not file_names:
        st.warning("No named audio recordings found.")
        return

    selected_file = st.selectbox(
        "Select an audio recording",
        file_names,
    )

    record = next(
        item for item in records
        if item.get("file_name") == selected_file
    )

    audio_path = AUDIO_DIR / selected_file

    st.divider()

    if audio_path.exists():
        st.subheader("Audio Player")
        st.audio(str(audio_path))
    else:
        st.warning("Audio file not found in audio_files.")

    st.subheader("Transcription")
    st.write(
        record.get("transcript") or
        "No transcript available."
    )

    medical_data = record.get("medical_information", {}) or {}

    if not isinstance(medical_data, dict):
        medical_data = {}

    st.divider()

    st.subheader("Medical Summary")
    st.write(
        medical_data.get("summary") or
        "Medical extraction not completed."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Symptoms")

        symptoms = medical_data.get("patient_symptoms") or []

        if symptoms:
            for symptom in symptoms:
                st.write(f"- {symptom}")
        else:
            st.info("No symptoms identified.")

    with col2:
        st.subheader("Diagnosis")
        st.write(
            medical_data.get("diagnosis") or
            "Not identified"
        )

    st.divider()

    st.subheader("Medicine Information")

    prescriptions = medical_data.get("prescription") or []

    if prescriptions:
        for medicine in prescriptions:
            if not isinstance(medicine, dict):
                continue

            st.markdown(
                f"**Medicine:** "
                f"{medicine.get('medicine_name') or 'Unclear'}"
            )

            st.write(
                f"Dosage: "
                f"{medicine.get('dosage') or 'Not specified'}"
            )

            st.write(
                f"Frequency: "
                f"{medicine.get('frequency') or 'Not specified'}"
            )

            st.write(
                f"Mention type: "
                f"{medicine.get('mention_type') or 'Not specified'}"
            )

            st.divider()
    else:
        st.info("No clearly identified medicines.")


# ======================================
# MEDICINE REPORT PAGE
# ======================================

def show_medicine_report():
    st.title("💊 Medicine Extraction Report")

    medicines = get_medicine_report()

    st.metric("Total Unique Medicines", len(medicines))

    if not medicines:
        st.info("No medicine information found.")
        return

    for index, medicine in enumerate(medicines, start=1):
        with st.container(border=True):
            st.subheader(f"{index}. {medicine['name']}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "Dosage: " +
                    (
                        ", ".join(sorted(medicine["dosage"]))
                        or "Not specified"
                    )
                )

                st.write(
                    "Frequency: " +
                    (
                        ", ".join(sorted(medicine["frequency"]))
                        or "Not specified"
                    )
                )

            with col2:
                st.write(
                    "Source: " +
                    ", ".join(sorted(medicine["source_files"]))
                )

                st.write(
                    "Mention type: " +
                    (
                        ", ".join(
                            sorted(medicine["mention_types"])
                        )
                        or "Not specified"
                    )
                )

            if medicine["fictional"]:
                st.warning(
                    "Fictional test example, not a real prescription."
                )


# ======================================
# RAG ASSISTANT PAGE
# ======================================

def show_medical_assistant():
    st.title("🤖 Ask Medical Assistant")

    st.write(
        "Ask questions about your stored medical audio recordings."
    )

    st.info(
        "This assistant searches test recordings. "
        "It does not provide medical diagnosis or treatment advice."
    )

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: Which medicine was mentioned for fever?"
        ),
    )

    if st.button(
        "Search and Generate Answer",
        type="primary",
    ):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Searching medical database..."):
                retrieved_chunks = retrieve_relevant_chunks(
                    question,
                    top_k=3,
                )

                answer = answer_medical_question(
                    question,
                    top_k=3,
                )

            st.subheader("AI Answer")
            st.write(answer)

            st.divider()

            st.subheader("Retrieved Sources")

            for item in retrieved_chunks:
                with st.container(border=True):
                    st.write(
                        "Source:",
                        item.get("file_name", "Unknown"),
                    )

                    st.write(
                        "Similarity:",
                        f"{item.get('similarity', 0):.4f}",
                    )

                    st.caption(
                        str(item.get("text", ""))[:500]
                    )

        except Exception as error:
            st.error(
                f"Unable to generate answer: {error}"
            )


# ======================================
# PROCESS NEW AUDIO PAGE
# ======================================

def show_process_audio():
    st.title("🎙️ Process New Audio")

    st.write(
        "Upload medical audio to generate a transcription, "
        "extract medical information, and find medicine "
        "catalogue matches."
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        key="new_audio_upload",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        st.caption(
            f"Selected: {uploaded_file.name} "
            f"({uploaded_file.size / 1024 / 1024:.2f} MB)"
        )

    process_clicked = st.button(
        "Process Audio",
        type="primary",
        disabled=uploaded_file is None,
    )

    if process_clicked:
        st.session_state.pop(
            "latest_audio_response",
            None,
        )

        if uploaded_file.size == 0:
            st.error("The selected audio file is empty.")

        elif uploaded_file.size > MAX_AUDIO_BYTES:
            st.error(
                "Audio must be 20 MB or smaller."
            )

        else:
            try:
                with st.spinner(
                    "Processing audio. Please wait..."
                ):
                    uploaded_file.seek(0)

                    # IMPORTANT:
                    # FastAPI expects the field name "audio".
                    response = requests.post(
                        API_URL,
                        files={
                            "audio": (
                                uploaded_file.name,
                                uploaded_file,
                                "application/octet-stream",
                            )
                        },
                        timeout=180,
                    )

                    response.raise_for_status()

                    result = response.json()

                    if (
                        not isinstance(result, dict)
                        or result.get("success") is not True
                    ):
                        raise ValueError(
                            "The API did not return a "
                            "successful processing result."
                        )

                    st.session_state[
                        "latest_audio_response"
                    ] = result

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Start the backend first."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Processing timed out. "
                    "Check the FastAPI terminal."
                )

            except requests.exceptions.HTTPError as error:
                detail = (
                    error.response.text[:1000]
                    if error.response is not None
                    else str(error)
                )

                st.error(
                    f"FastAPI returned an error: {detail}"
                )

            except (
                requests.exceptions.RequestException,
                ValueError,
            ) as error:
                st.error(
                    f"Unable to process audio: {error}"
                )

    result = st.session_state.get(
        "latest_audio_response"
    )

    if not result:
        return

    st.success("Audio processing completed!")

    data = result.get("data") or {}
    turns = data.get("turns") or []
    medical = data.get("medical") or {}

    # ==================================
    # PROCESSING INFORMATION
    # ==================================

    st.subheader("Processing Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detected Language",
        data.get("sourceLanguageCode") or "Unknown",
    )

    col2.metric(
        "Processing Status",
        data.get("processingStatus") or "Completed",
    )

    latency = data.get("latencyMs") or {}

    col3.metric(
        "Total Processing Time",
        f"{latency.get('total', 0)} ms",
    )

    # ==================================
    # TRANSCRIPTION
    # ==================================

    st.divider()

    st.subheader("📝 Transcription / English Translation")

    if turns:
        for turn in turns:
            if not isinstance(turn, dict):
                continue

            translations = turn.get("translations") or {}

            english_text = (
                translations.get("en-IN")
                or translations.get("en-US")
            )

            text = english_text or next(
                (
                    value
                    for value in translations.values()
                    if value
                ),
                None,
            )

            st.write(
                text
                or turn.get("originalText")
                or "No transcription text available."
            )

    else:
        st.info(
            "No transcription turns were returned."
        )

    # ==================================
    # MEDICAL EXTRACTION
    # ==================================

    st.divider()

    st.subheader("🏥 Medical Information")

    st.markdown("**Medical Summary**")

    st.write(
        medical.get("summary") or "Not available"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Symptoms**")

        symptoms = medical.get("patient_symptoms") or []

        if symptoms:
            for symptom in symptoms:
                st.write(f"- {symptom}")
        else:
            st.write("Not identified")

    with col2:
        st.markdown("**Diagnosis**")

        st.write(
            medical.get("diagnosis")
            or "Not identified"
        )

    if medical.get("fictional_sample"):
        st.warning(
            "This recording is identified "
            "as a fictional test sample."
        )

    # ==================================
    # PRESCRIPTION
    # ==================================

    st.divider()

    st.subheader("💊 Extracted Medicine Information")

    prescriptions = medical.get("prescription") or []

    if prescriptions:
        for index, medicine in enumerate(
            prescriptions,
            start=1,
        ):
            if not isinstance(medicine, dict):
                continue

            with st.container(border=True):
                st.markdown(
                    f"**Medicine {index}:** "
                    f"{medicine.get('medicine_name') or 'Unclear'}"
                )

                st.write(
                    "Dosage:",
                    medicine.get("dosage")
                    or "Not specified",
                )

                st.write(
                    "Frequency:",
                    medicine.get("frequency")
                    or "Not specified",
                )

                st.write(
                    "Duration:",
                    medicine.get("duration")
                    or "Not specified",
                )

                st.write(
                    "Instructions:",
                    medicine.get("instructions")
                    or "Not specified",
                )

                st.write(
                    "Mention Type:",
                    medicine.get("mention_type")
                    or "Not specified",
                )

    else:
        st.info(
            "No prescription information extracted."
        )

    uncertainties = (
        medical.get("uncertain_information") or []
    )

    if uncertainties:
        st.warning(
            "Uncertain information: "
            + "; ".join(map(str, uncertainties))
        )

    # ==================================
    # MEDICINE CATALOGUE MATCHING
    # ==================================

    st.divider()

    st.subheader("🔍 MongoDB Medicine Catalogue Matches")

    matches = (
        data.get("medicineCatalogueMatches") or []
    )

    if not matches:
        st.info(
            "No catalogue matches returned. "
            "Medicine names are not guessed "
            "when the audio is unclear."
        )

    for entry in matches:
        if not isinstance(entry, dict):
            continue

        st.markdown(
            "**Extracted Medicine:** "
            + str(
                entry.get("extractedMedicine")
                or "Unclear"
            )
        )

        st.caption(
            "Extracted Dosage: "
            + str(
                entry.get("extractedDosage")
                or "Not specified"
            )
        )

        products = (
            entry.get("catalogueMatches") or []
        )

        if not products:
            st.info(
                "No matching products found "
                "in the imported catalogue."
            )

        for product in products:
            if not isinstance(product, dict):
                continue

            with st.container(border=True):
                st.markdown(
                    "**Product:** "
                    + str(
                        product.get("productName")
                        or "Unknown"
                    )
                )

                st.write(
                    "Composition:",
                    product.get("composition")
                    or "Not available",
                )

                st.write(
                    "Match Type:",
                    product.get("matchType")
                    or "Not specified",
                )

                st.write(
                    "Strength Matches:",
                    product.get("strengthMatches"),
                )

                if product.get("isCombinationProduct"):
                    st.warning(
                        "Combination product: "
                        "not interchangeable with "
                        "a single-ingredient medicine."
                    )

                if product.get("productID"):
                    st.caption(
                        f"Product ID: {product['productID']}"
                    )

    st.caption(
        "Catalogue matches are informational only. "
        "They are not prescribing or substitution advice."
    )

    # ==================================
    # COMPLETE JSON RESPONSE
    # ==================================

    st.divider()

    st.subheader("📋 Complete API JSON Response")

    st.json(result)

    json_text = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        label="⬇️ Download JSON",
        data=json_text,
        file_name=(
            f"{result.get('requestId') or 'medical_audio_result'}.json"
        ),
        mime="application/json",
    )


# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("🏥 Medical Audio RAG")

st.sidebar.caption(
    "AI-powered medical audio assistant"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Process New Audio",
        "Audio Records",
        "Medicine Report",
        "Ask Medical Assistant",
    ],
)

st.sidebar.divider()

st.sidebar.subheader("Technologies")

st.sidebar.write("Python")
st.sidebar.write("Sarvam AI")
st.sidebar.write("MongoDB Atlas")
st.sidebar.write("Sentence Transformers")
st.sidebar.write("RAG")
st.sidebar.write("Streamlit")

if st.sidebar.button("Refresh Dashboard"):
    st.rerun()


# ======================================
# PAGE NAVIGATION
# ======================================

if page == "Overview":
    show_overview()

elif page == "Process New Audio":
    show_process_audio()

elif page == "Audio Records":
    show_audio_records()

elif page == "Medicine Report":
    show_medicine_report()

elif page == "Ask Medical Assistant":
    show_medical_assistant()


# ======================================
# FOOTER
# ======================================

st.divider()

st.caption(
    "Medical Audio RAG Assistant | "
    "Powered by Python, Sarvam AI and MongoDB"
)
