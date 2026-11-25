
from collections import defaultdict

from src.mongodb_service import transcriptions_collection


def get_medicine_details():
    """
    Collect medicine information from completed
    medical extraction records in MongoDB.
    """

    records = transcriptions_collection.find({
        "extraction_status": "completed"
    })

    medicine_data = defaultdict(lambda: {
        "mention_count": 0,
        "dosage": set(),
        "frequency": set(),
        "source_files": set(),
        "mention_types": set(),
        "fictional": False
    })

    for record in records:

        file_name = record.get("file_name", "Unknown file")
        medical_info = record.get("medical_information") or {}

        if not isinstance(medical_info, dict):
            continue

        prescriptions = medical_info.get("prescription", [])

        if not isinstance(prescriptions, list):
            continue

        for medicine in prescriptions:

            if not isinstance(medicine, dict):
                continue

            medicine_name = medicine.get("medicine_name")

            if not medicine_name or not str(medicine_name).strip():
                continue

            medicine_name = str(medicine_name).strip()
            key = medicine_name.lower()

            data = medicine_data[key]

            data["name"] = medicine_name
            data["mention_count"] += 1
            data["source_files"].add(file_name)

            dosage = medicine.get("dosage")
            frequency = medicine.get("frequency")
            mention_type = medicine.get("mention_type")

            if dosage:
                data["dosage"].add(str(dosage))

            if frequency:
                data["frequency"].add(str(frequency))

            if mention_type:
                data["mention_types"].add(str(mention_type))

            data["fictional"] = (
                data["fictional"]
                or bool(medical_info.get("fictional_sample", False))
            )

    results = list(medicine_data.values())

    results.sort(
        key=lambda item: item["mention_count"],
        reverse=True
    )

    return results


def display_medicine_report(top_n=10):

    medicines = get_medicine_details()

    print("\n======================================")
    print("       MEDICINE EXTRACTION REPORT")
    print("======================================")

    if not medicines:
        print("No clearly identified medicine names found.")
        return

    print("Total unique medicines:", len(medicines))
    print("Displaying top:", min(top_n, len(medicines)))

    for index, medicine in enumerate(medicines[:top_n], start=1):

        print(f"\n{index}. {medicine['name']}")
        print("   Mention count:", medicine["mention_count"])

        print(
            "   Dosage:",
            ", ".join(sorted(medicine["dosage"])) or "Not specified"
        )

        print(
            "   Frequency:",
            ", ".join(sorted(medicine["frequency"])) or "Not specified"
        )

        print(
            "   Source files:",
            ", ".join(sorted(medicine["source_files"]))
        )

        print(
            "   Mention type:",
            ", ".join(sorted(medicine["mention_types"])) or "Not specified"
        )

        print(
            "   Fictional sample:",
            "Yes" if medicine["fictional"] else "Not identified"
        )

    print("\n======================================")
    print("          REPORT COMPLETED")
    print("======================================")


if __name__ == "__main__":

    # Change to 5 if you want only the top 5 medicines.
    display_medicine_report(top_n=10)
