
import zipfile
import re
from pathlib import Path

import ijson
from pymongo import UpdateOne, ASCENDING
from pymongo.errors import PyMongoError

from src.mongodb_service import medicines_collection


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ZIP_FILENAME = "telemedicineapi.indiandrugs.zip"

BATCH_SIZE = 500


# --------------------------------------------------
# 2. Find medicine ZIP file
# --------------------------------------------------

def find_zip_file():

    expected_path = BASE_DIR / ZIP_FILENAME

    if expected_path.is_file():
        return expected_path

    zip_files = list(BASE_DIR.glob("*.zip"))

    if len(zip_files) == 1:
        print(
            "Using ZIP file:",
            zip_files[0].name
        )
        return zip_files[0]

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP file found in {BASE_DIR}"
        )

    raise ValueError(
        "Multiple ZIP files found. "
        "Set ZIP_FILENAME to the correct filename."
    )


# --------------------------------------------------
# 3. Import one batch into MongoDB
# --------------------------------------------------

def save_batch(operations):

    if not operations:
        return 0, 0

    result = medicines_collection.bulk_write(
        operations,
        ordered=False
    )

    return (
        result.upserted_count,
        result.modified_count
    )


# --------------------------------------------------
# 4. Import medicines from ZIP
# --------------------------------------------------

def import_medicines():

    zip_path = find_zip_file()

    print("\n================================")
    print("INDIAN MEDICINE DATABASE IMPORT")
    print("================================")

    print("ZIP file:", zip_path.name)
    print("Connecting to MongoDB...")

    medicines_collection.database.client.admin.command(
        "ping"
    )

    print("MongoDB connected successfully!")

    # Unique index prevents duplicate product IDs
    medicines_collection.create_index(
        [("productID", ASCENDING)],
        unique=True,
        name="unique_product_id"
    )

    total_processed = 0
    total_new = 0
    total_updated = 0
    total_skipped = 0

    operations = []

    with zipfile.ZipFile(zip_path, "r") as archive:

        json_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".json")
        ]

        if len(json_files) != 1:
            raise ValueError(
                "Expected exactly one JSON file in ZIP."
            )

        json_filename = json_files[0]

        print("Reading:", json_filename)
        print("Importing medicine records...\n")

        with archive.open(json_filename) as file:

            records = ijson.items(
                file,
                "item",
                use_float=True
            )

            for medicine in records:

                if not isinstance(medicine, dict):
                    total_skipped += 1
                    continue

                product_id = medicine.get("productID")

                if not product_id:
                    total_skipped += 1
                    continue

                # Ignore exported MongoDB _id.
                # MongoDB creates its own _id.
                medicine.pop("_id", None)

                operations.append(
                    UpdateOne(
                        {"productID": product_id},
                        {"$set": medicine},
                        upsert=True
                    )
                )

                if len(operations) >= BATCH_SIZE:

                    new_count, updated_count = save_batch(
                        operations
                    )

                    total_new += new_count
                    total_updated += updated_count
                    total_processed += len(operations)

                    operations.clear()

                    print(
                        f"Processed: {total_processed:,}",
                        flush=True
                    )

            # Save remaining records
            if operations:

                new_count, updated_count = save_batch(
                    operations
                )

                total_new += new_count
                total_updated += updated_count
                total_processed += len(operations)

                operations.clear()

    print("\n================================")
    print("IMPORT COMPLETED")
    print("================================")

    print(f"Processed: {total_processed:,}")
    print(f"New records: {total_new:,}")
    print(f"Updated records: {total_updated:,}")
    print(f"Skipped: {total_skipped:,}")

    total_records = medicines_collection.count_documents({})

    print(f"Total MongoDB records: {total_records:,}")

    print("================================")


# --------------------------------------------------
# 5. Search medicines by name or composition
# --------------------------------------------------

def search_medicine(medicine_name, limit=5):

    if not isinstance(medicine_name, str):
        return []

    medicine_name = medicine_name.strip()

    if not medicine_name:
        return []

    limit = max(1, min(int(limit), 20))

    pattern = re.escape(medicine_name)

    query = {
        "$or": [
            {
                "productName": {
                    "$regex": pattern,
                    "$options": "i"
                }
            },
            {
                "composition": {
                    "$regex": pattern,
                    "$options": "i"
                }
            }
        ]
    }

    projection = {
        "_id": 0,
        "productID": 1,
        "productName": 1,
        "composition": 1,
        "manufacturer": 1,
        "MRP": 1,
        "packaging": 1,
        "schedule": 1
    }

    results = medicines_collection.find(
        query,
        projection
    ).limit(limit)

    return list(results)


# --------------------------------------------------
# 6. Run importer
# --------------------------------------------------

if __name__ == "__main__":

    try:
        import_medicines()

    except (
        FileNotFoundError,
        ValueError,
        zipfile.BadZipFile,
        PyMongoError
    ) as error:

        print("\nIMPORT FAILED")
        print("Error:", error)
        raise
