
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")

DATABASE_NAME = os.getenv(
    "MONGODB_DATABASE",
    "medical_audio_rag"
)


# --------------------------------------------------
# 2. Validate configuration
# --------------------------------------------------

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI is missing in the .env file"
    )


# --------------------------------------------------
# 3. Create MongoDB client
# --------------------------------------------------

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

db = client[DATABASE_NAME]


# --------------------------------------------------
# 4. Define MongoDB collections
# --------------------------------------------------

# Stores uploaded audio file information
audio_collection = db["audio_records"]

# Stores audio chunks and embeddings for RAG
medical_chunks_collection = db["medical_chunks"]

# Stores speech-to-text transcription results
transcriptions_collection = db["transcriptions"]

# Stores Indian medicine catalogue records
medicines_collection = db["medicines_catalogue"]


# --------------------------------------------------
# 5. Connect and test MongoDB
# --------------------------------------------------

def test_mongodb_connection():

    try:
        client.admin.command("ping")

        print("MongoDB Atlas connected successfully!")
        print("Database:", db.name)

        required_collections = [
            "audio_records",
            "medical_chunks",
            "transcriptions",
            "medicines_catalogue"
        ]

        existing_collections = db.list_collection_names()

        for collection_name in required_collections:

            if collection_name not in existing_collections:

                db.create_collection(collection_name)

                print(
                    f"Created collection: {collection_name}"
                )

            else:

                print(
                    f"Collection already exists: {collection_name}"
                )

        return True

    except PyMongoError as error:

        print(
            "MongoDB connection failed:",
            str(error)
        )

        return False


# --------------------------------------------------
# 6. Reusable database connection
# --------------------------------------------------

def get_database():

    try:
        client.admin.command("ping")

        return db

    except PyMongoError as error:

        print(
            "Unable to access MongoDB:",
            str(error)
        )

        raise


# --------------------------------------------------
# 7. Get collection statistics
# --------------------------------------------------

def get_database_statistics():

    try:

        return {
            "database": db.name,
            "audio_records": (
                audio_collection.count_documents({})
            ),
            "medical_chunks": (
                medical_chunks_collection.count_documents({})
            ),
            "transcriptions": (
                transcriptions_collection.count_documents({})
            ),
            "medicines_catalogue": (
                medicines_collection.count_documents({})
            )
        }

    except PyMongoError as error:

        print(
            "Unable to retrieve database statistics:",
            str(error)
        )

        raise


# --------------------------------------------------
# 8. Search medicines by name or composition
# --------------------------------------------------

def search_medicines(medicine_name, limit=5):

    import re

    if not medicine_name or not medicine_name.strip():
        return []

    if limit < 1:
        return []

    pattern = re.escape(medicine_name.strip())

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
# 9. Close MongoDB connection
# --------------------------------------------------

def close_mongodb_connection():

    client.close()

    print("MongoDB connection closed.")


# --------------------------------------------------
# 10. Run MongoDB connection test
# --------------------------------------------------

if __name__ == "__main__":

    try:

        connected = test_mongodb_connection()

        if connected:

            statistics = get_database_statistics()

            print("\n================================")
            print("MEDICAL AUDIO DATABASE REPORT")
            print("================================")

            for collection_name, count in statistics.items():

                print(
                    f"{collection_name}: {count}"
                )

            print("================================")

    finally:

        close_mongodb_connection()
