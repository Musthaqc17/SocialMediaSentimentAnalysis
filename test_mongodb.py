from pymongo import MongoClient
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

uri = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
)

print("Testing MongoDB connection...")

try:
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    print("SUCCESS: MongoDB connected!")

    db = client["socialpulse"]
    print("Database name:", db.name)

    # Insert a temporary test document
    collection = db["connection_tests"]

    result = collection.insert_one({
        "message": "MongoDB connection successful"
    })

    print("Test document inserted:", result.inserted_id)

    client.close()

except Exception as error:
    print("MongoDB connection failed:")
    print(error)
    traceback.print_exc()
