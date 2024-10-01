from pymongo import MongoClient


def connect_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["test_database"]
        collection = db["test_collection"]
        # Insert a test document
        test_doc = {"message": "MongoDB connection successful!"}
        collection.insert_one(test_doc)
        print("MongoDB connection and test document insertion successful.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    connect_mongodb()
