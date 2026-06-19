import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def main():
    print("Testing local Qdrant Client (Serverless Mode)...")
    
    # Define a local database path
    db_path = "./qdrant_data"
    
    try:
        # Initialize client with a local path (serverless)
        print(f"Initializing QdrantClient with local storage at '{db_path}'...")
        client = QdrantClient(path=db_path)
        
        # Test collection creation
        collection_name = "test_collection"
        print(f"Creating a test collection '{collection_name}'...")
        
        # Recreate collection (equivalent to delete + create)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        
        # Check collections
        collections = client.get_collections()
        print("Successfully created collection!")
        print("Current collections:", [c.name for c in collections.collections])
        
        print("\nQdrant serverless mode is working perfectly!")
        
    except Exception as e:
        print(f"\nFailed to initialize/use local Qdrant Client: {e}")

if __name__ == "__main__":
    main()
