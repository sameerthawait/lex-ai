from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")
# Perform a simple query to see if it works
try:
    results = client.query_points(
        collection_name="legal_documents",
        query=[0.0] * 384,  # dummy vector of size 384
        limit=1
    )
    print("Success! query_points works.")
    print("Results type:", type(results))
    print("Results:", results)
    print("Points in results:")
    for point in results.points:
        print("Point ID:", point.id)
        print("Point payload:", point.payload)
except Exception as e:
    print("Error calling query_points:", e)

