from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")
print("Type of client:", type(client))
print("Available attributes:")
for attr in sorted(dir(client)):
    if not attr.startswith("_"):
        print(f" - {attr}")
