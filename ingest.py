import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# Load environment variables
load_dotenv()

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "legal_documents"

def get_qdrant_client():
    """Returns a Qdrant client pointing to either a remote server URL or local storage."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_url.strip():
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None
        )
    return QdrantClient(path=QDRANT_PATH)

def initialize_collection(client: QdrantClient, vector_size: int = 384):
    """Initializes the vector collection if it doesn't already exist."""
    # Check if collection exists
    collections = client.get_collections()
    exist = any(c.name == COLLECTION_NAME for c in collections.collections)
    
    if not exist:
        print(f"Creating collection '{COLLECTION_NAME}' with vector size {vector_size}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def extract_text_from_pdf(pdf_path: str):
    """Extracts text page-by-page from a PDF, preserving metadata."""
    reader = PdfReader(pdf_path)
    pages_data = []
    file_name = Path(pdf_path).name
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "text": text,
                "metadata": {
                    "source": file_name,
                    "page": idx + 1
                }
            })
    return pages_data

def ingest_pdf_file(client: QdrantClient, embeddings_model: HuggingFaceEmbeddings, pdf_path: str):
    """Chunks, embeds, and stores a PDF in Qdrant."""
    print(f"Parsing {pdf_path}...")
    pages_data = extract_text_from_pdf(pdf_path)
    if not pages_data:
        print(f"No text extracted from {pdf_path}. Skipping.")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for page in pages_data:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": page["metadata"]
            })
            
    print(f"Generated {len(chunks)} chunks from {pdf_path}. Embedding and upserting...")
    
    # Generate embeddings and upload in batches
    points = []
    batch_size = 32
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]
        
        # Get embeddings from local HuggingFace model
        embeddings = embeddings_model.embed_documents(texts)
        
        # Prepare Qdrant PointStructs
        for j, emb in enumerate(embeddings):
            chunk_data = batch[j]
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "text": chunk_data["text"],
                    "source": chunk_data["metadata"]["source"],
                    "page": chunk_data["metadata"]["page"]
                }
            ))
            
    # Upsert points
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Successfully ingested {len(points)} points from {pdf_path} into Qdrant.")

def main():
    # Make sure documents directory exists
    os.makedirs("documents", exist_ok=True)
    
    # Initialize Qdrant Client
    client = get_qdrant_client()
    
    # Initialize HuggingFace Embeddings (runs locally, no API needed)
    print(f"Loading HuggingFace embeddings model '{EMBEDDING_MODEL}'...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    # Pre-test embedding size dynamically
    try:
        test_emb = embeddings_model.embed_query("test query")
        vector_size = len(test_emb)
        print(f"Successfully fetched test embedding. Dimension: {vector_size}")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        print("Please check that the model name is correct and sentence-transformers is installed.")
        return
        
    initialize_collection(client, vector_size)
    
    # Find all PDFs in documents/
    pdf_files = glob.glob("documents/*.pdf")
    if not pdf_files:
        print("\nNo PDF files found in the 'documents/' folder.")
        print("Please place legal PDF documents in the 'documents/' folder and rerun this script.")
        return
        
    for pdf in pdf_files:
        ingest_pdf_file(client, embeddings_model, pdf)
        
    print("\nIngestion completed!")

if __name__ == "__main__":
    main()