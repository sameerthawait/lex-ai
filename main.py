import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Import our LangGraph workflow and ingestion modules
from graph.legal_graph import build_legal_graph, analyze_legal_query
from ingest import get_qdrant_client, initialize_collection, ingest_pdf_file
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(
    title="Legal Aid AI System",
    description="AI-Powered Legal Assistance using Nvidia Llama 3.3, HuggingFace Embeddings, and Qdrant"
)

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str

class SourceInfo(BaseModel):
    source: str
    page: str
    text: str

class RiskDimensions(BaseModel):
    legal_risk: int = 1
    financial_risk: int = 1
    time_sensitivity: int = 1
    complexity: int = 1

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    legal_domain: str
    secondary_domain: str
    confidence: float
    key_legal_concepts: List[str]
    likely_jurisdiction_matters: bool
    risk_score: int
    risk_dimensions: RiskDimensions
    immediate_actions: List[str]
    deadline_alerts: List[str]
    lawyer_recommended: bool
    lawyer_urgency: str
    agent_trace: List[str]

class DocumentInfo(BaseModel):
    filename: str
    size_kb: float
    status: str

# Ingestion Worker function for background tasks
def background_ingest(file_path: str):
    """Initializes embeddings and database client, then ingests the uploaded PDF."""
    client = None
    try:
        client = get_qdrant_client()
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # Test connection/dimension
        test_emb = embeddings_model.embed_query("test")
        vector_size = len(test_emb)
        
        initialize_collection(client, vector_size)
        ingest_pdf_file(client, embeddings_model, file_path)
    except Exception as e:
        print(f"Background ingestion failed for {file_path}: {e}")
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

# API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Check if there are documents in the documents directory to run document analysis
    uploaded_docs = []
    docs_dir = "documents"
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(docs_dir, filename)
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        uploaded_docs.append({
                            "filename": filename,
                            "text": text
                        })
                except Exception as e:
                    print(f"Error reading PDF {filename} for analysis: {e}")

    # Invoke the compiled LangGraph workflow
    output = analyze_legal_query(request.message, uploaded_docs=uploaded_docs)
    
    # Map outputs safely to ChatResponse
    sources = []
    for src in output.get("retrieved_cases", []):
        sources.append({
            "source": src.get("source", "Unknown"),
            "page": str(src.get("page", "Unknown")),
            "text": src.get("text", "")
        })
        
    return {
        "answer": output.get("final_response", ""),
        "sources": sources,
        "legal_domain": output.get("legal_domain", "other"),
        "secondary_domain": output.get("secondary_domain", "none"),
        "confidence": output.get("confidence", 0.0),
        "key_legal_concepts": output.get("key_legal_concepts", []),
        "likely_jurisdiction_matters": output.get("likely_jurisdiction_matters", False),
        "risk_score": output.get("risk_score", 1),
        "risk_dimensions": output.get("risk_dimensions", {
            "legal_risk": 1, "financial_risk": 1, "time_sensitivity": 1, "complexity": 1
        }),
        "immediate_actions": output.get("immediate_actions", []),
        "deadline_alerts": output.get("deadline_alerts", []),
        "lawyer_recommended": output.get("lawyer_recommended", False),
        "lawyer_urgency": output.get("lawyer_urgency", "optional"),
        "agent_trace": output.get("agent_trace", [])
    }

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    os.makedirs("documents", exist_ok=True)
    file_path = os.path.join("documents", file.filename)
    
    try:
        # Save file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Queue ingestion in background task to keep API responsive
        background_tasks.add_task(background_ingest, file_path)
        
        return {"message": f"File '{file.filename}' uploaded successfully. Ingestion started in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    docs_dir = "documents"
    os.makedirs(docs_dir, exist_ok=True)
    
    documents = []
    client = None
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        has_col = any(c.name == "legal_documents" for c in collections.collections)
    except Exception:
        has_col = False

    # Fetch all PDF files in folder
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(docs_dir, filename)
            size_kb = os.path.getsize(file_path) / 1024.0
            
            status = "Ingested"
            if not has_col:
                status = "Pending Ingestion"
            elif client:
                try:
                    from qdrant_client.http import models
                    count_res = client.count(
                        collection_name="legal_documents",
                        count_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=filename)
                                )
                            ]
                        )
                    )
                    if count_res.count == 0:
                        status = "Pending Ingestion"
                except Exception:
                    status = "Ingestion Status Unknown"
            else:
                status = "Ingestion Status Unknown"
                
            documents.append(DocumentInfo(
                filename=filename,
                size_kb=round(size_kb, 1),
                status=status
            ))
            
    if client:
        try:
            client.close()
        except Exception:
            pass
            
    return documents

# Serve Web Frontend (Fallback to index.html for SPA routing)
@app.get("/")
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Legal Aid Frontend is under construction. Backend is running successfully!"}

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI legal-aid-system backend on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
