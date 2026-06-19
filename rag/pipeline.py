import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

QDRANT_PATH = os.getenv("QDRANT_PATH")
QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Configure LlamaIndex to use local HuggingFace embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print(f"Configuring LlamaIndex with local HuggingFace embeddings ({EMBEDDING_MODEL})...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

# Note: LLM is not configured here since we use Nvidia Llama 3.3 via LangChain in rag_engine.py
# LlamaIndex will use a stub LLM (MockLLM) for query engine operations
from llama_index.core.llms import MockLLM
Settings.llm = MockLLM()

# Qdrant client is initialized inside build_legal_index to prevent file locking and teardown errors

RAG_SYSTEM_PROMPT = """
You are a legal research assistant. Given retrieved legal documents,
extract ONLY what is directly relevant to the user's situation.

Rules:
- Cite every source with case name or statute number
- Note the jurisdiction and date of each source
- Flag if sources conflict with each other
- Never extrapolate beyond what the documents say
- Use plain English, define legal terms when used

Retrieved documents:
{context}
"""

def build_legal_index(documents: list[dict]):
    """LlamaIndex integration - builds or updates the vector store index."""
    # Initialize Qdrant Client dynamically based on serverless or URL configuration
    if QDRANT_PATH:
        print(f"Connecting LlamaIndex QdrantClient to local path: '{QDRANT_PATH}'")
        client = QdrantClient(path=QDRANT_PATH)
    else:
        url = QDRANT_URL or "http://localhost:6333"
        print(f"Connecting LlamaIndex QdrantClient to URL: '{url}'")
        client = QdrantClient(url=url)

    # Note: LlamaIndex QdrantVectorStore handles collection creation/connection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="legal_docs"
    )
    
    docs = [
        Document(
            text=doc["text"],
            metadata={
                "source": doc.get("source", "Unknown"),
                "jurisdiction": doc.get("jurisdiction", "all"),
                "doc_type": doc.get("type", "statute"),  # case_law | statute | regulation
                "date": doc.get("date", "Unknown")
            }
        )
        for doc in documents
    ]
    
    index = VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store
    )
    
    # Close client immediately if in serverless/local mode to prevent file locking
    if QDRANT_PATH:
        client.close()
        
    return index

def query_legal_index(index, query_text: str):
    """Queries the LlamaIndex using the customized system prompt and appends safety disclaimers."""
    from llama_index.core import PromptTemplate
    
    qa_prompt = PromptTemplate(RAG_SYSTEM_PROMPT)
    query_engine = index.as_query_engine(
        text_qa_template=qa_prompt
    )
    
    response = query_engine.query(query_text)
    answer = str(response)
    
    # Guarantee legal disclaimer is appended
    disclaimer_text = (
        "\n\n---\n"
        "*Disclaimer: The information provided above is for educational purposes only and does not constitute "
        "formal legal advice. For official legal representation or counsel, please contact a licensed attorney "
        "or a local legal aid organization.*"
    )
    if "Disclaimer:" not in answer:
        answer = answer.strip() + disclaimer_text
        
    return answer