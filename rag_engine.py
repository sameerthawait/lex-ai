import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_API_BASE = os.getenv("GLM_API_BASE", "https://integrate.api.nvidia.com/v1")
GLM_MODEL = os.getenv("GLM_MODEL", "meta/llama-3.3-70b-instruct")
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

_embeddings_model = None

def get_embeddings_model():
    """Returns the HuggingFace embeddings model (runs locally, no API needed)."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    return _embeddings_model

def get_llm(temperature: float = 0.1):
    """Returns the LLM via OpenAI-compatible API configured in .env."""
    # Check if there are generic LLM env vars, otherwise default to Nvidia settings
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE") or os.getenv("GLM_API_BASE", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("LLM_MODEL") or os.getenv("GLM_MODEL", "meta/llama-3.3-70b-instruct")
    
    client_kwargs = {
        "api_key": api_key,
        "base_url": api_base,
        "model": model,
        "temperature": temperature,
        "max_retries": 2,
        "timeout": 30.0,
    }
    
    # Configure custom parameters for Nvidia/DeepSeek endpoints
    if api_base and "nvidia.com" in api_base.lower():
        client_kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        
    return ChatOpenAI(**client_kwargs)



def invoke_llm_with_retry(llm, messages, max_retries=5, initial_delay=2.0, backoff_factor=2.0):
    """Invokes the LLM with exponential backoff retry on errors."""
    import time
    import random
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_msg = str(e)
            if attempt < max_retries:
                sleep_time = delay + random.uniform(0.1, 1.0)
                print(f"[LLM Retry] Error calling LLM: {err_msg[:120]}... Retrying in {sleep_time:.2f} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
                delay *= backoff_factor
            else:
                raise e

def query_rag(query_text: str, top_k: int = 3, external_sources: list = None):
    """Searches Qdrant local storage and answers the query using Nvidia Llama 3.3 with citations."""
    client = get_qdrant_client()
    
    # Initialize HuggingFace Embeddings for search query vectorization
    embeddings_model = get_embeddings_model()
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        exist = any(c.name == COLLECTION_NAME for c in collections.collections)
    except Exception:
        exist = False
    
    search_results = []
    if exist:
        try:
            # Vectorize query
            query_vector = embeddings_model.embed_query(query_text)
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            )
            search_results = response.points
        except Exception as e:
            print(f"Warning: Local RAG query failed: {e}")
    
    client.close()
        
    if not search_results and not external_sources:
        return {
            "answer": "I couldn't find any relevant legal documents in my database or external case law databases to help answer your question.",
            "sources": []
        }
        
    # Construct context from search results and extract sources
    context_chunks = []
    sources = []
    
    if search_results:
        for result in search_results:
            text = result.payload.get("text", "")
            source_file = result.payload.get("source", "Unknown Document")
            page = result.payload.get("page", "Unknown Page")
            
            context_chunks.append(f"Source Document: {source_file}\nPage: {page}\nContent:\n{text}\n---")
            
            source_info = {"source": source_file, "page": str(page), "text": text}
            if source_info not in sources:
                sources.append(source_info)
                
    if external_sources:
        for ext in external_sources:
            text = ext.get("text", "")
            source_name = ext.get("source", "External Source")
            page = ext.get("page", "1")
            
            context_chunks.append(f"Source Document (External Case Law): {source_name}\nContent:\n{text}\n---")
            
            source_info = {"source": source_name, "page": str(page), "text": text}
            if source_info not in sources:
                sources.append(source_info)
                
    context_str = "\n".join(context_chunks)
    
    # Setup prompt
    system_prompt = (
        "You are an expert Legal Aid Assistant designed to help self-representing individuals "
        "and low-income tenants understand their rights. Answer the user's query clearly, "
        "concisely, and structurally using only the provided legal context context below. "
        "If you do not know the answer or if the context does not contain enough information, "
        "state that clearly.\n\n"
        "Cite the document name and page number when referring to specific information in your response. "
        "Use markdown for structure (bullet points, bold text).\n\n"
        "Here is the context retrieved from legal documents:\n"
        f"{context_str}\n\n"
        "CRITICAL INSTRUCTION: You MUST append the following exact Legal Disclaimer at the very end of your response, "
        "separated by a horizontal line (---):\n"
        "*Disclaimer: The information provided above is for educational purposes only and does not constitute "
        "formal legal advice. For official legal representation or counsel, please contact a licensed attorney "
        "or a local legal aid organization.*"
    )
    
    # Initialize Nvidia Llama 3.3 LLM
    try:
        llm = get_llm(temperature=0.1)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        
        response = invoke_llm_with_retry(llm, messages)
        answer = response.content
        
        # Programmatically guarantee the legal disclaimer is appended
        disclaimer_text = (
            "\n\n---\n"
            "*Disclaimer: The information provided above is for educational purposes only and does not constitute "
            "formal legal advice. For official legal representation or counsel, please contact a licensed attorney "
            "or a local legal aid organization.*"
        )
        if "Disclaimer:" not in answer:
            answer = answer.strip() + disclaimer_text
            
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        return {
            "answer": f"Error communicating with Nvidia API: {e}. Please check your API key and network connection.",
            "sources": sources
        }

if __name__ == "__main__":
    # Test execution
    test_query = "What is the maximum security deposit for an unfurnished apartment?"
    print(f"Testing RAG search with query: '{test_query}'\n")
    result = query_rag(test_query)
    print("Answer:")
    print(result["answer"])
    print("\nSources Cited:")
    for idx, src in enumerate(result["sources"]):
        print(f"[{idx+1}] {src['source']} (Page {src['page']})")