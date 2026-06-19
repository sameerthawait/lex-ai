import os
import sys
from dotenv import load_dotenv

# Ensure parent directory is in Python path for root imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
# Import state and agent nodes
from agents.orchestrator import (
    LegalState,
    orchestrator_node as orchestrator_agent,
    document_analysis_node as document_agent,
    retrieval_node as research_agent,
    risk_analysis_node as risk_agent,
    response_generator_node as synthesis_agent,
    safety_review_node as safety_agent
)

# Load environment variables
load_dotenv()

# Define classifier node separately to match the Phase 4 architectural nodes
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from agents.orchestrator import CLASSIFIER_PROMPT, extract_json_from_response
from rag_engine import invoke_llm_with_retry, get_llm

GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_API_BASE = os.getenv("GLM_API_BASE", "https://integrate.api.nvidia.com/v1")
GLM_MODEL = os.getenv("GLM_MODEL", "meta/llama-3.3-70b-instruct")

def classifier_agent(state: LegalState):
    """Invokes the domain classification prompt on the LLM."""
    print("[Agent Classifier] Invoking classification model...")
    llm = get_llm(temperature=0.0)
    formatted_prompt = CLASSIFIER_PROMPT.format(query=state["user_query"])
    messages = [SystemMessage(content="You are a JSON assistant."), HumanMessage(content=formatted_prompt)]
    
    try:
        response = invoke_llm_with_retry(llm, messages)
        data = extract_json_from_response(response.content)
        
        primary = data.get("primary_domain", "other")
        secondary = data.get("secondary_domain", "none")
        conf = data.get("confidence", 0.0)
        concepts = data.get("key_legal_concepts", [])
        juris_matters = data.get("likely_jurisdiction_matters", False)
        
        return {
            "legal_domain": primary,
            "secondary_domain": secondary,
            "confidence": conf,
            "key_legal_concepts": concepts,
            "likely_jurisdiction_matters": juris_matters,
            "agent_trace": [f"Classifier complete: Primary={primary} (Conf: {conf})"]
        }
    except Exception as e:
        return {
            "legal_domain": "other",
            "secondary_domain": "none",
            "confidence": 0.0,
            "key_legal_concepts": [],
            "likely_jurisdiction_matters": False,
            "agent_trace": [f"Classifier failed: {e}"]
        }

# Define conditional routing helper
def route_after_orchestration(state: LegalState) -> str:
    """Routes to document analysis if documents are present, else routes to classification."""
    docs = state.get("uploaded_docs", [])
    if docs:
        print("[Router] Uploaded documents detected. Routing to 'analyze_docs'.")
        return "analyze_docs"
    print("[Router] No documents uploaded. Routing to 'classify'.")
    return "classify"

def build_legal_graph():
    """Compiles the multi-agent LangGraph workflow with conditional routing."""
    graph = StateGraph(LegalState)
    
    # Add nodes
    graph.add_node("orchestrate", orchestrator_agent)
    graph.add_node("classify", classifier_agent)
    graph.add_node("research", research_agent)
    graph.add_node("analyze_docs", document_agent)
    graph.add_node("assess_risk", risk_agent)
    graph.add_node("synthesize", synthesis_agent)
    graph.add_node("safety_check", safety_agent)
    
    # Entry
    graph.set_entry_point("orchestrate")
    
    # Conditional routing
    graph.add_conditional_edges(
        "orchestrate",
        route_after_orchestration,  # returns next node name
        {
            "classify": "classify",
            "analyze_docs": "analyze_docs",
        }
    )
    
    # Static edges
    graph.add_edge("classify", "research")
    graph.add_edge("analyze_docs", "research")
    graph.add_edge("research", "assess_risk")
    graph.add_edge("assess_risk", "synthesize")
    graph.add_edge("synthesize", "safety_check")
    graph.add_edge("safety_check", END)
    
    return graph.compile()

def analyze_legal_query(query: str, uploaded_docs: list = None) -> dict:
    """Entry point to run the compiled LangGraph workflow with conditional routing."""
    graph = build_legal_graph()
    
    initial_state = {
        "user_query": query,
        "uploaded_docs": uploaded_docs or [],
        "legal_domain": "other",
        "jurisdiction": "California",
        "retrieved_cases": [],
        "document_analysis": {},
        "risk_score": 1,
        "final_response": "",
        "cost_usd": 0.0,
        "agent_trace": [],
        
        # Classifier defaults
        "secondary_domain": "none",
        "confidence": 0.0,
        "key_legal_concepts": [],
        "likely_jurisdiction_matters": False,
        
        # Risk defaults
        "risk_dimensions": {},
        "immediate_actions": [],
        "deadline_alerts": [],
        "lawyer_recommended": False,
        "lawyer_urgency": "optional",
        "risk_flags": []
    }
    
    return graph.invoke(initial_state)

if __name__ == "__main__":
    # Test compiled legal graph workflow
    print("Compiling and running legal graph workflow...")
    workflow = build_legal_graph()
    
    test_query = "My landlord served me an eviction notice because of repairs I requested."
    test_docs = [
        {
            "text": "LEASE COVENANTS: Tenant agrees to pay all repairs under $500. Landlord reserves the right to terminate lease with 24 hours notice for any reason."
        }
    ]
    
    initial_state = {
        "user_query": test_query,
        "uploaded_docs": test_docs,
        "legal_domain": "other",
        "jurisdiction": "Metropolis",
        "retrieved_cases": [],
        "document_analysis": {},
        "final_response": "",
        "cost_usd": 0.0,
        "agent_trace": [],
        
        # Classifier defaults
        "secondary_domain": "none",
        "confidence": 0.0,
        "key_legal_concepts": [],
        "likely_jurisdiction_matters": False,
        
        # Risk defaults
        "risk_score": 1,
        "risk_dimensions": {},
        "immediate_actions": [],
        "deadline_alerts": [],
        "lawyer_recommended": False,
        "lawyer_urgency": "optional",
        "risk_flags": []
    }
    
    print("\nExecuting graph invoke...")
    final_state = workflow.invoke(initial_state)
    
    print("\n--- DECISION LOGS / AGENT TRACE ---")
    for log in final_state["agent_trace"]:
        print(f"  * {log}")
        
    print("\n--- DETECTED RISK ASSESSMENT ---")
    print(f"Overall Risk Score: {final_state['risk_score']}/10")
    print(f"Lawyer Recommended: {final_state['lawyer_recommended']} (Urgency: {final_state['lawyer_urgency']})")
    
    print("\n--- FINAL CHAT RESPONSE ---")
    print(final_state["final_response"])