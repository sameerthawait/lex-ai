import os
import sys
import json
from dotenv import load_dotenv

# Ensure parent directory is in Python path for root imports and configure UTF-8 output
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import our retrieval engine
from rag_engine import query_rag, invoke_llm_with_retry, get_llm

# Load environment variables
load_dotenv()

GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_API_BASE = os.getenv("GLM_API_BASE", "https://integrate.api.nvidia.com/v1")
GLM_MODEL = os.getenv("GLM_MODEL", "meta/llama-3.3-70b-instruct")

# Helper to robustly extract and parse JSON from LLM outputs
def extract_json_from_response(content: str) -> dict:
    content = content.strip()
    
    # Try looking for a JSON block inside markdown markers
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
        
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        json_str = content[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse text as JSON. Raw text was: {content[:200]}") from e

# 1. State Definition
class LegalState(TypedDict):
    user_query: str
    uploaded_docs: list
    legal_domain: str          # tenant_rights | employment | contract | etc
    jurisdiction: str
    retrieved_cases: list
    document_analysis: dict
    final_response: str
    cost_usd: float
    agent_trace: Annotated[list, operator.add]  # full decision log
    
    # Custom classifier outputs
    secondary_domain: str
    confidence: float
    key_legal_concepts: list
    likely_jurisdiction_matters: bool
    
    # Multi-dimensional risk outputs
    risk_score: int            # overall_risk_score (1-10)
    risk_dimensions: dict      # legal_risk, financial_risk, time_sensitivity, complexity
    immediate_actions: list
    deadline_alerts: list
    lawyer_recommended: bool
    lawyer_urgency: str        # immediately | within_week | optional
    risk_flags: list           # mapping for backward compatibility/tracing

# 2. Prompts
CLASSIFIER_PROMPT = """
Classify this legal query into the most specific domain possible.

Query: {query}

Domains:
- tenant_rights (eviction, deposits, repairs, lease)
- employment (wrongful termination, discrimination, wages, FMLA)
- contract (breach, non-compete, service agreements)
- small_claims (disputes under $10K)
- family (divorce, custody, child support)
- criminal (charges, rights, expungement)
- consumer (debt collection, fraud, returns)
- immigration (visas, DACA, citizenship)

Return JSON:
{{
  "primary_domain": "tenant_rights",
  "secondary_domain": "none",
  "confidence": 0.95,
  "key_legal_concepts": ["eviction notice", "tenant repairs"],
  "likely_jurisdiction_matters": true
}}
"""

DOCUMENT_AGENT_PROMPT = """
You are a contract analyst. Analyze this legal document and find:

1. RED FLAGS — clauses that harm the user (mark severity 1-10)
2. MISSING PROTECTIONS — standard clauses that are absent
3. UNUSUAL TERMS — non-standard language the user should question
4. KEY DATES & DEADLINES — anything time-sensitive
5. FINANCIAL OBLIGATIONS — all money-related clauses

Document text:
{document_text}

User's concern:
{user_query}

Return structured JSON with each category.
Be specific — quote the exact problematic clause, then explain why it's a problem in plain English.
"""

RISK_AGENT_PROMPT = """
Based on the legal research and document analysis below, assess the user's risk.

Legal research findings: {research_findings}
Document analysis: {document_analysis}
User situation: {user_query}

Score each dimension 1-10 (10 = most urgent):
- legal_risk: likelihood of losing if this goes to court
- financial_risk: potential financial exposure
- time_sensitivity: how urgently they need to act
- complexity: how much they need a real lawyer

Return JSON:
{{
  "overall_risk_score": 1-10,
  "dimensions": {{
    "legal_risk": 1-10,
    "financial_risk": 1-10,
    "time_sensitivity": 1-10,
    "complexity": 1-10
  }},
  "immediate_actions": ["do X within 48 hours", ...],
  "deadline_alerts": ["statute of limitations expires...", ...],
  "lawyer_recommended": true/false,
  "lawyer_urgency": "immediately|within_week|optional"
}}
"""

SAFETY_AGENT_PROMPT = """
Review this legal response before delivery. Your job:

1. Add appropriate jurisdiction disclaimers
2. Flag if any advice could be construed as practicing law
3. Ensure we're informing, not advising
4. Add relevant hotlines/free legal resources for this domain

Domain: {domain}
Jurisdiction: {jurisdiction}
Risk level: {risk_score}/10

Response content to review:
{response_content}

Free resources to include by domain:
- tenant: local tenant unions, legal aid societies
- employment: EEOC (1-800-669-4000), state labor boards  
- criminal: public defender offices, innocence projects
- family: family law facilitators (free at most courthouses)

Always end with:
"This is general legal information, not legal advice.
For your specific situation, consult a licensed attorney in {jurisdiction}."
"""

# 3. Graph Nodes
def orchestrator_node(state: LegalState) -> Dict[str, Any]:
    """Lightweight orchestrator node that registers the user request and sets up trace."""
    print(f"[Orchestrator] Starting workflow for query: '{state['user_query']}'")
    return {
        "agent_trace": [f"Orchestration started for user query."]
    }

def document_analysis_node(state: LegalState) -> Dict[str, Any]:
    """Analyzes lease agreements or contracts if uploaded by the user."""
    docs = state.get("uploaded_docs", [])
    if not docs:
        return {
            "document_analysis": {},
            "agent_trace": ["Document Analysis skipped: No uploaded documents found."]
        }
        
    print("[Agent Document Analyst] Analyzing contract/lease document...")
    doc_text = "\n\n".join([doc.get("text", "") for doc in docs if doc.get("text")])
    if not doc_text.strip():
        return {
            "document_analysis": {},
            "agent_trace": ["Document Analysis skipped: Uploaded documents contain no text content."]
        }
        
    llm = get_llm(temperature=0.0)
    
    formatted_prompt = DOCUMENT_AGENT_PROMPT.format(
        document_text=doc_text,
        user_query=state["user_query"]
    )
    messages = [SystemMessage(content="You are a JSON document analyst."), HumanMessage(content=formatted_prompt)]
    
    try:
        response = invoke_llm_with_retry(llm, messages)
        analysis_data = extract_json_from_response(response.content)
        
        # Count red flags dynamically across keys safely
        red_flags_list = analysis_data.get("RED FLAGS", [])
        red_flags_count = len(red_flags_list) if isinstance(red_flags_list, list) else 0
        
        return {
            "document_analysis": analysis_data,
            "agent_trace": [f"Document Analysis complete. Detected {red_flags_count} RED FLAGS in uploaded document."]
        }
    except Exception as e:
        print(f"[Agent Document Analyst] Document analysis failed: {e}")
        return {
            "document_analysis": {},
            "agent_trace": [f"Document Analysis failed during processing: {e}"]
        }

def query_courtlistener(query_text: str, top_k: int = 2) -> list:
    """Queries the CourtListener search API for opinions matching the query_text."""
    import requests
    api_key = os.getenv("COURTLISTENER_API_KEY")
    if not api_key or not api_key.strip():
        print("[Agent Retrieval Warning] COURTLISTENER_API_KEY not configured in .env. Skipping external search.")
        return []

    url = "https://www.courtlistener.com/api/rest/v4/search/"
    headers = {
        "Authorization": f"Token {api_key}"
    }
    params = {
        "q": query_text,
        "type": "o" # opinions only
    }
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            
            cl_sources = []
            for item in results[:top_k]:
                case_name = item.get("caseName") or item.get("caseNameFull") or "Unknown Case"
                court = item.get("court") or "Unknown Court"
                date_filed = item.get("dateFiled") or "Unknown Date"
                citations = item.get("citation") or []
                citation_str = citations[0] if citations else "No Citation"
                
                # Fetch snippet from the lead opinion or the syllabus
                snippet = ""
                opinions = item.get("opinions", [])
                if opinions and isinstance(opinions, list):
                    snippet = opinions[0].get("snippet") or opinions[0].get("text") or ""
                if not snippet:
                    snippet = item.get("snippet") or item.get("syllabus") or ""
                
                if snippet:
                    source_name = f"CourtListener: {case_name} ({court}, {date_filed}) [Cites: {citation_str}]"
                    cl_sources.append({
                        "text": snippet[:1000],  # cap length to keep prompt within bounds
                        "source": source_name,
                        "page": "1"
                    })
            return cl_sources
        else:
            print(f"[Agent Retrieval Warning] CourtListener API returned status {resp.status_code}: {resp.text}")
            return []
    except Exception as e:
        print(f"[Agent Retrieval Warning] CourtListener search failed: {e}")
        return []

def retrieval_node(state: LegalState) -> Dict[str, Any]:
    """Retrieves relevant local guidelines or case law."""
    print("[Agent Retrieval] Searching vector store...")
    rag_result = query_rag(state["user_query"], top_k=2)
    
    cases_extracted = [
        {
            "text": src["text"],
            "source": src["source"],
            "page": str(src["page"])
        }
        for src in rag_result.get("sources", [])
    ]
    
    # Query CourtListener API
    print("[Agent Retrieval] Searching CourtListener case law database...")
    cl_cases = query_courtlistener(state["user_query"], top_k=2)
    
    all_cases = cases_extracted + cl_cases
    
    return {
        "retrieved_cases": all_cases,
        "agent_trace": [f"Retrieved {len(cases_extracted)} local RAG sources and {len(cl_cases)} CourtListener case laws."]
    }

def risk_analysis_node(state: LegalState) -> Dict[str, Any]:
    """Evaluates multi-dimensional risk metrics using RISK_AGENT_PROMPT."""
    print("[Agent Risk Assessor] Conducting multi-dimensional risk assessment...")
    llm = get_llm(temperature=0.0)
    
    research_str = "\n".join([f"- {src['source']}: {src['text'][:150]}..." for src in state["retrieved_cases"]])
    doc_analysis_str = json.dumps(state["document_analysis"], indent=2)
    
    formatted_prompt = RISK_AGENT_PROMPT.format(
        research_findings=research_str or "No research findings retrieved.",
        document_analysis=doc_analysis_str or "No document analysis available.",
        user_query=state["user_query"]
    )
    
    messages = [SystemMessage(content="You are a JSON risk assessment agent."), HumanMessage(content=formatted_prompt)]
    
    try:
        response = invoke_llm_with_retry(llm, messages)
        data = extract_json_from_response(response.content)
        
        overall = data.get("overall_risk_score", 1)
        dimensions = data.get("dimensions", {})
        actions = data.get("immediate_actions", [])
        deadlines = data.get("deadline_alerts", [])
        rec_lawyer = data.get("lawyer_recommended", False)
        urgency = data.get("lawyer_urgency", "optional")
        
        trace_msg = (
            f"Risk analysis complete: Overall Score={overall}/10, "
            f"Lawyer Recommended={rec_lawyer} (Urgency: {urgency})"
        )
        
        return {
            "risk_score": overall,
            "risk_dimensions": dimensions,
            "immediate_actions": actions,
            "deadline_alerts": deadlines,
            "lawyer_recommended": rec_lawyer,
            "lawyer_urgency": urgency,
            "risk_flags": deadlines,  # backwards compatibility mapping
            "agent_trace": [trace_msg]
        }
    except Exception as e:
        print(f"[Agent Risk Assessor] Risk assessment failed: {e}")
        return {
            "risk_score": 1,
            "risk_dimensions": {},
            "immediate_actions": [],
            "deadline_alerts": [],
            "lawyer_recommended": False,
            "lawyer_urgency": "optional",
            "risk_flags": [],
            "agent_trace": [f"Risk scoring failed: {e}"]
        }

def response_generator_node(state: LegalState) -> Dict[str, Any]:
    """Generates draft legal response incorporating contexts."""
    print("[Agent Generator] Generating initial legal answer...")
    
    # Extract any CourtListener sources from the retrieved cases
    cl_sources = [
        src for src in state.get("retrieved_cases", [])
        if "CourtListener" in src.get("source", "")
    ]
    
    # Run the query generator with the merged external sources
    rag_result = query_rag(state["user_query"], external_sources=cl_sources)
    final_answer = rag_result["answer"]
    
    return {
        "final_response": final_answer,
        "agent_trace": ["Generated draft response based on hybrid local RAG & CourtListener search."]
    }

def safety_review_node(state: LegalState) -> Dict[str, Any]:
    """Performs final safety review and disclaimer verification using SAFETY_AGENT_PROMPT."""
    print("[Agent Safety Reviewer] Reviewing final response for compliance...")
    llm = get_llm(temperature=0.0)
    
    formatted_prompt = SAFETY_AGENT_PROMPT.format(
        domain=state["legal_domain"],
        jurisdiction=state["jurisdiction"],
        risk_score=state["risk_score"],
        response_content=state["final_response"]
    )
    
    messages = [HumanMessage(content=formatted_prompt)]
    
    try:
        response = invoke_llm_with_retry(llm, messages)
        reviewed_answer = response.content.strip()
        
        # Programmatically guarantee the legal disclaimer remains at the very end
        disclaimer_suffix = f"\n\n---\n*Disclaimer: This is general legal information, not legal advice. For your specific situation, consult a licensed attorney in {state['jurisdiction']}.*"
        if "general legal information" not in reviewed_answer:
            reviewed_answer = reviewed_answer + disclaimer_suffix
            
        return {
            "final_response": reviewed_answer,
            "agent_trace": ["Safety compliance review completed and disclaimers applied."]
        }
    except Exception as e:
        print(f"[Agent Safety Reviewer] Safety review node failed: {e}")
        # Return fallback with disclaimer attached
        fallback_answer = state["final_response"]
        disclaimer_suffix = f"\n\n---\n*Disclaimer: This is general legal information, not legal advice. For your specific situation, consult a licensed attorney in {state['jurisdiction']}.*"
        if "general legal information" not in fallback_answer:
            fallback_answer = fallback_answer.strip() + disclaimer_suffix
        return {
            "final_response": fallback_answer,
            "agent_trace": [f"Safety review node failed (Fallback response applied): {e}"]
        }

# 4. Build LangGraph Workflow
def build_orchestrator():
    builder = StateGraph(LegalState)
    
    # Add nodes
    builder.add_node("orchestrate", orchestrator_node)
    builder.add_node("document_analysis", document_analysis_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("risk_analysis", risk_analysis_node)
    builder.add_node("generate", response_generator_node)
    builder.add_node("safety_review", safety_review_node)
    
    # Set entry point
    builder.set_entry_point("orchestrate")
    
    # Define execution graph sequence
    builder.add_edge("orchestrate", "document_analysis")
    builder.add_edge("document_analysis", "retrieve")
    builder.add_edge("retrieve", "risk_analysis")
    builder.add_edge("risk_analysis", "generate")
    builder.add_edge("generate", "safety_review")
    builder.add_edge("safety_review", END)
    
    return builder.compile()

def analyze_legal_query(query: str, uploaded_docs: list = None):
    """Entry point to run the compiled LangGraph workflow."""
    graph = build_orchestrator()
    
    initial_state = {
        "user_query": query,
        "uploaded_docs": uploaded_docs or [],
        "legal_domain": "other",
        "jurisdiction": "Metropolis",
        "retrieved_cases": [],
        "document_analysis": {},
        "risk_score": 1,
        "final_response": "",
        "cost_usd": 0.0,
        "agent_trace": [],
        
        # Classifications defaults
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
    # Test LangGraph flow execution
    test_query = "My landlord served me an eviction notice because of repairs I requested."
    test_docs = [
        {
            "text": "LEASE COVENANTS: Tenant agrees to pay all repairs under $500. Landlord reserves the right to terminate lease with 24 hours notice for any reason."
        }
    ]
    
    print(f"Executing LangGraph flow for query: '{test_query}'\n")
    final_state = analyze_legal_query(test_query, uploaded_docs=test_docs)
    
    print("\n--- DECISION LOGS / AGENT TRACE ---")
    for log in final_state["agent_trace"]:
        print(f"  * {log}")
        
    print("\n--- DETECTED DOCUMENT ANALYSIS ---")
    print(json.dumps(final_state["document_analysis"], indent=2))
    
    print("\n--- DETECTED RISK ANALYSIS METRICS ---")
    print(f"Overall Risk Score: {final_state['risk_score']}/10")
    print(f"Risk Dimensions: {json.dumps(final_state['risk_dimensions'], indent=2)}")
    print(f"Immediate Actions Needed: {final_state['immediate_actions']}")
    print(f"Deadline Alerts: {final_state['deadline_alerts']}")
    print(f"Lawyer Recommended: {final_state['lawyer_recommended']} (Urgency: {final_state['lawyer_urgency']})")
        
    print("\n--- FINAL CHAT RESPONSE ---")
    print(final_state["final_response"])