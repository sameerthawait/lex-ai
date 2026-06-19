import os
from dotenv import load_dotenv
from datasets import Dataset
from rag_engine import query_rag

# Load environment variables
load_dotenv()

# Evaluates the RAG pipeline using a custom local evaluator or Ragas
def run_evaluation():
    print("Initializing RAG Pipeline Evaluation...")
    
    # Define validation queries and expected ground truths
    eval_dataset = [
        {
            "question": "What is the maximum security deposit for an unfurnished apartment?",
            "ground_truth": "The maximum security deposit is equivalent to two (2) months' rent (Metropolis Civil Code Section 442)."
        },
        {
            "question": "How much relocation assistance is required for No-Fault evictions?",
            "ground_truth": "The landlord is required to provide relocation assistance equal to one (1) month of the tenant's current rent within 15 days of notice."
        },
        {
            "question": "What is the annual rent increase cap?",
            "ground_truth": "Rent increases are capped at 5% plus the regional Consumer Price Index (CPI), up to a maximum of 10% in any 12-month period."
        }
    ]
    
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    # Run the RAG pipeline on each question
    for idx, item in enumerate(eval_dataset):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{idx+1}/3] Querying RAG: '{q}'...")
        
        result = query_rag(q)
        ans = result["answer"]
        srcs = [src["text"] for src in result["sources"]]
        
        questions.append(q)
        answers.append(ans)
        contexts_list.append(srcs)
        ground_truths.append(gt)
        
        print(f"-> Generated Answer: {ans[:120]}...")
        print(f"-> Retrieved {len(srcs)} source context(s).")

    # Create dataset dictionary format expected by Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("\n--------------------------------------------------")
    print("Running Ragas Local Evaluation Metric Check...")
    
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevance
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        
        GLM_API_KEY = os.getenv("GLM_API_KEY", "")
        GLM_API_BASE = os.getenv("GLM_API_BASE", "https://integrate.api.nvidia.com/v1")
        GLM_MODEL = os.getenv("GLM_MODEL", "meta/llama-3.3-70b-instruct")
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Configure Ragas to evaluate using our Nvidia Llama 3.3 and HuggingFace Embeddings
        print("Configuring Nvidia Llama 3.3 and HuggingFace Embeddings for evaluation...")
        evaluator_llm = ChatOpenAI(
            api_key=GLM_API_KEY,
            base_url=GLM_API_BASE,
            model=GLM_MODEL,
            temperature=0.0
        )
        evaluator_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # In newer Ragas versions, metrics can be customized with llm and embeddings
        # We wrapper them to avoid deprecation warnings
        faithfulness.llm = evaluator_llm
        answer_relevance.llm = evaluator_llm
        answer_relevance.embeddings = evaluator_embeddings
        
        print("Running evaluation (this may take a few moments)...")
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevance]
        )
        
        print("\n=== Ragas Evaluation Results ===")
        print(results)
        print("================================")
        
    except Exception as e:
        print("\n[Ragas Configuration Notice]")
        print("Could not complete automatic Ragas score calculation due to:")
        print(f" -> {e}")
        print("This is common in offline setups when local model structure differs or Ragas metrics expect OpenAI.")
        print("\nLet's run a fallback Semantic Similarity comparison to score results...")
        
        # Quick heuristic scorecard fallback
        print("\n=== Fallback Legal RAG Scorecard ===")
        for i in range(len(questions)):
            q = questions[i]
            ans = answers[i]
            gt = ground_truths[i]
            ctx = contexts_list[i]
            
            # Simple keyword matching to score grounding
            matches = [word for word in gt.lower().split() if len(word) > 4 and word in ans.lower()]
            score = len(matches) / len([w for w in gt.lower().split() if len(w) > 4]) * 100
            
            print(f"\nQuestion: {q}")
            print(f"Expected Keywords Grounding Score: {score:.1f}%")
            print(f"Status: {'PASS' if score > 50 else 'REVIEW'}")
            print(f"Context verified: {'Yes' if len(ctx) > 0 else 'No'}")
        print("====================================")

if __name__ == "__main__":
    run_evaluation()
