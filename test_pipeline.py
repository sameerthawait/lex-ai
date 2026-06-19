from rag.pipeline import build_legal_index

def main():
    print("Testing LlamaIndex pipeline...")
    
    # Mock document list
    documents = [
        {
            "text": "Metropolis Civil Code Section 440-A protects tenants from landlord retaliation.",
            "source": "Metropolis Civil Code",
            "jurisdiction": "Metropolis",
            "type": "statute",
            "date": "2026-01-01"
        },
        {
            "text": "In Smith v. Jones, the court held that security deposits must be returned within 21 days.",
            "source": "Smith v. Jones (123 F.3d 456)",
            "jurisdiction": "Metropolis Appeal Court",
            "type": "case_law",
            "date": "2024-05-15"
        }
    ]
    
    try:
        index = build_legal_index(documents)
        print("Successfully built LlamaIndex vector store!")
        
        # Test a query engine execution
        print("Executing test query...")
        # Since build_legal_index closed the client, let's query the index or use a query engine
        query_engine = index.as_query_engine()
        response = query_engine.query("What was held in Smith v. Jones?")
        print("\nResponse:")
        print(response)
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")

if __name__ == "__main__":
    main()
