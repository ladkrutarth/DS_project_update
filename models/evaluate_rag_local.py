from models.rag_engine_local import RAGEngineLocal
from scripts.track_metrics import MetricTracker
from utils.logger import logger

def evaluate_rag():
    tracker = MetricTracker("rag_evaluation")
    logger.info("Initializing Local RAG Engine...")
    rag = RAGEngineLocal()
    
    test_cases = [
        {
            "query": "What are common credit card fraud trends?",
            "keywords": ["fraud", "credit", "card", "complaint"]
        },
        {
            "query": "How to dispute a charge?",
            "keywords": ["dispute", "charge", "billing", "error"]
        },
        {
            "query": "Identity theft protection",
            "keywords": ["identity", "theft", "protection", "security"]
        }
    ]
    
    total_score = 0
    logger.info("--- RAG Retrieval Evaluation ---")
    
    for case in test_cases:
        query = case["query"]
        logger.info(f"Query: '{query}'")
        
        results = rag.query(query, n_results=3)
        
        hit_count = 0
        for i, res in enumerate(results):
            doc = res["text"]
            is_hit = any(kw.lower() in doc.lower() for kw in case["keywords"])
            if is_hit:
                hit_count += 1
            
            snippet = doc[:100].replace('\n', ' ')
            status = "✅ RELEVANT" if is_hit else "❌ IRRELEVANT"
            print(f"  [{i+1}] {status} | Snippet: {snippet}...")
        
        precision = hit_count / 3
        total_score += precision
        print(f"Precision@3: {precision:.2f}")
        
    avg_precision = total_score / len(test_cases)
    logger.info(f"Average Precision@3: {avg_precision:.2f}")
    tracker.log_metric("avg_precision_at_3", avg_precision)
    tracker.save_artifact("test_cases.json", test_cases)
    tracker.save()
    
    return avg_precision

if __name__ == "__main__":
    evaluate_rag()
