from models.guard_agent_local import LocalGuardAgent
from scripts.track_metrics import MetricTracker
from utils.logger import logger

def evaluate_agent():
    tracker = MetricTracker("agent_evaluation")
    logger.info("Initializing GuardAgent...")
    agent = LocalGuardAgent()
    
    scenarios = [
        # ── User Queries (expect: get_user_risk_profile) ──
        {"query": "Investigate USER_123 for potential fraud.",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Does USER_456 have a high risk score?",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Show risk profile for USER_0",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Analyze risk for USER_789",
         "expected_tool": "get_user_risk_profile", "category": "User"},

        # ── Knowledge Queries (expect: query_rag) ──
        {"query": "What are the latest CFPB trends for credit card disputes?",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "Explain what 1h velocity means in fraud detection.",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "How to dispute a charge?",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "What is identity theft protection?",
         "expected_tool": "query_rag", "category": "Knowledge"},

        # ── System Queries (expect: get_high_risk_transactions) ──
        {"query": "Show the top high risk transactions in the system.",
         "expected_tool": "get_high_risk_transactions", "category": "System"},
        {"query": "What are the most dangerous transactions?",
         "expected_tool": "get_high_risk_transactions", "category": "System"},
    ]
    
    success_count = 0
    category_results = {"User": [0, 0], "Knowledge": [0, 0], "System": [0, 0]}
    
    
    logger.info(f"GuardAgent Evaluation Suite — {len(scenarios)} Scenarios")
    
    for i, scenario in enumerate(scenarios):
        query = scenario["query"]
        expected = scenario["expected_tool"]
        cat = scenario["category"]
        logger.info(f"[{i+1}/{len(scenarios)}] ({cat}) '{query}'")
        
        result = agent.analyze(query)
        actions = result.get("actions", [])
        
        # Check if the expected tool was called in any step
        found_tool = any(a["tool"] == expected for a in actions)
        category_results[cat][1] += 1  # total
        
        if found_tool:
            logger.info(f"  ✅ Correct Tool Called: {expected}")
            success_count += 1
            category_results[cat][0] += 1  # correct
        else:
            logger.error(f"  ❌ Expected '{expected}' — not called.")
            if actions:
                logger.warning(f"     Called instead: {[a['tool'] for a in actions]}")
            else:
                logger.warning(f"     No tools were called.")
    
    # ── Summary ──
    accuracy = success_count / len(scenarios)
    logger.info(f"Overall Accuracy: {success_count}/{len(scenarios)} = {accuracy*100:.0f}%")
    
    tracker.log_metric("overall_accuracy", accuracy)
    tracker.log_metric("category_breakdown", category_results)
    tracker.save_artifact("scenarios_analyzed.json", scenarios)
    tracker.save()
    
    return accuracy

if __name__ == "__main__":
    evaluate_agent()
