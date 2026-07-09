"""
src/automl_agents/tools/eval.py

Automated Agent Evaluation framework verifying AutoML agent decisions against
known ground-truth rubrics on the synthetic dataset.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def run_agent_evals(state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs automated reasoning checks on the synthetic dataset run state.
    
    Verifies leakage detection, sensible imputation, and problem type inference.
    """
    logger.info("Running Agent Evaluation Rubric...")
    
    results = {}
    
    # 1. Target Leakage Detection check
    # Check if 'leaky_churn_copy' was successfully dropped
    prep_plan = state.get("prep_plan") or {}
    drop_cols = prep_plan.get("drop_cols") or []
    
    leakage_detected = "leaky_churn_copy" in drop_cols
    results["leakage_detection"] = {
        "pass": leakage_detected,
        "detail": "Successfully dropped target duplicate 'leaky_churn_copy'" if leakage_detected 
                  else "Failed to drop target duplicate 'leaky_churn_copy' (or no prep plan exists)."
    }
    
    # 2. Imputation check
    # In synthetic dataset, 'annual_income' and 'tenure_months' have nulls.
    # Check if imputation strategy is specified for these columns.
    impute_dict = prep_plan.get("impute") or {}
    imputation_ok = ("annual_income" in impute_dict) or ("tenure_months" in impute_dict)
    results["imputation_validity"] = {
        "pass": imputation_ok,
        "detail": f"Imputation strategies found: {impute_dict}" if imputation_ok 
                  else "No imputation strategies found for null columns."
    }
    
    # 3. Problem type inference check
    # Churn dataset should be classified as classification
    eda_report = state.get("eda_report")
    problem_type_ok = False
    inferred_type = "None"
    if eda_report:
        if hasattr(eda_report, "problem_type"):
            inferred_type = eda_report.problem_type
        elif isinstance(eda_report, dict):
            inferred_type = eda_report.get("problem_type")
        problem_type_ok = (inferred_type == "classification")
        
    results["problem_type_inference"] = {
        "pass": problem_type_ok,
        "detail": f"Correctly inferred problem type '{inferred_type}'" if problem_type_ok 
                  else f"Incorrectly inferred problem type '{inferred_type}' (expected 'classification')"
    }
    
    # Calculate aggregated pass rate
    passed_asserts = sum(1 for v in results.values() if v["pass"])
    total_asserts = len(results)
    pass_rate = (passed_asserts / total_asserts) * 100.0
    
    eval_report = {
        "pass_rate": pass_rate,
        "results": results
    }
    
    logger.info(f"Agent Eval Completed. Pass Rate: {pass_rate:.1f}% ({passed_asserts}/{total_asserts} checks passed)")
    return eval_report
