import re
from src.ai_assistant import RemediationPlan


def validate_plan_grounding(plan: RemediationPlan, analysis: dict) -> dict:
    """
    Deterministic guardrail verifying that the LLM's RemediationPlan
    aligns with dataset ground truth and cites retrieved RAG policies.
    """
    issues = []
    total_missing = analysis.get("total_missing", 0)
    duplicate_rows = analysis.get("duplicate_rows", 0)

    # Check 1: Imputation should only trigger when missing values actually exist
    if total_missing == 0 and plan.imputation_method != "None":
        issues.append("Recommended imputation despite 0 missing values.")

    # Check 2: Duplicate purge should trigger when duplicate rows exist
    if duplicate_rows > 0 and not plan.drop_duplicates:
        issues.append("Failed to recommend purging existing duplicate rows.")

    # Check 3: Verify governance policies were cited
    cited_ids = re.findall(r"\[([A-Z]+-\d+)\]", " ".join(plan.cited_policies))
    if not plan.cited_policies:
        issues.append("No governance policies cited in structured output.")

    return {
        "passed": len(issues) == 0,
        "cited_policy_ids": cited_ids,
        "issues": issues,
    }