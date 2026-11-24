import pandas as pd
from src.analyzer import analyze_dataset
from src.cleaner import clean_dataset
from src.rag_engine import PolicyRAGEngine
from src.ai_assistant import generate_forensic_plan, RemediationPlan
from src.guardrails import validate_plan_grounding


def _sample_dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 115, 25, 40],
            "experience_years": [2, 6, 10, 2, 15],
            "loc": ["kochi", "Kochi ", "DELHI", "kochi", "Mumbai"],
            "salary": [45000.0, 80000.0, 950000.0, 45000.0, None],
        }
    )


def test_analyzer_detects_anomalies():
    df = _sample_dirty_df()
    stats = analyze_dataset(df)
    assert stats["duplicate_rows"] == 1
    assert stats["total_missing"] == 1
    assert "age" in stats["outliers_by_col"]
    assert stats["quality_score"] < 100


def test_rag_policy_retrieval():
    df = _sample_dirty_df()
    stats = analyze_dataset(df)
    rag = PolicyRAGEngine()
    policies = rag.retrieve_relevant_policies(stats, top_k=3)
    assert len(policies) == 3
    assert any("POLICY-" in p for p in policies)


def test_structured_plan_and_guardrail():
    df = _sample_dirty_df()
    stats = analyze_dataset(df)
    rag = PolicyRAGEngine()
    policies = rag.retrieve_relevant_policies(stats)

    plan, _ = generate_forensic_plan(stats, policies)
    assert isinstance(plan, RemediationPlan)

    guardrail_report = validate_plan_grounding(plan, stats)
    assert guardrail_report["passed"] is True, f"Guardrail failed: {guardrail_report['issues']}"


def test_cleaner_improves_integrity_score():
    df = _sample_dirty_df()
    before = analyze_dataset(df)["quality_score"]
    cleaned_df, _ = clean_dataset(
        df,
        drop_dups=True,
        standardize_text=True,
        missing_method="Linear Regression Prediction",
    )
    after = analyze_dataset(cleaned_df)["quality_score"]
    assert after > before
    assert cleaned_df.isna().sum().sum() == 0