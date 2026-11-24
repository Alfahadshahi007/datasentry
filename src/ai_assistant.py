import os
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

ImputationOption = Literal[
    "None",
    "Column Median",
    "Linear Interpolation (Between Rows)",
    "Linear Regression Prediction",
]


class RemediationPlan(BaseModel):
    executive_summary: str = Field(
        description="Plain-English forensic summary of dataset health and policy violations."
    )
    cited_policies: list[str] = Field(
        description="Exact policy IDs and rules triggered by this dataset."
    )
    drop_duplicates: bool = Field(
        description="Whether exact duplicate rows should be purged."
    )
    standardize_text: bool = Field(
        description="Whether string columns should be stripped and Title-Cased."
    )
    imputation_method: ImputationOption = Field(
        description="Recommended missing-value imputation strategy."
    )
    rationale: str = Field(
        description="Technical justification for the selected remediation parameters."
    )


def _build_fallback_plan(analysis: dict, retrieved_policies: list[str]) -> RemediationPlan:
    has_dups = analysis.get("duplicate_rows", 0) > 0
    has_missing = analysis.get("total_missing", 0) > 0
    numeric_cols = analysis.get("numeric_columns", [])

    if not has_missing:
        imp_method: ImputationOption = "None"
    elif len(numeric_cols) >= 2:
        imp_method = "Linear Regression Prediction"
    else:
        imp_method = "Column Median"

    summary = (
        f"Dataset integrity is **{analysis.get('quality_score', 0)}%** across "
        f"**{analysis.get('total_rows', 0)} rows**. Detected "
        f"**{analysis.get('total_missing', 0)} missing cells**, "
        f"**{analysis.get('duplicate_rows', 0)} duplicate rows**, and "
        f"**{analysis.get('total_outliers', 0)} statistical outliers**."
    )

    return RemediationPlan(
        executive_summary=summary,
        cited_policies=retrieved_policies[:3],
        drop_duplicates=has_dups,
        standardize_text=True,
        imputation_method=imp_method,
        rationale=(
            f"Selected '{imp_method}' imputation based on {len(numeric_cols)} available numeric "
            f"columns and active governance policies."
        ),
    )


def generate_forensic_plan(
    analysis: dict, retrieved_policies: list[str]
) -> tuple[RemediationPlan, str]:
    """
    Returns (RemediationPlan, engine_source) using Gemini Structured Outputs + Qdrant RAG,
    with automatic fallback to deterministic policy logic.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _build_fallback_plan(analysis, retrieved_policies), "Local Policy Engine (No API Key)"

    policies_block = "\n".join(f"- {p}" for p in retrieved_policies)
    prompt = f"""
You are DataSentry, an enterprise AI Data Governance & Forensic Remediation Agent.
Analyze the following aggregated dataset telemetry (no raw PII rows are shared) and cross-check it against the retrieved company governance rules.

### Retrieved Governance Policies (Qdrant RAG Context):
{policies_block}

### Dataset Telemetry:
- Rows: {analysis.get('total_rows')} | Columns: {analysis.get('total_columns')}
- Column Names: {analysis.get('column_names')}
- Numeric Columns: {analysis.get('numeric_columns')}
- Integrity Health Score: {analysis.get('quality_score')}%
- Missing Values by Column: {analysis.get('missing_by_col')}
- Duplicate Rows: {analysis.get('duplicate_rows')}
- IQR Outliers by Column: {analysis.get('outliers_by_col')}

Produce a strictly structured remediation plan grounded in the retrieved policies.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RemediationPlan,
                temperature=0.2,
            ),
        )
        if response.parsed:
            return response.parsed, "Gemini 2.5-Flash + Qdrant RAG"
    except Exception:
        pass

    return _build_fallback_plan(analysis, retrieved_policies), "Local Policy Engine (Fallback)"