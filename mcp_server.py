import pandas as pd
from fastmcp import FastMCP
from src.analyzer import analyze_dataset
from src.cleaner import clean_dataset
from src.rag_engine import PolicyRAGEngine

mcp = FastMCP(name="DataSentry-MCP-Server")
rag_engine = PolicyRAGEngine()


@mcp.tool
def audit_csv_file(file_path: str) -> dict:
    """Profiles a CSV file for integrity score, missing values, duplicates, IQR outliers, and matching governance policies."""
    df = pd.read_csv(file_path)
    analysis = analyze_dataset(df)
    policies = rag_engine.retrieve_relevant_policies(analysis)
    return {
        "analysis": analysis,
        "triggered_policies": policies,
    }


@mcp.tool
def repair_csv_file(
    input_path: str,
    output_path: str,
    drop_dups: bool = True,
    standardize_text: bool = True,
    missing_method: str = "Linear Regression Prediction",
) -> dict:
    """Repairs a dirty CSV dataset using deterministic cleaning and Scikit-Learn imputation, saving the result to output_path."""
    df = pd.read_csv(input_path)
    before_score = analyze_dataset(df)["quality_score"]

    cleaned_df, logs = clean_dataset(
        df,
        drop_dups=drop_dups,
        standardize_text=standardize_text,
        missing_method=missing_method,
    )
    cleaned_df.to_csv(output_path, index=False)
    after_score = analyze_dataset(cleaned_df)["quality_score"]

    return {
        "status": "success",
        "output_path": output_path,
        "integrity_score_before": before_score,
        "integrity_score_after": after_score,
        "actions_executed": logs,
    }


if __name__ == "__main__":
    mcp.run()