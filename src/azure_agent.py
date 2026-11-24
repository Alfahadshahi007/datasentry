import os
import json
from typing import Literal
from pydantic import BaseModel
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import InteractiveBrowserCredential


class ForensicPlan(BaseModel):
    executive_summary: str
    cited_policies: list[str]
    drop_duplicates: bool
    standardize_text: bool
    imputation_method: Literal[
        "None",
        "Column Median",
        "Linear Interpolation (Between Rows)",
        "Linear Regression Prediction",
    ]
    rationale: str


def run_azure_forensic_agent(
    analysis: dict, retrieved_policies: list[dict]
) -> tuple[ForensicPlan, str]:
    """
    Executes DataSentry's forensic audit using Azure AI Agent Service (v2 SDK)
    and returns a validated Pydantic ForensicPlan.
    """
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    if not endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT is not set.")

    credential = InteractiveBrowserCredential(tenant_id=tenant_id)
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    )

    policy_context = "\n".join(
        f"- [{p.get('id', 'POL')}] {p.get('title', '')}: {p.get('rule', '')}"
        for p in retrieved_policies
    )

    instructions = (
        "You are DataSentry, an enterprise forensic data governance agent. "
        "Analyze the provided dataset telemetry and retrieved Qdrant governance policies. "
        "Respond strictly with valid JSON matching this exact schema and no extra text:\n"
        '{"executive_summary": str, "cited_policies": [str], '
        '"drop_duplicates": bool, "standardize_text": bool, '
        '"imputation_method": "None" | "Column Median" | '
        '"Linear Interpolation (Between Rows)" | "Linear Regression Prediction", '
        '"rationale": str}'
    )

    prompt = (
        f"Dataset Telemetry:\n{json.dumps(analysis, default=str)}\n\n"
        f"Retrieved Governance Policies:\n{policy_context}"
    )

    with project_client:
        agent = project_client.agents.create_version(
            agent_name="datasentry-forensic-agent",
            definition=PromptAgentDefinition(
                model=model_name,
                instructions=instructions,
            ),
        )

        try:
            openai_client = project_client.get_openai_client()
            response = openai_client.responses.create(
                input=prompt,
                extra_body={
                    "agent_reference": {
                        "name": agent.name,
                        "type": "agent_reference",
                    }
                },
            )

            raw_text = (response.output_text or "").strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()

            plan = ForensicPlan.model_validate_json(raw_text)
            return plan, f"Azure AI Agent Service ({model_name})"

        finally:
            project_client.agents.delete_version(
                agent_name=agent.name,
                agent_version=agent.version,
            )
