"""
Track K: AI Engineering - K2 Prompt Engineering & AI APIs
Pydantic schemas for strictly validated structured LLM outputs.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ReasoningStep(BaseModel):
    """Represents an atomic reasoning step in a Chain-of-Thought (CoT) trace."""
    step_number: int = Field(..., ge=1, description="Sequential index of the reasoning step.")
    thought: str = Field(..., min_length=3, description="Detailed explanation of the deduction or verification.")
    intermediate_conclusion: Optional[str] = Field(
        None, description="Sub-conclusion reached at this step."
    )


class TaskAnalysisResult(BaseModel):
    """
    Standard schema for analyzing complex queries, evaluating reasoning,
    and returning structured, machine-parseable conclusions.
    """
    task_id: str = Field(..., description="Unique identifier for the task or query.")
    prompt_technique: Literal["zero-shot", "few-shot", "chain-of-thought"] = Field(
        ..., description="Prompting strategy used to generate this response."
    )
    chain_of_thought: List[ReasoningStep] = Field(
        default_factory=list,
        description="Structured reasoning steps preceding the final verdict."
    )
    final_verdict: str = Field(..., min_length=1, description="Concise, definite resolution.")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model calibrated confidence between 0.0 and 1.0."
    )
    key_factors: List[str] = Field(
        default_factory=list, description="Primary signals supporting the final verdict."
    )


class CustomerSupportTriage(BaseModel):
    """
    Real-world entity extraction & triage schema for customer support tickets.
    """
    ticket_id: str = Field(..., description="ID of the incoming support ticket.")
    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(
        ..., description="Assessed emotional tone of the communication."
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Operational urgency classification."
    )
    category: Literal[
        "billing", "technical_bug", "feature_request", "account_access", "general_inquiry"
    ] = Field(..., description="Routing category.")
    entities: List[str] = Field(
        default_factory=list, description="Extracted entities like product names, dates, amounts."
    )
    action_items: List[str] = Field(
        ..., min_length=1, description="Concrete immediate next steps for the support agent."
    )
    summary: str = Field(
        ..., max_length=280, description="Short executive summary (<= 280 characters)."
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        if len(v) > 280:
            raise ValueError("Summary must not exceed 280 characters.")
        return v
