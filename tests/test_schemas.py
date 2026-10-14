"""
Unit tests for Pydantic structured output schemas.
Validates strict type enforcement, range checks, and custom validators.
"""

import pytest
from pydantic import ValidationError

from src.schemas import ReasoningStep, TaskAnalysisResult, CustomerSupportTriage


class TestReasoningStep:
    def test_valid_reasoning_step(self):
        step = ReasoningStep(
            step_number=1,
            thought="Checking if the burst limit of 600 req is respected.",
            intermediate_conclusion="580 req is within 600 req allowance."
        )
        assert step.step_number == 1
        assert "580 req" in step.intermediate_conclusion

    def test_invalid_step_number(self):
        with pytest.raises(ValidationError):
            ReasoningStep(step_number=0, thought="Invalid zero index")


class TestTaskAnalysisResult:
    def test_valid_task_analysis_result(self):
        res = TaskAnalysisResult(
            task_id="TASK-001",
            prompt_technique="chain-of-thought",
            chain_of_thought=[
                ReasoningStep(step_number=1, thought="Step 1 analysis"),
                ReasoningStep(step_number=2, thought="Step 2 analysis"),
            ],
            final_verdict="Violation occurred during cooldown.",
            confidence_score=0.95,
            key_factors=["Burst triggered cooldown", "Subsequent requests blocked"],
        )
        assert res.task_id == "TASK-001"
        assert res.confidence_score == 0.95
        assert len(res.chain_of_thought) == 2

    def test_invalid_confidence_score_range(self):
        # Confidence score must be between 0.0 and 1.0
        with pytest.raises(ValidationError):
            TaskAnalysisResult(
                task_id="TASK-ERR",
                prompt_technique="zero-shot",
                final_verdict="Verdict",
                confidence_score=1.5,  # Out of range
            )

    def test_invalid_technique_literal(self):
        with pytest.raises(ValidationError):
            TaskAnalysisResult(
                task_id="TASK-ERR",
                prompt_technique="unsupported-technique",  # Invalid literal
                final_verdict="Verdict",
                confidence_score=0.8,
            )


class TestCustomerSupportTriage:
    def test_valid_triage(self):
        triage = CustomerSupportTriage(
            ticket_id="TCK-100",
            sentiment="negative",
            urgency="critical",
            category="technical_bug",
            entities=["SDK v2.4", "ACC-88310"],
            action_items=["Rollback to v2.3 immediately", "Notify on-call engineer"],
            summary="SDK v2.4 upgrade caused 500 errors stalling billing for 4,200 users.",
        )
        assert triage.ticket_id == "TCK-100"
        assert triage.urgency == "critical"

    def test_summary_too_long_fails_validation(self):
        # Validator ensures summary does not exceed 280 characters
        long_summary = "A" * 300
        with pytest.raises(ValidationError):
            CustomerSupportTriage(
                ticket_id="TCK-FAIL",
                sentiment="neutral",
                urgency="low",
                category="general_inquiry",
                action_items=["Check docs"],
                summary=long_summary,
            )

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            CustomerSupportTriage(
                ticket_id="TCK-FAIL",
                sentiment="positive",
                urgency="low",
                category="unsupported_category",
                action_items=["Review"],
                summary="Valid summary length.",
            )
