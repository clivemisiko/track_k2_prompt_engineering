"""
Unit tests for the end-to-end prompt engineering and structured pipelines.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline import (
    PROMPTS,
    REASONING_TASK,
    run_prompt_comparison,
    run_structured_output_pipeline,
    run_cot_structured_analysis,
)
from src.schemas import CustomerSupportTriage, TaskAnalysisResult, ReasoningStep


class TestPipelineComponents:
    def test_prompts_dictionary_completeness(self):
        """Ensure all required prompting techniques are configured."""
        required = {"zero_shot", "few_shot", "chain_of_thought"}
        assert required.issubset(set(PROMPTS.keys()))
        for key in required:
            assert REASONING_TASK.strip() in PROMPTS[key]

    def test_run_prompt_comparison_mocked(self):
        """Validates that run_prompt_comparison calls the client for each technique."""
        mock_client = MagicMock()
        mock_client.generate_text.side_effect = [
            "Zero-shot output: violation occurred.",
            "Few-shot output: analysis shows cooldown violation.",
            "Chain-of-thought: Step 1... Step 2... Verdict: violation.",
        ]

        results = run_prompt_comparison(mock_client)

        assert len(results) == 3
        assert "zero_shot" in results
        assert "few_shot" in results
        assert "chain_of_thought" in results
        assert mock_client.generate_text.call_count == 3

    def test_run_structured_output_pipeline_mocked(self):
        """Validates that run_structured_output_pipeline returns validated model."""
        mock_client = MagicMock()
        mock_triage = CustomerSupportTriage(
            ticket_id="TCK-9942",
            sentiment="negative",
            urgency="critical",
            category="technical_bug",
            entities=["SDK v2.4", "ACC-88310"],
            action_items=["Revert immediately"],
            summary="SDK upgrade broke analytics.",
        )
        mock_client.generate_structured.return_value = mock_triage

        result = run_structured_output_pipeline(mock_client)
        assert result.ticket_id == "TCK-9942"
        assert result.category == "technical_bug"
        assert result.urgency == "critical"
        mock_client.generate_structured.assert_called_once()

    def test_run_cot_structured_analysis_mocked(self):
        """Validates that run_cot_structured_analysis returns parsed TaskAnalysisResult."""
        mock_client = MagicMock()
        mock_result = TaskAnalysisResult(
            task_id="RATE-LIMIT-01",
            prompt_technique="chain-of-thought",
            chain_of_thought=[
                ReasoningStep(
                    step_number=1,
                    thought="Examining Tier B constraints",
                    intermediate_conclusion="500 req/min base, 600 burst for 15s, 30s cooldown",
                )
            ],
            final_verdict="Violation triggered during 30-second cooldown.",
            confidence_score=0.98,
            key_factors=["Sent 50 requests while in cooldown"],
        )
        mock_client.generate_structured.return_value = mock_result

        result = run_cot_structured_analysis(mock_client)
        assert result.task_id == "RATE-LIMIT-01"
        assert len(result.chain_of_thought) == 1
        assert result.confidence_score == 0.98
