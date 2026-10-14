"""
Track K: AI Engineering - K2 Prompt Engineering & AI APIs
Pipeline demonstrating Prompting Techniques (Zero-Shot vs Few-Shot vs Chain-of-Thought)
and production structured output parsing with robust failure recovery.
"""

import os
import sys
import json
from typing import Dict, Any

from src.client import GeminiClient, LLMAPIError, LLMRateLimitError, LLMTimeoutError
from src.schemas import TaskAnalysisResult, CustomerSupportTriage, ReasoningStep


# A tricky multi-constraint deduction problem where Zero-Shot typically trips up or hallucinates,
# but Chain-of-Thought correctly reasons step-by-step through constraints.
REASONING_TASK = """
Scenario: An API rate-limiter has three tiers:
- Tier A: Max 100 req/min, burst allowance 120 for up to 30 seconds, cooldown 60 seconds after burst.
- Tier B: Max 500 req/min, burst allowance 600 for up to 15 seconds, cooldown 30 seconds after burst.
- Tier C: Max 2000 req/min, no burst allowance.

A client on Tier B sends requests at:
- Minute 1: 450 requests steadily.
- Minute 2: Starts with a sudden burst of 580 requests in the first 12 seconds, then attempts 50 more requests within the next 10 seconds.

Question: Did the client violate the rate limit or trigger throttling in Minute 2? Explain exactly why or why not, identify the cooldown window, and state the final verdict.
"""

# Prompt definitions for technique ablation
PROMPTS = {
    "zero_shot": f"""
Answer the following problem directly:
{REASONING_TASK}
Give your final verdict and short explanation.
""",

    "few_shot": f"""
Here are examples of evaluating rate limiter compliance:

Example 1:
Tier A client sends 110 requests in 20 seconds.
Analysis: 110 requests is within the burst allowance of 120 requests for 30s. No violation during burst, but 60s cooldown is triggered immediately after.
Verdict: Compliant with burst, cooldown triggered.

Example 2:
Tier C client attempts 2100 requests in 45 seconds.
Analysis: Tier C has max 2000 req/min with NO burst allowance. Exceeding 2000 req is an immediate hard limit breach.
Verdict: Rate limit violation.

Now solve this problem following the exact pattern:
{REASONING_TASK}
""",

    "chain_of_thought": f"""
Solve the following rate limiter problem using step-by-step reasoning.
Before stating any conclusion, work through each condition systematically:
Step 1: Identify client tier and applicable limits (steady, burst limit, burst duration, cooldown).
Step 2: Evaluate Minute 1 activity against Minute 1 quota.
Step 3: Evaluate Minute 2 burst volume (580 req in 12s) against burst allowance and duration.
Step 4: Determine the state of the client immediately after the 12-second burst (cooldown trigger).
Step 5: Evaluate the subsequent 50 requests attempted in the next 10 seconds against the cooldown state.
Step 6: Synthesize the findings into the final verdict.

Problem:
{REASONING_TASK}
"""
}


def run_prompt_comparison(client: GeminiClient) -> Dict[str, str]:
    """
    Runs Zero-shot, Few-shot, and Chain-of-Thought prompting on the identical task.
    Returns the outputs for side-by-side comparison.
    """
    results = {}
    print("\n" + "=" * 80)
    print("RUNNING PROMPT TECHNIQUE COMPARISON (Zero-Shot vs Few-Shot vs Chain-of-Thought)")
    print("=" * 80)

    for technique, prompt_text in PROMPTS.items():
        print(f"\n[Executing: {technique.upper()}] ...")
        try:
            output = client.generate_text(
                prompt=prompt_text,
                temperature=0.1,  # Low temperature to isolate prompt structure impact
            )
            results[technique] = output
            print(f"--- Output for {technique} ---")
            print(output.strip())
            print("-" * 60)
        except Exception as e:
            print(f"Error during {technique}: {e}")
            results[technique] = f"Error: {e}"

    return results


def run_structured_output_pipeline(client: GeminiClient) -> CustomerSupportTriage:
    """
    Demonstrates reliable, type-safe structured output generation.
    Constrains the LLM to output conforming to the CustomerSupportTriage schema.
    """
    print("\n" + "=" * 80)
    print("RUNNING STRUCTURED OUTPUT PIPELINE (Customer Support Triage)")
    print("=" * 80)

    sample_ticket = """
    Ticket ID: TCK-9942
    From: sarah.connor@cyberdyne.tech
    Subject: URGENT: Production database integration down after v2.4 upgrade!
    
    Hi Support,
    Our entire billing analytics pipeline crashed 20 minutes ago right after upgrading your SDK
    to v2.4. We're seeing recurring HTTP 500 errors and our automated billing run for 4,200 active
    customers is stalled. We are losing transactions every minute! Please revert or advise hotfix ASAP.
    Our account number is ACC-88310.
    """

    system_instruction = (
        "You are an automated support operations triage agent. Analyze the ticket "
        "and return strictly validated JSON matching the target schema."
    )

    prompt = f"Analyze and triage the following support ticket:\n{sample_ticket}"

    print("Sending prompt with response_schema=CustomerSupportTriage ...")
    try:
        triage_result: CustomerSupportTriage = client.generate_structured(
            prompt=prompt,
            response_schema=CustomerSupportTriage,
            system_instruction=system_instruction,
            temperature=0.0,
        )

        print("\nSuccessfully parsed into Pydantic model without manual cleanup!")
        print(f"Ticket ID:     {triage_result.ticket_id}")
        print(f"Category:      {triage_result.category}")
        print(f"Urgency:       {triage_result.urgency}")
        print(f"Sentiment:     {triage_result.sentiment}")
        print(f"Entities:      {triage_result.entities}")
        print(f"Action Items:  {triage_result.action_items}")
        print(f"Summary:       {triage_result.summary}")
        return triage_result

    except (LLMRateLimitError, LLMTimeoutError, LLMAPIError) as e:
        print(f"Caught upstream API exception gracefully: {e}")
        raise
    except ValueError as e:
        print(f"Schema parsing error: {e}")
        raise


def run_cot_structured_analysis(client: GeminiClient) -> TaskAnalysisResult:
    """
    Combines Chain-of-Thought prompting with structured Pydantic schema validation.
    """
    print("\n" + "=" * 80)
    print("RUNNING CHAIN-OF-THOUGHT WITH STRUCTURED SCHEMA")
    print("=" * 80)

    prompt = f"""
    Perform a complete Chain-of-Thought analysis for the rate-limiter problem below.
    Deconstruct the reasoning into atomic sequential steps with thoughts and intermediate conclusions.
    
    Problem:
    {REASONING_TASK}
    """

    system_instruction = (
        "You are an expert distributed systems engineer and rate-limiting auditor. "
        "Provide thorough, step-by-step reasoning steps leading to an indisputable verdict."
    )

    result: TaskAnalysisResult = client.generate_structured(
        prompt=prompt,
        response_schema=TaskAnalysisResult,
        system_instruction=system_instruction,
        temperature=0.1,
    )

    print("\nChain-of-Thought Result Parsed:")
    print(f"Task ID:          {result.task_id}")
    print(f"Technique:        {result.prompt_technique}")
    print(f"Confidence:       {result.confidence_score * 100:.1f}%")
    print(f"Final Verdict:    {result.final_verdict}")
    print("\nReasoning Steps:")
    for step in result.chain_of_thought:
        print(f"  Step {step.step_number}: {step.thought}")
        if step.intermediate_conclusion:
            print(f"    --> {step.intermediate_conclusion}")

    return result


def main():
    """Main execution entry point."""
    print("Initializing GeminiClient from environment variables...")
    try:
        client = GeminiClient()
        print(f"Client initialized successfully (Model: {client.default_model})")
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    # 1. Compare prompt techniques
    run_prompt_comparison(client)

    # 2. Run structured output pipeline (CustomerSupportTriage)
    run_structured_output_pipeline(client)

    # 3. Run structured Chain-of-Thought (TaskAnalysisResult)
    run_cot_structured_analysis(client)

    print("\nAll pipeline tasks executed successfully!")


if __name__ == "__main__":
    main()
