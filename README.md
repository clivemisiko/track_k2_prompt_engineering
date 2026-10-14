# Track K: AI Engineering — K2 Prompt Engineering & AI APIs

A production-grade implementation and experimental benchmark for getting reliable, structured behaviour out of Large Language Models (LLMs) through prompting alone, native schema enforcement, and resilient API client engineering.

---

## 🎯 Milestone Objectives

1. **Prompt Engineering Paradigms:** Benchmark **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting on a common temporal reasoning benchmark; document and explain observable differences in output quality.
2. **Constrained Generation:** Use system prompts, temperature modulation, and native JSON schema (Pydantic) to eliminate hallucinated formatting and guarantee machine-parseable outputs without manual string cleanup.
3. **Resilient API Architecture:** Integrate a real AI API (`google-genai` / Gemini 3 Flash Preview) with exponential backoff and error handling for HTTP 429 (rate limits), upstream timeouts, and missing credentials.
4. **Automated Verification:** Comprehensive test suite validating schemas, error handling, retries, and end-to-end pipeline execution.

---

## 📁 Repository Structure

```text
track_k2_prompt_engineering/
├── .env.example                      # Template for environment variables
├── .gitignore                        # Standard exclusions (.env, .venv, __pycache__)
├── requirements.txt                  # Production dependencies
├── pytest.ini                        # Pytest runner configuration
├── README.md                         # Architecture, benchmarks, and run guide
├── generate_notebook.py              # Script to build prompt_experiments.ipynb
├── notebooks/
│   └── prompt_experiments.ipynb     # 10 documented prompt experiments & quality matrix
├── src/
│   ├── __init__.py
│   ├── client.py                     # Resilient GeminiClient with exponential backoff & rate limit handling
│   ├── schemas.py                    # Pydantic models for structured output & validation
│   └── pipeline.py                   # Standalone CLI comparing techniques & structured triage
└── tests/
    ├── __init__.py
    ├── test_schemas.py               # Validates Pydantic types, bounds, and custom validators
    ├── test_client.py                # Tests API key loading, 429 backoff retries, and timeouts
    └── test_pipeline.py              # Tests prompt completeness and pipeline integration
```

---

## 🔬 Benchmark Progression & Experiment Matrix

All 10 experiments are documented in [`notebooks/prompt_experiments.ipynb`](notebooks/prompt_experiments.ipynb).

| Exp # | Technique / Configuration | Reasoning Depth | Format Reliability | Hallucination Resistance | Observed Quality Difference |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Zero-Shot Baseline** | Low | Unpredictable | Moderate | Answers immediately without intermediate tokens; skips temporal cooldown window. |
| **2** | **Few-Shot In-Context** | Moderate | High | Good | Adopts exemplar format and isolates parameters cleanly, but lacks internal logic trace. |
| **3** | **Chain-of-Thought (CoT)** | Very High | Moderate | Very High | Generates step-by-step intermediate state tokens; accurately deduces cooldown violation. |
| **4** | **System Persona Grounding** | Moderate | High | High | Senior Auditor role eliminates conversational pleasantries and enforces technical rigor. |
| **5** | **Temperature Ablation** | N/A | Variable | $T=0.0$ High, $T=1.2$ Low | $T=0.0$ yields deterministic deduction; $T=1.2$ introduces hallucinated terms. |
| **6** | **Prompt-Constrained JSON** | Moderate | Fragile (Fails parser) | Moderate | Prepends markdown tags (````json ... ````) and commentary, causing `json.loads` failure. |
| **7** | **Native Schema (Pydantic)** | Moderate | **100% Parseable** | High | Logit-level constraint guarantees clean deserialization into Pydantic models. |
| **8** | **Anti-Hallucination Guardrails** | High | High | Maximum | Negative constraints prevent invented facts; returns `'UNSPECIFIED'` for missing data. |
| **9** | **Boundary Edge-Case Few-Shot** | High | High | High | Demonstrates exact limit handling ($=15.0\text{s}, =600\text{ req}$) to resolve edge ambiguities. |
| **10** | **Structured CoT** | **Maximum** | **100% Parseable** | **Maximum** | Full explainability paired with guaranteed programmatic deserialization. |

---

## 🛡️ Reliability & Error Handling

The client implementation in [`src/client.py`](src/client.py) satisfies all production resilience criteria:

- **Environment Variable Security:** Loaded via `python-dotenv` and resolved from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Raises a descriptive `MissingAPIKeyError` if absent.
- **Rate Limit (HTTP 429 / Resource Exhausted):** Caught by `_execute_with_retry` and backed off exponentially using `tenacity` (`min=2s`, `max=10s`, `max_retries=3`).
- **Timeout Management:** Catches network timeouts (`TimeoutError`) and retries up to configured thresholds.
- **Native JSON Schema Mode:** Passed directly via `types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)`, completely bypassing fragile regex or manual string cleaning.

---

## 🚀 How to Run

### 1. Environment Setup

```powershell
# Navigate to the workspace
cd C:\Users\Misiko\track_k2_prompt_engineering

# Create virtual environment (if not already created)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory (or copy from `.env.example`):
```ini
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_MODEL=gemini-2.5-flash
DEFAULT_TEMPERATURE=0.2
REQUEST_TIMEOUT_SECONDS=30
MAX_RETRIES=3
```

### 3. Run Automated Tests

Execute the full test suite with verbose output:
```powershell
pytest tests/ -v
```
All unit tests validate schema validation, rate-limit backoff recovery, timeout handling, and pipeline components without consuming live API tokens.

### 4. Execute the Standalone Pipeline

Run the end-to-end prompt engineering and structured triage pipeline:
```powershell
python -m src.pipeline
```

### 5. Launch the Experiment Notebook

```powershell
jupyter lab notebooks/prompt_experiments.ipynb
```

---

## ✅ Pass Criteria Verification

| Pass Criterion | Status | Evidence / Implementation |
| :--- | :---: | :--- |
| **Prompting comparison shows real, explainable quality differences** | ✅ **Passed** | Documented in Notebook Exp 1–3 and `src/pipeline.py`. CoT resolves multi-step cooldown states where zero-shot fails. |
| **Structured output actually parses without manual cleanup** | ✅ **Passed** | Implemented in `src/schemas.py` and `GeminiClient.generate_structured()`. Validated with Pydantic. |
| **API key handled via env vars** | ✅ **Passed** | Managed through `.env` and `os.getenv` in `src/client.py`; no hardcoded credentials. |
| **Failures (rate limit, timeout) are caught, not left to crash script** | ✅ **Passed** | Exponential backoff retry loop in `src/client.py`; unit tests in `tests/test_client.py`. |
| **At least 10 prompt experiments documented** | ✅ **Passed** | 10 distinct experiments with hypotheses, executions, and comparison matrix in `notebooks/prompt_experiments.ipynb`. |
