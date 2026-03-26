# MVM2 — Multi-LLM Math Verification System

A multi-model consensus verification system that solves math problems using **5 LLMs simultaneously**, cross-checks their step-by-step reasoning, and verifies answers using **SymPy symbolic computation**. Built for accuracy, not speed — because getting the right answer matters more than getting a fast one.

## How It Works

```
Problem Input (text or image)
        │
        ▼
┌───────────────────┐
│  5 LLMs solve in  │  GPT-OSS-120B, Llama-4-Scout, Qwen3-32B,
│  parallel          │  Llama-3.3-70B, Gemma-2-27B
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Step-by-Step     │  Parse each model's solution into
│  Parser           │  structured STEP → MATH → RESULT format
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  SymPy-Powered    │  Mathematically compare answers using
│  Consensus        │  symbolic equivalence (not string matching)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Symbolic         │  SymPy independently computes the answer
│  Verification     │  and cross-checks all model outputs
└────────┬──────────┘
         │
    Disagreement?
    │ yes
    ▼
┌───────────────────┐
│  Multi-Agent      │  Re-query disagreeing models,
│  Debate           │  run debate rounds
└────────┬──────────┘
         │
         ▼
  Verified Answer + Confidence Score + PDF Report
```

### Key Features

- **5 diverse LLMs** across Groq and NVIDIA NIM — different architectures catch different errors
- **SymPy symbolic equivalence** — `e^x*sin(x) + e^x*cos(x)` matches `exp(x)(sin(x)+cos(x))` because they're mathematically the same
- **Independent SymPy verification** — the system doesn't just vote, it independently computes the answer and checks if any model got it right
- **Real-time streaming UI** — watch models respond one by one, see consensus form, observe debates
- **LaTeX-compiled PDF reports** — publication-quality reports with proper math rendering via pdflatex
- **Image OCR** — upload photos of handwritten math problems (via Llama-4-Scout vision)
- **Handles everything** — arithmetic, algebra, calculus, linear algebra, differential equations, competition math

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for [Groq](https://console.groq.com/) and [NVIDIA NIM](https://build.nvidia.com/)

### 1. Clone and configure

```bash
git clone https://github.com/pulkitpareek18/mvm2.git
cd mvm2
cp .env.example .env
```

Edit `.env` with your API keys:

```env
GEMINI_API_KEY=your-gemini-key       # Optional (quota issues on free tier)
GROQ_API_KEY=your-groq-key           # Required
NVIDIA_API_KEY=your-nvidia-nim-key   # Required
```

### 2. Start with Docker

```bash
docker compose up -d --build
```

This starts two containers:
- **API server** at `http://localhost:8000` (FastAPI)
- **Frontend** at `http://localhost:8501` (Streamlit)

### 3. Open the UI

Go to **http://localhost:8501** and enter a math problem. Or use the API directly:

```bash
curl -X POST http://localhost:8000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"text": "Find the derivative of x^3 * sin(x)"}'
```

### 4. Download PDF Report

After verification, click "Download PDF Report" in the UI for a LaTeX-compiled report with proper math formatting.

## Local Development (without Docker)

```bash
# Requires Python 3.11+
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,frontend]"

# Start API
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (separate terminal)
streamlit run frontend/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Check all LLM providers |
| `POST` | `/api/solve` | Solve and verify (blocking) |
| `POST` | `/api/solve/stream` | Solve with SSE streaming events |
| `POST` | `/api/report/pdf` | Generate LaTeX PDF report |

### SSE Streaming Events

The `/api/solve/stream` endpoint emits real-time events:

```
data: {"stage": "solving", "event": "model_done", "data": {"model": "qwen3-32b", "latency_ms": 2500, "status": "ok"}, "progress": 0.25}
```

Events: `pipeline_start`, `model_start`, `model_done`, `parse_done`, `reparse_start`, `answer_agreement`, `step_consensus`, `symbolic_start`, `step_verified`, `resolution_start`, `debate_round`, `verify_done`, `final_result`

## Architecture

```
src/
├── main.py                    # FastAPI entrypoint
├── config.py                  # Settings (models, thresholds, timeouts)
├── models/                    # Pydantic data models
│   ├── problem.py             # MathProblem, ProblemType enum
│   ├── solution.py            # SolutionStep, ModelSolution
│   └── verification.py        # StepAlignment, VerificationResult
├── providers/                 # LLM provider adapters
│   ├── base.py                # LLMProvider protocol
│   ├── openai_compat.py       # Groq + NVIDIA NIM adapter (OpenAI-compatible)
│   ├── gemini.py              # Google Gemini adapter
│   └── registry.py            # Model registry
├── pipeline/                  # Core verification pipeline
│   ├── orchestrator.py        # Top-level pipeline coordinator
│   ├── solver.py              # Parallel model dispatch (asyncio.as_completed)
│   ├── step_parser.py         # 4-strategy parser (strict, loose, think-tags, reparse)
│   ├── step_aligner.py        # Cross-model step alignment
│   ├── consensus.py           # SymPy-powered equivalence + majority voting
│   ├── error_resolver.py      # Re-query + debate (budgeted)
│   ├── reparser.py            # LLM fallback for unparseable outputs
│   └── input_processor.py     # OCR + problem classification
├── symbolic/                  # Neuro-symbolic verification
│   ├── verifier.py            # Step-level SymPy verification
│   ├── answer_verifier.py     # Independent answer computation
│   ├── code_generator.py      # Math → Python/SymPy code
│   └── sandbox.py             # Sandboxed execution
├── prompts/                   # Prompt templates
│   ├── solver_prompt.py       # Structured step-by-step prompt
│   ├── debate_prompt.py       # Multi-agent debate prompt
│   └── code_gen_prompt.py     # Math-to-code prompt
└── api/                       # HTTP layer
    ├── routes.py              # REST + SSE endpoints
    └── pdf_report.py          # LaTeX PDF generation
```

## Model Lineup

| Model | Provider | Size | Role |
|-------|----------|------|------|
| openai/gpt-oss-120b | Groq | 120B | Primary solver |
| llama-4-scout-17b-16e | Groq | 17B | Fast + vision (OCR) |
| qwen/qwen3-32b | Groq | 32B | Diverse reasoning |
| meta/llama-3.3-70b-instruct | NVIDIA NIM | 70B | Cross-provider diversity |
| google/gemma-2-27b-it | NVIDIA NIM | 27B | Different architecture |

## Research Background

This system implements ideas from several published approaches:

- **Self-Consistency** (Wang et al.) — multiple reasoning paths + majority voting
- **Process Reward Models** (OpenAI, "Let's Verify Step by Step") — score each step, not just the final answer
- **Graph of Verification** — DAG-structured step verification
- **SymCode / MATH-VF** — neuro-symbolic verification via SymPy
- **Multi-Agent Debate** — models critique each other's reasoning

## Configuration

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `solver_temperature` | 0.3 | LLM temperature for solving |
| `parallel_timeout_seconds` | 90 | Max time per model |
| `consensus_threshold` | 0.6 | Min agreement ratio (3/5) |
| `max_debate_rounds` | 1 | Debate rounds per disputed step |
| `max_total_llm_calls` | 15 | Budget cap for resolution phase |
| `enable_symbolic_verification` | true | SymPy step verification |

## License

MIT
