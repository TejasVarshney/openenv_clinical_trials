# Clinical Trial Matcher (OpenEnv Hackathon)

Team: pseudo su

This repository contains a deterministic OpenEnv-compatible environment where an agent matches unstructured patient EHR summaries to clinical trials.

## What Is Implemented

- Strict Pydantic models for observation/action/reward in [env/models.py](env/models.py)
- Core environment API with `reset()`, `step()`, and `state()` in [env/env.py](env/env.py)
- Deterministic grading logic in [env/grader.py](env/grader.py)
- Three tasks with increasing difficulty:
  - task1_easy_explicit (10 patients, 1 trial)
  - task2_medium_ontology (20 patients, 3 trials)
  - task3_hard_incomplete_data (30 patients, 5 trials)
- OpenEnv server app exposed through [server/app.py](server/app.py)
- Root `inference.py` using OpenAI client and strict `[START]`, `[STEP]`, `[END]` logs
- Dockerfile ready for Hugging Face Space deployment

## Project Layout

- [openenv.yaml](openenv.yaml)
- [inference.py](inference.py)
- [Dockerfile](Dockerfile)
- [requirements.txt](requirements.txt)
- [env/models.py](env/models.py)
- [env/env.py](env/env.py)
- [env/grader.py](env/grader.py)
- [env/data/task1_data.json](env/data/task1_data.json)
- [env/data/task2_data.json](env/data/task2_data.json)
- [env/data/task3_data.json](env/data/task3_data.json)
- [env/data/ground_truth.json](env/data/ground_truth.json)
- [server/app.py](server/app.py)
- [tests/test_env.py](tests/test_env.py)
- [tests/test_grader.py](tests/test_grader.py)

## Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validate OpenEnv Spec

```bash
openenv validate
```

If `openenv` is not on PATH, use:

```bash
.venv/Scripts/openenv.exe validate
```

## Run Tests

```bash
pytest -q
```

## Run Server

```bash
python -m server.app
```

Server endpoints include `/reset`, `/step`, `/state`, `/schema`, `/health`.

## Run Baseline Inference

Set environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Then run:

```bash
python inference.py
```

The script emits strict line logs:

- `[START] task=<task> env=<env> model=<model>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>`

## Docker

```bash
docker build -t clinical-trial-matcher .
docker run -p 8000:8000 clinical-trial-matcher
```

## Submission Pre-Validation

Use your provided `validate-submission.sh` script against the deployed Space URL. The server in this repo exposes `/reset` for Step 1 compatibility.
