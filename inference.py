"""Baseline inference runner for the Clinical Trial Matcher environment."""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Sequence

from openai import OpenAI

from env.env import ClinicalTrialMatcherEnv
from env.models import Action, Enroll, Observation, Reject, RequestLabResult, parse_action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/Llama-3.1-Nemotron-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "NONE")
API_KEY = OPENAI_API_KEY or HF_TOKEN

print(API_BASE_URL, MODEL_NAME)

BENCHMARK = "clinical-trial-matcher"
TASKS: tuple[str, ...] = (
    "task1_easy_explicit",
    "task2_medium_ontology",
    "task3_hard_incomplete_data",
)
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "180"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical trial matching assistant.
    Decide exactly one action as strict JSON with keys that match this schema:
    - enroll: {"action_type":"enroll","trial_id":"TRIAL_X"}
    - reject: {"action_type":"reject","reason":"short reason"}
    - request_lab: {"action_type":"request_lab","test_name":"HbA1c"}

    Rules:
    - If key lab data is pending/missing, request labs first.
    - If exclusion criteria are clearly present, reject.
    - Otherwise select the best trial among available options.
    - Return JSON only. No markdown, no extra keys.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_value = str(done).lower()
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: Sequence[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    success_value = str(success).lower()
    print(
        f"[END] success={success_value} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def compact_action(action: Action) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)


def has_lab_value(note: str, lab_name: str) -> bool:
    pattern = rf"\b{re.escape(lab_name)}\b\s*[:=]?\s*[<>]?[\d]"
    return re.search(pattern, note, flags=re.IGNORECASE) is not None


def extract_lab_value(note: str, lab_name: str) -> float | None:
    pattern = rf"\b{re.escape(lab_name)}\b\s*[:=]?\s*([<>]?)\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, note, flags=re.IGNORECASE)
    if match is None:
        return None

    try:
        return float(match.group(2))
    except ValueError:
        return None


def choose_available_trial(
    observation: Observation,
    preferred_ids: Sequence[str],
) -> str | None:
    available_ids = {trial.trial_id.upper() for trial in observation.available_trials}
    for trial_id in preferred_ids:
        if trial_id.upper() in available_ids:
            return trial_id.upper()
    if observation.available_trials:
        return observation.available_trials[0].trial_id
    return None


def extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def heuristic_action(observation: Observation) -> Action:
    note = observation.active_patient_ehr
    lower_note = note.lower()
    hard_mode = choose_available_trial(observation, ("TRIAL_E",)) is not None

    labs = ["HbA1c", "eGFR", "LDL", "TSH", "Platelet", "ANC", "AST", "ALT"]
    for lab in labs:
        pending_pattern = rf"\b{re.escape(lab.lower())}\b\s+pending"
        if re.search(pending_pattern, lower_note) and not has_lab_value(note, lab):
            return RequestLabResult(test_name=lab)

    exclusion_markers = [
        "asthma",
        "ckd stage 4",
        "pregnan",
        "heart failure",
        "active hepatitis",
        "dialysis",
        "untreated atrial fibrillation",
        "active bacteremia",
        "active sepsis",
        "dka admission",
    ]
    if any(marker in lower_note for marker in exclusion_markers):
        return Reject(reason="Exclusion criterion found in EHR.")

    if "type 2 diabetes" in lower_note:
        if "dka" in lower_note and "no dka" not in lower_note:
            return Reject(reason="Recent DKA exclusion in diabetes protocol.")

        hba1c = extract_lab_value(note, "HbA1c")
        if hba1c is None:
            if hard_mode:
                return RequestLabResult(test_name="HbA1c")
            trial_id = choose_available_trial(observation, ("TRIAL_C", "TRIAL_A"))
            if trial_id is not None:
                return Enroll(trial_id=trial_id)
            return Reject(reason="Insufficient diabetes data for enrollment.")
        if hba1c < 7.5:
            trial_id = choose_available_trial(observation, ("TRIAL_E", "TRIAL_C", "TRIAL_A"))
            if trial_id is None:
                return Reject(reason="No diabetes trial available.")
            return Enroll(trial_id=trial_id)
        return Reject(reason="HbA1c above threshold for diabetes trial.")

    if "ckd" in lower_note or "egfr" in lower_note:
        if "dialysis" in lower_note:
            return Reject(reason="Dialysis exclusion in renal trial.")

        egfr = extract_lab_value(note, "eGFR")
        if egfr is None:
            if hard_mode:
                return RequestLabResult(test_name="eGFR")
            trial_id = choose_available_trial(observation, ("TRIAL_B",))
            if trial_id is not None:
                return Enroll(trial_id=trial_id)
            return Reject(reason="Insufficient renal data for enrollment.")
        if egfr >= 60:
            trial_id = choose_available_trial(observation, ("TRIAL_F", "TRIAL_B"))
            if trial_id is None:
                return Reject(reason="No renal/hypertension trial available.")
            return Enroll(trial_id=trial_id)
        return Reject(reason="eGFR below renal trial threshold.")

    if "hyperlipidemia" in lower_note or "ldl" in lower_note:
        ldl = extract_lab_value(note, "LDL")
        ast = extract_lab_value(note, "AST")
        alt = extract_lab_value(note, "ALT")
        ast_pending = re.search(r"\bast\b\s+pending", lower_note) is not None
        alt_pending = re.search(r"\balt\b\s+pending", lower_note) is not None

        if ldl is None:
            if hard_mode:
                return RequestLabResult(test_name="LDL")
            return Reject(reason="LDL value unavailable for lipid decision.")
        if ast_pending and ast is None:
            return RequestLabResult(test_name="AST")
        if alt_pending and alt is None:
            return RequestLabResult(test_name="ALT")

        if "active hepatitis" in lower_note or "active liver disease" in lower_note:
            return Reject(reason="Active liver disease exclusion for lipid trial.")
        if (ast is not None and ast > 120) or (alt is not None and alt > 120):
            return Reject(reason="Liver enzyme exclusion for lipid trial.")
        if ldl >= 130:
            trial_id = choose_available_trial(observation, ("TRIAL_G", "TRIAL_D"))
            if trial_id is None:
                return Reject(reason="No lipid trial available.")
            return Enroll(trial_id=trial_id)
        return Reject(reason="LDL below lipid trial threshold.")

    if "hypothyroidism" in lower_note or "tsh" in lower_note:
        if "untreated atrial fibrillation" in lower_note:
            return Reject(reason="Atrial fibrillation exclusion in thyroid trial.")

        tsh = extract_lab_value(note, "TSH")
        if tsh is None:
            return RequestLabResult(test_name="TSH")
        if 0.5 <= tsh <= 5.0:
            trial_id = choose_available_trial(observation, ("TRIAL_H",))
            if trial_id is None:
                return Reject(reason="No thyroid trial available.")
            return Enroll(trial_id=trial_id)
        return Reject(reason="TSH outside thyroid trial range.")

    if (
        "oncology" in lower_note
        or "cancer" in lower_note
        or re.search(r"\banc\b", lower_note) is not None
    ):
        if "active infection" in lower_note or "sepsis" in lower_note or "bacteremia" in lower_note:
            return Reject(reason="Active infection exclusion in oncology support trial.")

        platelet = extract_lab_value(note, "Platelet")
        anc = extract_lab_value(note, "ANC")

        if platelet is None:
            return RequestLabResult(test_name="Platelet")
        if anc is None:
            return RequestLabResult(test_name="ANC")
        if platelet > 150000 and anc > 1500:
            trial_id = choose_available_trial(observation, ("TRIAL_I",))
            if trial_id is None:
                return Reject(reason="No oncology support trial available.")
            return Enroll(trial_id=trial_id)
        return Reject(reason="Hematology values below oncology trial threshold.")

    if "bp" in lower_note or "lisinopril" in lower_note or "losartan" in lower_note:
        trial_id = choose_available_trial(observation, ("TRIAL_B",))
        if trial_id is not None:
            return Enroll(trial_id=trial_id)

    if "atorvastatin" in lower_note or "rosuvastatin" in lower_note:
        trial_id = choose_available_trial(observation, ("TRIAL_D", "TRIAL_G"))
        if trial_id is not None:
            return Enroll(trial_id=trial_id)

    for trial in observation.available_trials:
        trial_lower = trial.trial_id.lower()
        if trial_lower in lower_note:
            return Enroll(trial_id=trial.trial_id)

    if observation.available_trials:
        return Enroll(trial_id=observation.available_trials[0].trial_id)

    return Reject(reason="No trial options available.")


def build_user_prompt(observation: Observation, task_id: str) -> str:
    trials_text = "\n".join(
        [
            f"- {trial.trial_id}: include=({trial.inclusion_criteria}) exclude=({trial.exclusion_criteria})"
            for trial in observation.available_trials
        ]
    )

    return textwrap.dedent(
        f"""
        task_id: {task_id}
        patients_remaining: {observation.patients_remaining}

        active_patient_ehr:
        {observation.active_patient_ehr}

        available_trials:
        {trials_text}

        Return exactly one JSON action.
        """
    ).strip()


def model_action(client: OpenAI | None, observation: Observation, task_id: str) -> tuple[Action, str | None]:
    if client is None:
        return heuristic_action(observation), "missing_api_key"

    prompt = build_user_prompt(observation=observation, task_id=task_id)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (response.choices[0].message.content or "").strip()
        payload = json.loads(extract_json_object(content))
        action = parse_action(payload)
        return action, None
    except Exception as exc:
        return heuristic_action(observation), f"llm_fallback:{exc}"


def run_task(client: OpenAI | None, env: ClinicalTrialMatcherEnv, task_id: str) -> None:
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_id)
        done = False

        while not done and steps < MAX_STEPS_PER_TASK:
            action, action_error = model_action(client=client, observation=observation, task_id=task_id)
            observation, reward, done, info = env.step(action)

            steps += 1
            rewards.append(reward)
            log_step(
                step=steps,
                action=compact_action(action),
                reward=reward,
                done=done,
                error=action_error,
            )

            score = float(info.get("normalized_score", 0.0))

        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        success = False
        log_step(
            step=steps + 1,
            action='{"action_type":"error"}',
            reward=0.0,
            done=True,
            error=str(exc),
        )
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    client: OpenAI | None = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = ClinicalTrialMatcherEnv()
    for task_id in TASKS:
        run_task(client=client, env=env, task_id=task_id)


if __name__ == "__main__":
    main()
