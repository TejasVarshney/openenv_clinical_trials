"""Behavioral tests for ClinicalTrialMatcherEnv."""

from __future__ import annotations

from env.env import ClinicalTrialMatcherEnv
from env.models import Enroll, RequestLabResult


def test_request_lab_does_not_advance_patient_queue() -> None:
    env = ClinicalTrialMatcherEnv(task_id="task3_hard_incomplete_data")
    observation = env.reset(task_id="task3_hard_incomplete_data")
    remaining_before = observation.patients_remaining

    next_observation, reward, done, info = env.step(RequestLabResult(test_name="HbA1c"))

    assert done is False
    assert info["advanced_queue"] is False
    assert next_observation.patients_remaining == remaining_before
    assert reward == 0.0


def test_redundant_request_lab_penalty() -> None:
    env = ClinicalTrialMatcherEnv(task_id="task3_hard_incomplete_data")
    env.reset(task_id="task3_hard_incomplete_data")

    env.step(RequestLabResult(test_name="HbA1c"))
    _, reward, _, _ = env.step(RequestLabResult(test_name="HbA1c"))

    assert reward == -0.1


def test_enroll_advances_queue() -> None:
    env = ClinicalTrialMatcherEnv(task_id="task1_easy_explicit")
    observation = env.reset(task_id="task1_easy_explicit")

    _, _, _, _ = env.step(Enroll(trial_id="TRIAL_A"))
    state = env.state()

    assert state["patient_index"] == 1
    assert state["patients_remaining"] == observation.patients_remaining - 1


def test_blind_enrollment_penalty_for_missing_required_lab() -> None:
    env = ClinicalTrialMatcherEnv(task_id="task3_hard_incomplete_data")
    env.reset(task_id="task3_hard_incomplete_data")

    _, reward, _, info = env.step(Enroll(trial_id="TRIAL_E"))

    assert reward == -1.0
    assert info["missing_required_labs"] == ["HbA1c"]


def test_normalized_score_bounds() -> None:
    env = ClinicalTrialMatcherEnv(task_id="task1_easy_explicit")
    env.reset(task_id="task1_easy_explicit")

    for _ in range(10):
        env.step(Enroll(trial_id="TRIAL_A"))

    score = env.normalized_score
    assert 0.0 <= score <= 1.0
