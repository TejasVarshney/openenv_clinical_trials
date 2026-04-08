"""Tests for deterministic grading behavior."""

from __future__ import annotations

from pathlib import Path

from env.grader import ClinicalTrialGrader
from env.models import Enroll, Reject, RequestLabResult


DATA_DIR = Path(__file__).resolve().parents[1] / "env" / "data"


def test_correct_enrollment_reward_is_positive() -> None:
    grader = ClinicalTrialGrader(DATA_DIR)
    reward = grader.grade(
        task_id="task1_easy_explicit",
        patient_id="T1P001",
        action=Enroll(trial_id="TRIAL_A"),
    )
    assert reward.value == 1.0


def test_wrong_reject_penalty_is_partial() -> None:
    grader = ClinicalTrialGrader(DATA_DIR)
    reward = grader.grade(
        task_id="task2_medium_ontology",
        patient_id="T2P001",
        action=Reject(reason="Not sure"),
    )
    assert reward.value == -0.4


def test_unnecessary_lab_request_penalty() -> None:
    grader = ClinicalTrialGrader(DATA_DIR)
    reward = grader.grade(
        task_id="task3_hard_incomplete_data",
        patient_id="T3P021",
        action=RequestLabResult(test_name="HbA1c"),
        lab_already_present=True,
    )
    assert reward.value == -0.1


def test_score_normalization_clamps() -> None:
    assert ClinicalTrialGrader.normalize_score(100.0, 10.0) == 1.0
    assert ClinicalTrialGrader.normalize_score(-100.0, 10.0) == 0.0
