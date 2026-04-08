"""Deterministic grading logic for all Clinical Trial Matcher tasks."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Action, Enroll, Reject, RequestLabResult, Reward


class ClinicalTrialGrader:
    """Grades actions deterministically from the hidden ground truth mapping."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._ground_truth = self._load_ground_truth(data_dir / "ground_truth.json")

    @staticmethod
    def _load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        truth: dict[str, dict[str, str]] = {}

        for task_id, mapping in raw_data.items():
            if not isinstance(mapping, dict):
                raise ValueError("Each task entry in ground_truth.json must be an object.")
            task_map: dict[str, str] = {}
            for patient_id, label in mapping.items():
                if not isinstance(label, str):
                    raise ValueError("Ground-truth labels must be strings.")
                task_map[patient_id.strip().upper()] = label.strip().upper()
            truth[task_id] = task_map

        return truth

    def expected_label(self, task_id: str, patient_id: str) -> str:
        """Return the hidden correct label for a patient in a task."""
        task_map = self._ground_truth.get(task_id)
        if task_map is None:
            raise KeyError(f"Unknown task_id '{task_id}' in grader.")

        key = patient_id.strip().upper()
        if key not in task_map:
            raise KeyError(f"Unknown patient_id '{patient_id}' for task '{task_id}'.")
        return task_map[key]

    def grade(
        self,
        task_id: str,
        patient_id: str,
        action: Action,
        *,
        lab_already_present: bool = False,
        missing_required_labs: bool = False,
    ) -> Reward:
        """Assign deterministic step reward and reason."""
        expected = self.expected_label(task_id=task_id, patient_id=patient_id)

        match action:
            case RequestLabResult(test_name=test_name):
                if lab_already_present:
                    return Reward(
                        value=-0.1,
                        reason=(
                            f"Unnecessary RequestLabResult for '{test_name}' because the value "
                            "was already visible in the EHR."
                        ),
                    )
                return Reward(
                    value=0.0,
                    reason=(
                        f"Requested lab '{test_name}'. Decision deferred until more evidence is visible."
                    ),
                )

            case Enroll(trial_id=trial_id):
                if missing_required_labs:
                    return Reward(
                        value=-1.0,
                        reason=(
                            "Enrollment attempted before required labs were available. "
                            "Blind enrollment carries a heavy penalty."
                        ),
                    )

                if trial_id == expected:
                    return Reward(
                        value=1.0,
                        reason=f"Correct enrollment: patient matched to {trial_id}.",
                    )

                if expected == "REJECT":
                    return Reward(
                        value=-1.0,
                        reason=(
                            "Enrollment violates exclusion criteria for this patient. "
                            "Correct label is reject."
                        ),
                    )

                return Reward(
                    value=-0.4,
                    reason=(
                        f"Incorrect enrollment to {trial_id}. Expected enrollment is {expected}."
                    ),
                )

            case Reject(reason=reason):
                if expected == "REJECT":
                    return Reward(
                        value=1.0,
                        reason=f"Correct rejection: {reason}",
                    )

                return Reward(
                    value=-0.4,
                    reason=(
                        f"Incorrect rejection. Patient should be enrolled in {expected}."
                    ),
                )

        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    @staticmethod
    def normalize_score(total_reward: float, max_possible_reward: float) -> float:
        """Normalize cumulative episode reward into [0.0, 1.0]."""
        if max_possible_reward <= 0:
            return 0.0
        raw = total_reward / max_possible_reward
        return max(0.0, min(1.0, raw))
