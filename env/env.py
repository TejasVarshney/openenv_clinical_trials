"""Core Clinical Trial Matcher environment implementing reset/step/state."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast
from uuid import uuid4

from .grader import ClinicalTrialGrader
from .models import (
    Action,
    Enroll,
    Observation,
    PatientRecord,
    Reject,
    RequestLabResult,
    TaskData,
    TrialInfo,
    parse_action,
)


class ClinicalTrialMatcherEnv:
    """Deterministic clinical-trial matching environment for OpenEnv tasks."""

    _TASK_FILES: tuple[str, ...] = (
        "task1_data.json",
        "task2_data.json",
        "task3_data.json",
    )

    def __init__(self, task_id: str = "task1_easy_explicit", data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or (Path(__file__).resolve().parent / "data")
        self._tasks = self._load_tasks(self._data_dir)
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {sorted(self._tasks)}")

        self._task_id = task_id
        self._task = self._tasks[self._task_id]
        self._grader = ClinicalTrialGrader(self._data_dir)

        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._patient_index: int = 0
        self._cumulative_reward: float = 0.0
        self._reward_history: list[float] = []
        self._visible_ehr_by_patient: dict[str, str] = {}
        self._done: bool = False

        self.reset(task_id=task_id)

    @property
    def task_id(self) -> str:
        """Return the currently active task identifier."""
        return self._task_id

    def reset(self, task_id: str | None = None) -> Observation:
        """Reset environment state and return the first observation."""
        if task_id is not None:
            if task_id not in self._tasks:
                raise ValueError(f"Unknown task_id '{task_id}'. Available: {sorted(self._tasks)}")
            self._task_id = task_id
            self._task = self._tasks[self._task_id]

        self._episode_id = str(uuid4())
        self._step_count = 0
        self._patient_index = 0
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._done = False
        self._visible_ehr_by_patient = {
            patient.patient_id: patient.ehr for patient in self._task.patients
        }

        return self._build_observation(done=False)

    def step(self, action_input: Action | dict[str, object]) -> tuple[Observation, float, bool, dict[str, object]]:
        """Execute one environment step and return (observation, reward, done, info)."""
        action = parse_action(action_input)

        if self._done:
            terminal_observation = self._build_observation(done=True)
            info = {
                "task_id": self._task_id,
                "episode_id": self._episode_id,
                "reason": "Episode already completed. Call reset() to start a new episode.",
                "normalized_score": self.normalized_score,
            }
            return terminal_observation, 0.0, True, info

        patient = self._current_patient()
        visible_ehr = self._visible_ehr_by_patient[patient.patient_id]
        info: dict[str, object] = {
            "task_id": self._task_id,
            "episode_id": self._episode_id,
            "patient_id": patient.patient_id,
            "step_count": self._step_count + 1,
        }

        reward_value: float
        reward_reason: str

        if isinstance(action, RequestLabResult):
            already_present = self._has_lab_value(visible_ehr, action.test_name)

            if already_present:
                reward_obj = self._grader.grade(
                    task_id=self._task_id,
                    patient_id=patient.patient_id,
                    action=action,
                    lab_already_present=True,
                )
            else:
                hidden_value = self._lookup_hidden_lab(patient=patient, test_name=action.test_name)
                if hidden_value is None:
                    reward_obj = self._grader.grade(
                        task_id=self._task_id,
                        patient_id=patient.patient_id,
                        action=action,
                        lab_already_present=True,
                    )
                    reward_obj.reason = (
                        f"Requested lab '{action.test_name}' is unavailable for this patient."
                    )
                else:
                    self._visible_ehr_by_patient[patient.patient_id] = (
                        f"{visible_ehr}\nLab {action.test_name}: {hidden_value}"
                    )
                    reward_obj = self._grader.grade(
                        task_id=self._task_id,
                        patient_id=patient.patient_id,
                        action=action,
                        lab_already_present=False,
                    )

            reward_value = reward_obj.value
            reward_reason = reward_obj.reason
            done = False
            observation = self._build_observation(done=False)
            info["advanced_queue"] = False

        else:
            missing_required_labs = self._missing_required_labs(patient=patient, visible_ehr=visible_ehr)
            reward_obj = self._grader.grade(
                task_id=self._task_id,
                patient_id=patient.patient_id,
                action=cast(Enroll | Reject, action),
                missing_required_labs=bool(missing_required_labs),
            )

            reward_value = reward_obj.value
            reward_reason = reward_obj.reason

            self._patient_index += 1
            done = self._patient_index >= len(self._task.patients)
            self._done = done

            observation = self._build_observation(done=done)
            info["advanced_queue"] = True
            info["missing_required_labs"] = missing_required_labs

        self._step_count += 1
        self._cumulative_reward += reward_value
        self._reward_history.append(reward_value)

        info["reward_reason"] = reward_reason
        info["cumulative_reward"] = round(self._cumulative_reward, 6)
        info["normalized_score"] = self.normalized_score

        return observation, reward_value, done, info

    def state(self) -> dict[str, object]:
        """Return serializable environment state for debugging and grading."""
        return {
            "task_id": self._task_id,
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "patient_index": self._patient_index,
            "patients_total": len(self._task.patients),
            "patients_remaining": self._patients_remaining(),
            "cumulative_reward": round(self._cumulative_reward, 6),
            "normalized_score": self.normalized_score,
            "done": self._done,
            "reward_history": [round(value, 6) for value in self._reward_history],
        }

    @property
    def normalized_score(self) -> float:
        """Normalized cumulative score clamped to [0.0, 1.0]."""
        return self._grader.normalize_score(
            total_reward=self._cumulative_reward,
            max_possible_reward=float(len(self._task.patients)),
        )

    @staticmethod
    def _load_tasks(data_dir: Path) -> dict[str, TaskData]:
        tasks: dict[str, TaskData] = {}

        for filename in ClinicalTrialMatcherEnv._TASK_FILES:
            task_path = data_dir / filename
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            task = TaskData.model_validate(payload)
            tasks[task.task_id] = task

        if len(tasks) != 3:
            raise ValueError("Exactly three task definitions are required.")

        return tasks

    def _current_patient(self) -> PatientRecord:
        if self._patient_index >= len(self._task.patients):
            raise IndexError("No active patient. The episode has already ended.")
        return self._task.patients[self._patient_index]

    def _patients_remaining(self) -> int:
        if self._done:
            return 0
        return len(self._task.patients) - self._patient_index

    def _build_observation(self, done: bool) -> Observation:
        trials: list[TrialInfo] = self._task.trials

        if done:
            return Observation(
                active_patient_ehr="",
                available_trials=trials,
                patients_remaining=0,
            )

        patient = self._current_patient()
        ehr = self._visible_ehr_by_patient[patient.patient_id]

        return Observation(
            active_patient_ehr=ehr,
            available_trials=trials,
            patients_remaining=self._patients_remaining(),
        )

    def _lookup_hidden_lab(self, patient: PatientRecord, test_name: str) -> str | None:
        target = test_name.strip().lower()
        for name, value in patient.hidden_labs.items():
            if name.strip().lower() == target:
                return value
        return None

    @staticmethod
    def _has_lab_value(ehr_text: str, test_name: str) -> bool:
        escaped = re.escape(test_name.strip())
        pattern = rf"\b{escaped}\b\s*[:=]?\s*[<>]?[\d]"
        return re.search(pattern, ehr_text, flags=re.IGNORECASE) is not None

    def _missing_required_labs(self, patient: PatientRecord, visible_ehr: str) -> list[str]:
        missing: list[str] = []
        for test_name in patient.required_labs:
            if not self._has_lab_value(visible_ehr, test_name):
                missing.append(test_name)
        return missing
