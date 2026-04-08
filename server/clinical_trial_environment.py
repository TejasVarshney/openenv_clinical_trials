"""OpenEnv Environment adapter around the core Clinical Trial Matcher environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.env import ClinicalTrialMatcherEnv
from env.models import Enroll, Reject, RequestLabResult

from .api_models import (
    ClinicalTrialMatcherAction,
    ClinicalTrialMatcherObservation,
    TrialInfoPayload,
)


class ClinicalTrialEnvironment(
    Environment[ClinicalTrialMatcherAction, ClinicalTrialMatcherObservation, State]
):
    """HTTP adapter that exposes the gym-like environment through OpenEnv server interfaces."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._env = ClinicalTrialMatcherEnv()
        self._state = State(episode_id=self._env.state()["episode_id"], step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ClinicalTrialMatcherObservation:
        task_id = kwargs.get("task") or kwargs.get("task_id")
        observation = self._env.reset(task_id=task_id)
        snapshot = self._env.state()

        self._state = State(
            episode_id=str(snapshot["episode_id"]),
            step_count=int(snapshot["step_count"]),
            task_id=str(snapshot["task_id"]),
            patients_remaining=int(snapshot["patients_remaining"]),
            normalized_score=float(snapshot["normalized_score"]),
        )

        return self._to_server_observation(
            observation=observation,
            reward=0.0,
            done=False,
            reward_reason="Episode reset.",
            metadata={"task_id": self._env.task_id},
        )

    def step(
        self,
        action: ClinicalTrialMatcherAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ClinicalTrialMatcherObservation:
        if action.action_type == "enroll":
            internal_action = Enroll(trial_id=action.trial_id or "")
        elif action.action_type == "reject":
            internal_action = Reject(reason=action.reason or "No reason provided.")
        else:
            internal_action = RequestLabResult(test_name=action.test_name or "")

        observation, reward, done, info = self._env.step(internal_action)
        snapshot = self._env.state()

        self._state = State(
            episode_id=str(snapshot["episode_id"]),
            step_count=int(snapshot["step_count"]),
            task_id=str(snapshot["task_id"]),
            patients_remaining=int(snapshot["patients_remaining"]),
            normalized_score=float(snapshot["normalized_score"]),
        )

        return self._to_server_observation(
            observation=observation,
            reward=reward,
            done=done,
            reward_reason=str(info.get("reward_reason", "")),
            metadata=info,
        )

    @property
    def state(self) -> State:
        snapshot = self._env.state()
        return State(
            episode_id=str(snapshot["episode_id"]),
            step_count=int(snapshot["step_count"]),
            task_id=str(snapshot["task_id"]),
            patient_index=int(snapshot["patient_index"]),
            patients_remaining=int(snapshot["patients_remaining"]),
            cumulative_reward=float(snapshot["cumulative_reward"]),
            normalized_score=float(snapshot["normalized_score"]),
            done=bool(snapshot["done"]),
        )

    def close(self) -> None:
        return

    @staticmethod
    def _to_server_observation(
        observation: Any,
        reward: float,
        done: bool,
        reward_reason: str,
        metadata: dict[str, object],
    ) -> ClinicalTrialMatcherObservation:
        trial_payloads = [
            TrialInfoPayload(
                trial_id=trial.trial_id,
                inclusion_criteria=trial.inclusion_criteria,
                exclusion_criteria=trial.exclusion_criteria,
            )
            for trial in observation.available_trials
        ]

        return ClinicalTrialMatcherObservation(
            active_patient_ehr=observation.active_patient_ehr,
            available_trials=trial_payloads,
            patients_remaining=observation.patients_remaining,
            done=done,
            reward=reward,
            reward_reason=reward_reason,
            metadata=metadata,
        )
