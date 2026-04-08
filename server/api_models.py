"""OpenEnv HTTP action/observation models for the Clinical Trial Matcher server."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrialInfoPayload(BaseModel):
    """Serializable trial payload for server responses."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(..., description="Trial identifier.")
    inclusion_criteria: str = Field(..., description="Inclusion criteria text.")
    exclusion_criteria: str = Field(..., description="Exclusion criteria text.")


class ClinicalTrialMatcherAction(Action):
    """Action schema exposed by the HTTP server."""

    action_type: Literal["enroll", "reject", "request_lab"] = Field(
        ...,
        description="Action discriminator.",
    )
    trial_id: str | None = Field(
        default=None,
        description="Target trial identifier when action_type is enroll.",
    )
    reason: str | None = Field(
        default=None,
        description="Rejection rationale when action_type is reject.",
    )
    test_name: str | None = Field(
        default=None,
        description="Lab test requested when action_type is request_lab.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "ClinicalTrialMatcherAction":
        if self.action_type == "enroll" and not self.trial_id:
            raise ValueError("trial_id is required when action_type='enroll'.")
        if self.action_type == "reject" and not self.reason:
            raise ValueError("reason is required when action_type='reject'.")
        if self.action_type == "request_lab" and not self.test_name:
            raise ValueError("test_name is required when action_type='request_lab'.")
        return self


class ClinicalTrialMatcherObservation(Observation):
    """Observation schema returned by HTTP and WebSocket endpoints."""

    active_patient_ehr: str = Field(
        ..., description="Current patient EHR note visible to the agent."
    )
    available_trials: list[TrialInfoPayload] = Field(
        ...,
        description="All trials available for the active task.",
    )
    patients_remaining: int = Field(
        ...,
        ge=0,
        description="Patients remaining in queue including current patient.",
    )
    reward_reason: str = Field(
        default="",
        description="Explanation for the most recent reward value.",
    )
