"""Pydantic models for the Clinical Trial Matcher OpenEnv environment."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class TrialInfo(BaseModel):
    """Metadata describing an active clinical trial available to the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the trial.",
    )
    inclusion_criteria: str = Field(
        ...,
        min_length=1,
        description="Human-readable inclusion criteria text.",
    )
    exclusion_criteria: str = Field(
        ...,
        min_length=1,
        description="Human-readable exclusion criteria text.",
    )

    @field_validator("trial_id")
    @classmethod
    def normalize_trial_id(cls, value: str) -> str:
        """Normalize trial identifiers while preserving deterministic matching."""
        return value.strip().upper()


class Observation(BaseModel):
    """Observation presented to the agent at each environment step."""

    model_config = ConfigDict(extra="forbid")

    active_patient_ehr: str = Field(
        ...,
        min_length=0,
        description="Unstructured EHR summary for the active patient.",
    )
    available_trials: list[TrialInfo] = Field(
        ...,
        min_length=1,
        description="Full pool of trials for this task (not pre-filtered for eligibility).",
    )
    patients_remaining: int = Field(
        ...,
        ge=0,
        description="Number of patients left in the queue including the active one.",
    )


class Enroll(BaseModel):
    """Action to enroll the active patient into a selected clinical trial."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["enroll"] = Field(
        default="enroll",
        description="Discriminator for the enrollment action.",
    )
    trial_id: str = Field(
        ...,
        min_length=1,
        description="Trial identifier chosen by the agent.",
    )

    @field_validator("trial_id")
    @classmethod
    def normalize_trial_id(cls, value: str) -> str:
        return value.strip().upper()


class Reject(BaseModel):
    """Action to reject the active patient from all available trials."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["reject"] = Field(
        default="reject",
        description="Discriminator for the rejection action.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Short rationale for rejection.",
    )


class RequestLabResult(BaseModel):
    """Action to reveal hidden lab data for the active patient without advancing the queue."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["request_lab"] = Field(
        default="request_lab",
        description="Discriminator for lab-request action.",
    )
    test_name: str = Field(
        ...,
        min_length=1,
        description="Lab test to request, for example HbA1c, eGFR, LDL.",
    )

    @field_validator("test_name")
    @classmethod
    def normalize_test_name(cls, value: str) -> str:
        return value.strip()


Action = Enroll | Reject | RequestLabResult
ActionSchema = Annotated[Action, Field(discriminator="action_type")]
_ACTION_ADAPTER = TypeAdapter(ActionSchema)


class Reward(BaseModel):
    """Reward payload for deterministic grading with human-readable rationale."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Reward scalar in the closed range [-1.0, +1.0].",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of why the reward was assigned.",
    )


class PatientRecord(BaseModel):
    """Patient entry used by task datasets."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier.",
    )
    ehr: str = Field(
        ...,
        min_length=1,
        description="Base EHR note shown before optional lab requests.",
    )
    hidden_labs: dict[str, str] = Field(
        default_factory=dict,
        description="Labs that can be revealed through request_lab actions.",
    )
    required_labs: list[str] = Field(
        default_factory=list,
        description="Labs that must be present before safe enrollment decisions.",
    )

    @field_validator("patient_id")
    @classmethod
    def normalize_patient_id(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def dedupe_required_labs(self) -> "PatientRecord":
        seen: set[str] = set()
        normalized: list[str] = []
        for name in self.required_labs:
            clean_name = name.strip()
            if clean_name and clean_name.lower() not in seen:
                seen.add(clean_name.lower())
                normalized.append(clean_name)
        self.required_labs = normalized
        return self


class TaskData(BaseModel):
    """Task definition loaded from JSON for one OpenEnv difficulty level."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        ...,
        min_length=1,
        description="Stable task identifier used in resets and scoring.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable task summary.",
    )
    trials: list[TrialInfo] = Field(
        ...,
        min_length=1,
        description="Trials available for this task.",
    )
    patients: list[PatientRecord] = Field(
        ...,
        min_length=1,
        description="Ordered patient queue for deterministic episodes.",
    )

    @model_validator(mode="after")
    def ensure_unique_patients(self) -> "TaskData":
        ids = [patient.patient_id for patient in self.patients]
        if len(ids) != len(set(ids)):
            raise ValueError("Patient IDs must be unique within a task.")
        return self


def parse_action(value: Action | dict[str, object]) -> Action:
    """Parse incoming action payloads into the strict Action union type."""
    if isinstance(value, (Enroll, Reject, RequestLabResult)):
        return value
    return _ACTION_ADAPTER.validate_python(value)
