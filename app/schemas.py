"""Typed request/response models for the V-Vet symptom extraction contract."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignId(str, Enum):
    """The only sign IDs the API may ever return."""

    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


class FindingValue(str, Enum):
    """The only allowed values for a finding."""

    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"


class Finding(BaseModel):
    """One extracted sign with its value and verbatim evidence quote."""

    model_config = ConfigDict(extra="forbid")

    sign_id: SignId
    value: FindingValue
    quote: str = Field(min_length=1)


class ExtractRequest(BaseModel):
    """POST /extract request body."""

    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ExtractResponse(BaseModel):
    """POST /extract response body."""

    findings: list[Finding]
    escalate: bool
    reason: str
