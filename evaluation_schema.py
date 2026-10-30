"""One acceptance policy for returned and cached Evaluator assessments."""

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    approved: StrictBool = Field(
        description="True only when the answer satisfies the question and evidence."
    )
    feedback: str = Field(
        min_length=1,
        description="Explain the decision and any specific corrections needed.",
    )


def validate_evaluation_response(message, generation_info=None) -> EvaluationResult:
    """Validate untouched JSON and completion metadata, without JSON repair."""
    if message is None or not isinstance(message.content, str):
        raise ValueError("Evaluator must return a JSON text response.")
    metadata_sources = [
        dict(generation_info or {}),
        dict(getattr(message, "response_metadata", {}) or {}),
    ]
    finish_reasons = [
        source["finish_reason"] for source in metadata_sources
        if source.get("finish_reason") is not None
    ]
    if not finish_reasons or any(reason != "stop" for reason in finish_reasons):
        raise ValueError("Evaluator completion was unsuccessful or unfinished.")
    bad_native_reasons = {
        "length", "max_tokens", "max_output_tokens", "token_limit",
        "content_filter", "error",
    }
    if any(
        str(source.get("native_finish_reason", "")).lower() in bad_native_reasons
        for source in metadata_sources
    ):
        raise ValueError("Evaluator provider reported an incomplete assessment.")
    if (getattr(message, "additional_kwargs", {}) or {}).get("refusal"):
        raise ValueError("Evaluator refused to provide an assessment.")
    return EvaluationResult.model_validate_json(message.content, strict=True)


def validate_evaluation_generation(generation) -> EvaluationResult:
    return validate_evaluation_response(
        generation.message,
        getattr(generation, "generation_info", None),
    )
