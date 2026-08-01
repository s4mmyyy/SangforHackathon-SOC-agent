"""统一、无领域依赖的 LLM 结构化输出解析、调用和安全审计。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Generic, Optional, TypeVar, Union

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)
LocationPart = Union[str, int]


class LLMOutputErrorCode(str, Enum):
    CLIENT_PROTOCOL_INVALID = "LLM_CLIENT_PROTOCOL_INVALID"
    CALL_FAILED = "LLM_CALL_FAILED"
    EMPTY = "LLM_OUTPUT_EMPTY"
    PARSE_FAILED = "LLM_OUTPUT_PARSE_FAILED"
    AMBIGUOUS = "LLM_OUTPUT_AMBIGUOUS"
    SCHEMA_INVALID = "LLM_OUTPUT_SCHEMA_INVALID"
    REFUSED = "LLM_OUTPUT_REFUSED"


@dataclass(frozen=True)
class LLMValidationIssue:
    location: tuple[LocationPart, ...]
    error_type: str


@dataclass(frozen=True)
class LLMOutputFailure:
    code: LLMOutputErrorCode
    exception_type: Optional[str] = None
    validation_issues: tuple[LLMValidationIssue, ...] = ()


@dataclass(frozen=True)
class LLMOutputAudit:
    interface: str
    schema_name: str
    parse_source: Optional[str] = None
    output_sha256: Optional[str] = None
    fallback_used: bool = False
    failure_code: Optional[LLMOutputErrorCode] = None
    exception_type: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass(frozen=True)
class StructuredOutputResult(Generic[T]):
    audit: LLMOutputAudit
    value: Optional[T] = None
    failure: Optional[LLMOutputFailure] = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.failure is None


class StructuredOutputError(Exception):
    """只暴露稳定错误码；详细原文和供应方异常不会进入异常消息。"""

    def __init__(self, result: StructuredOutputResult[Any]):
        self.result = result
        failure = getattr(result, "failure", None)
        code = (
            failure.code
            if isinstance(failure, LLMOutputFailure)
            and isinstance(failure.code, LLMOutputErrorCode)
            else LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID
        )
        super().__init__(code.value)

    def __str__(self) -> str:
        failure = getattr(self.result, "failure", None)
        if not isinstance(failure, LLMOutputFailure):
            return LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID.value
        code = failure.code
        return (
            code.value
            if isinstance(code, LLMOutputErrorCode)
            else LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID.value
        )


@dataclass(frozen=True)
class _CandidateFailure(Exception):
    code: LLMOutputErrorCode
    exception_type: Optional[str] = None
    validation_issues: tuple[LLMValidationIssue, ...] = ()


_MISSING = object()
_WRAPPER_KEYS = {"raw", "parsed", "parsing_error"}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_SAFE_LOCATION = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_FENCED_JSON = re.compile(
    r"\s*```(?:json)?[ \t]*(?:\r?\n)?(?P<body>.*?)```\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


@dataclass(frozen=True)
class _ToolCandidate:
    name: Optional[str]
    args: Any
    call_id: Optional[str]
    invalid: bool
    source: str


def _clean_identifier(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None


def _clean_exception_type(value: Any) -> Optional[str]:
    return _clean_identifier(value)


def _clean_error_type(value: Any) -> str:
    return _clean_identifier(value) or "validation_error"


def _clean_location_part(value: Any) -> LocationPart:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _SAFE_LOCATION.fullmatch(value):
        return value
    return "<field>"


def _sanitize_validation_issues(issues: Any) -> tuple[LLMValidationIssue, ...]:
    if not isinstance(issues, (list, tuple)):
        return ()
    cleaned: list[LLMValidationIssue] = []
    for issue in issues[:50]:
        if not isinstance(issue, LLMValidationIssue):
            continue
        error_type = _clean_error_type(issue.error_type)
        raw_location = issue.location if isinstance(issue.location, (list, tuple)) else ()
        location = tuple(_clean_location_part(part) for part in raw_location[:10])
        if error_type == "extra_forbidden" and location:
            location = (*location[:-1], "<extra>")
        cleaned.append(LLMValidationIssue(location=location, error_type=error_type))
    return tuple(cleaned)


def _schema_name(schema: type[BaseModel]) -> str:
    name = getattr(schema, "__name__", None)
    return name if isinstance(name, str) and name else type(schema).__name__


def _model_payload(value: BaseModel) -> Any:
    try:
        return value.model_dump(mode="python", by_alias=True, round_trip=True)
    except Exception as exc:
        raise _CandidateFailure(
            LLMOutputErrorCode.PARSE_FAILED,
            exception_type=_clean_exception_type(type(exc).__name__),
        )


def _hashable_value(value: Any) -> Any:
    return _model_payload(value) if isinstance(value, BaseModel) else value


def _stable_output_hash(value: Any) -> str:
    """只返回稳定摘要；调用方从不保存被哈希的正文。"""
    try:
        rendered = json.dumps(
            _hashable_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        rendered = json.dumps({"unserializable_type": type(value).__name__}, sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validation_issues(exc: ValidationError) -> tuple[LLMValidationIssue, ...]:
    try:
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:
        try:
            errors = exc.errors()
        except Exception:
            return ()
    except Exception:
        return ()
    issues: list[LLMValidationIssue] = []
    for error in errors[:50]:
        error_type = _clean_error_type(error.get("type"))
        raw_location = error.get("loc", ())
        if not isinstance(raw_location, (list, tuple)):
            raw_location = ()
        location = tuple(_clean_location_part(part) for part in raw_location[:10])
        if error_type == "extra_forbidden" and location:
            location = (*location[:-1], "<extra>")
        issues.append(LLMValidationIssue(location=location, error_type=error_type))
    return tuple(issues)


def _read_member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    try:
        return getattr(value, name, default)
    except Exception as exc:
        raise _CandidateFailure(
            LLMOutputErrorCode.PARSE_FAILED,
            exception_type=_clean_exception_type(type(exc).__name__),
        )


def _strict_json_loads(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError()
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise _NonFiniteNumberError()

    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _expected_tool_name(schema: type[BaseModel]) -> str:
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        converted = convert_to_openai_tool(schema)
        name = converted.get("function", {}).get("name")
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    try:
        title = schema.model_json_schema().get("title")
        if isinstance(title, str) and title:
            return title
    except Exception:
        pass
    return _schema_name(schema)


def _call_items(value: Any) -> list[Any]:
    if value is _MISSING or value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise _CandidateFailure(LLMOutputErrorCode.PARSE_FAILED, "TypeError")
    return list(value)


def _tool_candidate(item: Any, *, invalid: bool, source: str) -> _ToolCandidate:
    name = _read_member(item, "name", _MISSING)
    args = _read_member(item, "args", _MISSING)
    function = _read_member(item, "function", _MISSING)
    if function is not _MISSING and function is not None:
        if name is _MISSING:
            name = _read_member(function, "name", _MISSING)
        if args is _MISSING:
            args = _read_member(function, "arguments", _MISSING)
    call_id = _read_member(item, "id", None)
    return _ToolCandidate(
        name=name if isinstance(name, str) else None,
        args=args,
        call_id=call_id[:128] if isinstance(call_id, str) else None,
        invalid=invalid,
        source=source,
    )


def _tool_args_fingerprint(value: Any) -> str:
    try:
        normalized = _model_payload(value) if isinstance(value, BaseModel) else value
        if isinstance(normalized, str):
            try:
                normalized = _strict_json_loads(normalized)
            except (json.JSONDecodeError, _DuplicateKeyError, _NonFiniteNumberError):
                return "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        rendered = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        rendered = f"opaque:{type(value).__name__}:{id(value)}"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _mirrors(left: _ToolCandidate, right: _ToolCandidate) -> bool:
    same_payload = (
        left.name == right.name
        and _tool_args_fingerprint(left.args) == _tool_args_fingerprint(right.args)
    )
    if left.call_id and right.call_id:
        return left.call_id == right.call_id and same_payload
    return same_payload


def _extract_tool_calls(raw: Any) -> list[_ToolCandidate]:
    if raw is None:
        return []
    valid_items = _call_items(_read_member(raw, "tool_calls", _MISSING))
    invalid_items = _call_items(_read_member(raw, "invalid_tool_calls", _MISSING))
    standardized = [
        *[_tool_candidate(item, invalid=False, source="tool_calls") for item in valid_items],
        *[_tool_candidate(item, invalid=True, source="invalid_tool_calls") for item in invalid_items],
    ]

    additional_kwargs = _read_member(raw, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return standardized
    additional_items = _call_items(additional_kwargs.get("tool_calls", _MISSING))
    additional = [
        _tool_candidate(item, invalid=False, source="additional_kwargs.tool_calls")
        for item in additional_items
    ]
    function_call = additional_kwargs.get("function_call", _MISSING)
    if function_call is not _MISSING and function_call is not None:
        additional.append(_tool_candidate(
            {"function": function_call},
            invalid=False,
            source="additional_kwargs.function_call",
        ))
    if not standardized:
        return additional

    unmatched_standardized = list(range(len(standardized)))
    non_mirrored: list[_ToolCandidate] = []
    for call in additional:
        mirror_index = next((
            index for index in unmatched_standardized
            if _mirrors(standardized[index], call)
        ), None)
        if mirror_index is None:
            non_mirrored.append(call)
        else:
            unmatched_standardized.remove(mirror_index)
    return [*standardized, *non_mirrored]


def _refusal_present(raw: Any) -> bool:
    if raw is None:
        return False
    values = [_read_member(raw, "refusal", _MISSING)]
    additional_kwargs = _read_member(raw, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        values.append(additional_kwargs.get("refusal", _MISSING))
    for value in values:
        if value is _MISSING or value is None or value is False:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _message_like(value: Any, schema: type[BaseModel]) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in (
            "tool_calls", "invalid_tool_calls", "additional_kwargs", "refusal",
        ))
    if isinstance(value, schema) or isinstance(value, str) or value is None:
        return False
    if not isinstance(value, BaseModel):
        return True
    return any(
        _read_member(value, name, _MISSING) is not _MISSING
        for name in ("tool_calls", "invalid_tool_calls", "additional_kwargs", "refusal")
    )


def _parse_text_candidate(text: str) -> tuple[Any, bool]:
    if not text.strip():
        raise _CandidateFailure(LLMOutputErrorCode.EMPTY)
    try:
        return _strict_json_loads(text), False
    except _DuplicateKeyError:
        raise _CandidateFailure(LLMOutputErrorCode.AMBIGUOUS, "DuplicateKeyError")
    except (json.JSONDecodeError, _NonFiniteNumberError) as direct_exc:
        fenced = _FENCED_JSON.fullmatch(text)
        if fenced is None:
            raise _CandidateFailure(
                LLMOutputErrorCode.PARSE_FAILED,
                _clean_exception_type(type(direct_exc).__name__),
            )
        body = fenced.group("body")
        if not body.strip():
            raise _CandidateFailure(LLMOutputErrorCode.EMPTY)
        try:
            return _strict_json_loads(body), True
        except _DuplicateKeyError:
            raise _CandidateFailure(LLMOutputErrorCode.AMBIGUOUS, "DuplicateKeyError")
        except (json.JSONDecodeError, _NonFiniteNumberError) as fenced_exc:
            raise _CandidateFailure(
                LLMOutputErrorCode.PARSE_FAILED,
                _clean_exception_type(type(fenced_exc).__name__),
            )


def _reject_nonfinite(value: Any, seen: Optional[set[int]] = None) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _CandidateFailure(LLMOutputErrorCode.PARSE_FAILED, "NonFiniteNumberError")
    if not isinstance(value, (dict, list, tuple)):
        return
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    items = value.values() if isinstance(value, dict) else value
    for item in items:
        _reject_nonfinite(item, seen)


def _validate_candidate(candidate: Any, schema: type[T]) -> tuple[T, bool]:
    if candidate is _MISSING:
        raise _CandidateFailure(LLMOutputErrorCode.PARSE_FAILED, "ArgumentsMissing")
    if candidate is None:
        raise _CandidateFailure(LLMOutputErrorCode.EMPTY)
    fenced = False
    value = candidate
    if isinstance(value, str):
        value, fenced = _parse_text_candidate(value)
    elif isinstance(value, BaseModel):
        value = _model_payload(value)
    _reject_nonfinite(value)
    try:
        validated = schema.model_validate(value)
        _reject_nonfinite(_model_payload(validated))
        return validated, fenced
    except _CandidateFailure:
        raise
    except ValidationError as exc:
        raise _CandidateFailure(
            LLMOutputErrorCode.SCHEMA_INVALID,
            "ValidationError",
            _validation_issues(exc),
        )
    except Exception as exc:
        raise _CandidateFailure(
            LLMOutputErrorCode.PARSE_FAILED,
            _clean_exception_type(type(exc).__name__),
        )


def _result_failure(
    *,
    schema: type[T],
    interface: str,
    code: LLMOutputErrorCode,
    parse_source: Optional[str] = None,
    output_candidate: Any = _MISSING,
    fallback_used: bool = False,
    exception_type: Optional[str] = None,
    validation_issues: tuple[LLMValidationIssue, ...] = (),
    duration_ms: Optional[float] = None,
) -> StructuredOutputResult[T]:
    stable_code = code if isinstance(code, LLMOutputErrorCode) else LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID
    clean_exception = _clean_exception_type(exception_type)
    clean_issues = _sanitize_validation_issues(validation_issues)
    failure = LLMOutputFailure(
        code=stable_code,
        exception_type=clean_exception,
        validation_issues=clean_issues,
    )
    return StructuredOutputResult(
        failure=failure,
        audit=LLMOutputAudit(
            interface=interface,
            schema_name=_schema_name(schema),
            parse_source=parse_source,
            output_sha256=(
                _stable_output_hash(output_candidate)
                if output_candidate is not _MISSING
                else None
            ),
            fallback_used=fallback_used,
            failure_code=stable_code,
            exception_type=clean_exception,
            duration_ms=duration_ms,
        ),
    )


def _result_success(
    *,
    value: T,
    schema: type[T],
    interface: str,
    parse_source: str,
    output_candidate: Any,
    fallback_used: bool,
    duration_ms: Optional[float] = None,
) -> StructuredOutputResult[T]:
    return StructuredOutputResult(
        value=value,
        audit=LLMOutputAudit(
            interface=interface,
            schema_name=_schema_name(schema),
            parse_source=parse_source,
            output_sha256=_stable_output_hash(output_candidate),
            fallback_used=fallback_used,
            duration_ms=duration_ms,
        ),
    )


def _parse_one(
    candidate: Any,
    schema: type[T],
    *,
    interface: str,
    parse_source: str,
    fallback_used: bool,
    parser_exception_type: Optional[str] = None,
) -> StructuredOutputResult[T]:
    try:
        value, fenced = _validate_candidate(candidate, schema)
    except _CandidateFailure as failure:
        exception_type = failure.exception_type
        if parser_exception_type and failure.code not in {
            LLMOutputErrorCode.SCHEMA_INVALID,
            LLMOutputErrorCode.AMBIGUOUS,
        }:
            exception_type = parser_exception_type
        return _result_failure(
            schema=schema,
            interface=interface,
            code=failure.code,
            parse_source=parse_source,
            output_candidate=candidate,
            fallback_used=fallback_used,
            exception_type=exception_type,
            validation_issues=failure.validation_issues,
        )
    source = f"{parse_source}.fenced" if fenced else parse_source
    return _result_success(
        value=value,
        schema=schema,
        interface=interface,
        parse_source=source,
        output_candidate=candidate,
        fallback_used=fallback_used or fenced,
    )


def _validated_payload(value: BaseModel) -> Any:
    return _model_payload(value)


def _tool_result(
    calls: list[_ToolCandidate],
    schema: type[T],
    *,
    interface: str,
    parse_prefix: str,
    parsed_value: Optional[T] = None,
    invalid_allowed: bool = True,
    parser_exception_type: Optional[str] = None,
) -> Optional[StructuredOutputResult[T]]:
    if not calls:
        return None
    if len(calls) != 1:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.AMBIGUOUS,
            parse_source=f"{parse_prefix}.tool_calls",
            output_candidate=[call.args for call in calls],
            fallback_used=parsed_value is None,
            exception_type="MultipleToolCalls",
        )
    call = calls[0]
    if call.name != _expected_tool_name(schema):
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.PARSE_FAILED,
            parse_source=f"{parse_prefix}.{call.source}",
            output_candidate=call.args,
            fallback_used=parsed_value is None,
            exception_type="ToolNameMismatch",
        )
    if call.invalid and not invalid_allowed:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.PARSE_FAILED,
            parse_source=f"{parse_prefix}.{call.source}",
            output_candidate=call.args,
            exception_type="InvalidToolCall",
        )
    result = _parse_one(
        call.args,
        schema,
        interface=interface,
        parse_source=f"{parse_prefix}.{call.source}.arguments",
        fallback_used=parsed_value is None,
        parser_exception_type=parser_exception_type,
    )
    if not result.ok or parsed_value is None:
        return result
    try:
        matches = _validated_payload(result.value) == _validated_payload(parsed_value)
    except _CandidateFailure as failure:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=failure.code,
            parse_source=f"{parse_prefix}.{call.source}.arguments",
            exception_type=failure.exception_type,
        )
    if not matches:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.AMBIGUOUS,
            parse_source=f"{parse_prefix}.{call.source}.arguments",
            output_candidate=call.args,
            exception_type="ParsedArgumentsMismatch",
        )
    return None


def _parsing_error_result(
    result: StructuredOutputResult[T],
    parsing_error: Any,
    schema: type[T],
    *,
    interface: str,
) -> StructuredOutputResult[T]:
    if result.ok or parsing_error is None:
        return result
    if result.failure.code == LLMOutputErrorCode.AMBIGUOUS or result.failure.exception_type in {
        "ToolNameMismatch", "MultipleToolCalls", "InvalidToolCall",
    }:
        return result
    if isinstance(parsing_error, ValidationError):
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.SCHEMA_INVALID,
            parse_source=result.audit.parse_source,
            fallback_used=result.audit.fallback_used,
            exception_type="ValidationError",
            validation_issues=_validation_issues(parsing_error),
        )
    if result.failure.code == LLMOutputErrorCode.EMPTY:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.PARSE_FAILED,
            parse_source=result.audit.parse_source,
            fallback_used=result.audit.fallback_used,
            exception_type=_clean_exception_type(type(parsing_error).__name__),
        )
    return result


def _parse_structured_output(
    raw_output: Any,
    schema: type[T],
    *,
    interface: str,
) -> StructuredOutputResult[T]:
    if isinstance(raw_output, dict) and _WRAPPER_KEYS.issubset(raw_output):
        parsed = raw_output.get("parsed")
        parsing_error = raw_output.get("parsing_error")
        raw = raw_output.get("raw")
        if parsed is not None and parsing_error is not None:
            return _result_failure(
                schema=schema,
                interface=interface,
                code=LLMOutputErrorCode.AMBIGUOUS,
                parse_source="include_raw",
                exception_type="ParsedAndErrorPresent",
            )
        if _refusal_present(raw):
            return _result_failure(
                schema=schema,
                interface=interface,
                code=LLMOutputErrorCode.REFUSED,
                parse_source="include_raw.raw.refusal",
                output_candidate=raw,
                exception_type="ModelRefusal",
            )
        calls = _extract_tool_calls(raw)
        parser_exception_type = (
            _clean_exception_type(type(parsing_error).__name__)
            if parsing_error is not None
            else None
        )
        if parsed is not None:
            parsed_result = _parse_one(
                parsed,
                schema,
                interface=interface,
                parse_source="include_raw.parsed",
                fallback_used=False,
            )
            if not parsed_result.ok:
                return parsed_result
            tool_check = _tool_result(
                calls,
                schema,
                interface=interface,
                parse_prefix="include_raw.raw",
                parsed_value=parsed_result.value,
                invalid_allowed=False,
            )
            return tool_check or parsed_result
        tool_result = _tool_result(
            calls,
            schema,
            interface=interface,
            parse_prefix="include_raw.raw",
            parser_exception_type=parser_exception_type,
        )
        if tool_result is not None:
            return _parsing_error_result(tool_result, parsing_error, schema, interface=interface)
        content = _read_member(raw, "content", None)
        content_result = _parse_one(
            content,
            schema,
            interface=interface,
            parse_source="include_raw.raw.content",
            fallback_used=True,
            parser_exception_type=parser_exception_type,
        )
        return _parsing_error_result(content_result, parsing_error, schema, interface=interface)

    if _message_like(raw_output, schema):
        if _refusal_present(raw_output):
            return _result_failure(
                schema=schema,
                interface=interface,
                code=LLMOutputErrorCode.REFUSED,
                parse_source="message.refusal",
                output_candidate=raw_output,
                exception_type="ModelRefusal",
            )
        calls = _extract_tool_calls(raw_output)
        tool_result = _tool_result(
            calls,
            schema,
            interface=interface,
            parse_prefix="message",
        )
        if tool_result is not None:
            return tool_result
        content = _read_member(raw_output, "content", _MISSING)
        if content is not _MISSING:
            return _parse_one(
                content,
                schema,
                interface=interface,
                parse_source="message.content",
                fallback_used=False,
            )

    if isinstance(raw_output, schema):
        source = "direct_model"
    elif isinstance(raw_output, dict):
        source = "direct_dict"
    elif isinstance(raw_output, str):
        source = "string"
    elif isinstance(raw_output, BaseModel):
        source = "direct_model"
    elif raw_output is None:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.EMPTY,
            parse_source="direct",
        )
    else:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.PARSE_FAILED,
            parse_source="direct",
            output_candidate=raw_output,
            exception_type="TypeError",
        )
    return _parse_one(
        raw_output,
        schema,
        interface=interface,
        parse_source=source,
        fallback_used=False,
    )


def parse_structured_output(
    raw_output: Any,
    schema: type[T],
    *,
    interface: str,
) -> StructuredOutputResult[T]:
    """按固定优先级解析候选，并始终以 Pydantic schema 完成最终验证。"""
    try:
        return _parse_structured_output(raw_output, schema, interface=interface)
    except _CandidateFailure as failure:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=failure.code,
            parse_source="protocol",
            exception_type=failure.exception_type,
            validation_issues=failure.validation_issues,
        )
    except Exception as exc:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.PARSE_FAILED,
            parse_source="protocol",
            exception_type=_clean_exception_type(type(exc).__name__),
        )


def _stable_error_code(value: Any) -> Optional[LLMOutputErrorCode]:
    if isinstance(value, LLMOutputErrorCode):
        return value
    if isinstance(value, str):
        try:
            return LLMOutputErrorCode(value)
        except ValueError:
            return None
    return None


def _rebuild_client_result(
    response: StructuredOutputResult[Any],
    schema: type[T],
    *,
    interface: str,
    duration_ms: float,
) -> StructuredOutputResult[T]:
    try:
        value = response.value
        failure = response.failure
    except Exception as exc:
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
            parse_source="client_result",
            exception_type=_clean_exception_type(type(exc).__name__),
            duration_ms=duration_ms,
        )
    if value is not None and failure is None:
        result = _parse_one(
            value,
            schema,
            interface=interface,
            parse_source="client_result.value",
            fallback_used=False,
        )
        return replace(result, audit=replace(result.audit, duration_ms=duration_ms))
    if value is None and isinstance(failure, LLMOutputFailure):
        try:
            code = _stable_error_code(failure.code)
            exception_type = _clean_exception_type(failure.exception_type)
            validation_issues = _sanitize_validation_issues(failure.validation_issues)
        except Exception as exc:
            return _result_failure(
                schema=schema,
                interface=interface,
                code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
                parse_source="client_result.failure",
                exception_type=_clean_exception_type(type(exc).__name__),
                duration_ms=duration_ms,
            )
        if code is None:
            code = LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID
        return _result_failure(
            schema=schema,
            interface=interface,
            code=code,
            parse_source="client_result.failure",
            exception_type=exception_type,
            validation_issues=validation_issues,
            duration_ms=duration_ms,
        )
    return _result_failure(
        schema=schema,
        interface=interface,
        code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
        parse_source="client_result",
        exception_type="StructuredResultInvalid",
        duration_ms=duration_ms,
    )


def _exception_result(
    exc: Exception,
    schema: type[T],
    *,
    interface: str,
    duration_ms: Optional[float],
) -> StructuredOutputResult[T]:
    if isinstance(exc, StructuredOutputError):
        rebuilt = _rebuild_client_result(
            exc.result,
            schema,
            interface=interface,
            duration_ms=duration_ms or 0.0,
        )
        if not rebuilt.ok:
            return rebuilt
        return _result_failure(
            schema=schema,
            interface=interface,
            code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
            parse_source="structured_output_error",
            exception_type="StructuredOutputErrorInvalid",
            duration_ms=duration_ms,
        )
    if isinstance(exc, ValidationError):
        code = LLMOutputErrorCode.SCHEMA_INVALID
        issues = _validation_issues(exc)
    elif isinstance(exc, (json.JSONDecodeError, _DuplicateKeyError, _NonFiniteNumberError)) or type(exc).__name__ == "OutputParserException":
        code = LLMOutputErrorCode.PARSE_FAILED
        issues = ()
    else:
        code = LLMOutputErrorCode.CALL_FAILED
        issues = ()
    return _result_failure(
        schema=schema,
        interface=interface,
        code=code,
        exception_type=_clean_exception_type(type(exc).__name__),
        validation_issues=issues,
        duration_ms=duration_ms,
    )


def _with_duration(
    result: StructuredOutputResult[T],
    duration_ms: float,
) -> StructuredOutputResult[T]:
    return replace(result, audit=replace(result.audit, duration_ms=duration_ms))


def request_structured_output(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
) -> StructuredOutputResult[T]:
    """只选择一个客户端接口并调用一次，不在失败后降级为第二次模型请求。"""
    started = time.perf_counter()
    try:
        invoke_structured = getattr(llm_client, "invoke_structured", None)
        if callable(invoke_structured):
            interface = "invoke_structured"
            call = lambda: invoke_structured(system_prompt, user_prompt, schema)
        else:
            structured_chat = getattr(llm_client, "structured_chat", None)
            if callable(structured_chat):
                interface = "structured_chat"
                call = lambda: structured_chat(system_prompt, user_prompt, schema)
            else:
                chat = getattr(llm_client, "chat", None)
                if callable(chat):
                    interface = "chat"
                    call = lambda: chat(system_prompt, user_prompt)
                else:
                    return _result_failure(
                        schema=schema,
                        interface="unsupported",
                        code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
    except Exception as exc:
        return _result_failure(
            schema=schema,
            interface="protocol_inspection",
            code=LLMOutputErrorCode.CLIENT_PROTOCOL_INVALID,
            exception_type=_clean_exception_type(type(exc).__name__),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    try:
        response = call()
    except Exception as exc:
        return _exception_result(
            exc,
            schema,
            interface=interface,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    duration_ms = (time.perf_counter() - started) * 1000
    if isinstance(response, StructuredOutputResult):
        return _rebuild_client_result(
            response,
            schema,
            interface=interface,
            duration_ms=duration_ms,
        )
    parsed = parse_structured_output(response, schema, interface=interface)
    return _with_duration(parsed, duration_ms)


# 延迟依赖句柄也便于离线测试替换；模块导入本身不加载 LangChain。
ChatOpenAI: Any = None
SystemMessage: Any = None
HumanMessage: Any = None


@contextmanager
def _tracing_disabled():
    try:
        from langsmith import tracing_context
    except ImportError:
        yield
        return
    with tracing_context(enabled=False):
        yield


def _message_classes() -> tuple[Any, Any]:
    global SystemMessage, HumanMessage
    if SystemMessage is None or HumanMessage is None:
        try:
            from langchain_core.messages import HumanMessage as LangChainHumanMessage
            from langchain_core.messages import SystemMessage as LangChainSystemMessage
        except ImportError as exc:
            raise RuntimeError(
                "使用 ChatOpenAIAdapter 需要安装 langchain-openai 和 langchain-core。"
            ) from exc
        SystemMessage = LangChainSystemMessage
        HumanMessage = LangChainHumanMessage
    return SystemMessage, HumanMessage


def _chat_openai_class() -> Any:
    global ChatOpenAI
    if ChatOpenAI is None:
        try:
            from langchain_openai import ChatOpenAI as LangChainChatOpenAI
        except ImportError:
            return None
        ChatOpenAI = LangChainChatOpenAI
    return ChatOpenAI


class ChatOpenAIAdapter:
    """将 LangChain ChatOpenAI 适配为一次调用的安全结构化接口。"""

    def __init__(self, llm: Any):
        self.llm = llm

    @staticmethod
    def _messages(system_prompt: str, user_prompt: str) -> list[Any]:
        system_message, human_message = _message_classes()
        return [
            system_message(content=system_prompt),
            human_message(content=user_prompt),
        ]

    @staticmethod
    def _content_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        content = _read_member(response, "content", None)
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if not isinstance(text, str):
                    text = block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = self._messages(system_prompt, user_prompt)
        with _tracing_disabled():
            response = self.llm.invoke(messages)
        return self._content_text(response)

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> StructuredOutputResult[T]:
        started = time.perf_counter()
        try:
            messages = self._messages(system_prompt, user_prompt)
            runnable = self.llm.with_structured_output(
                schema,
                method="function_calling",
                include_raw=True,
            )
            with _tracing_disabled():
                response = runnable.invoke(messages)
        except Exception as exc:
            return _exception_result(
                exc,
                schema,
                interface="ChatOpenAIAdapter.invoke_structured",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        result = parse_structured_output(
            response,
            schema,
            interface="ChatOpenAIAdapter.invoke_structured",
        )
        return _with_duration(result, (time.perf_counter() - started) * 1000)

    def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> dict[str, Any]:
        result = self.invoke_structured(system_prompt, user_prompt, schema)
        if not result.ok:
            raise StructuredOutputError(result)
        return result.value.model_dump(mode="json")


def create_default_llm() -> Any:
    """按现有环境变量延迟创建 ChatOpenAI；缺配置或可选 SDK 时返回 None。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()

    model = os.getenv("LLM_MODEL_ID", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not model or not api_key:
        return None
    chat_openai = _chat_openai_class()
    if chat_openai is None:
        return None
    return chat_openai(
        model=model,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
        request_timeout=60,
        max_retries=3,
        extra_body={"thinking": {"type": "disabled"}},
    )


_create_default_llm = create_default_llm
