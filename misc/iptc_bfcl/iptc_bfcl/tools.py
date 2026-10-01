import asyncio
import hashlib
import json
import keyword
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from jitgen_openai import CodeLanguage
from jitgen_openai.mcp_bridge import _bind_arguments

from iptc_bfcl.dataset import BfclEntry
from iptc_bfcl.metrics import RunRecorder, ToolCallRecord

_PROMPTS = Path(__file__).parent / "prompts"
# Appended to the `eval` tool's description: how to write the code, per language.
EVAL_CODE_GUIDANCE: dict[str, str] = {
    "python": (_PROMPTS / "eval_code.md").read_text(encoding="utf-8"),
    "javascript": (_PROMPTS / "eval_code_js.md").read_text(encoding="utf-8"),
}

# OpenAI's limit on function names (`^[a-zA-Z0-9_-]{1,64}$`).
MAX_NAME_LENGTH = 64
# Names a tool can't take without shadowing something the generated code needs.
# Other Python builtins (BFCL has functions called `sum` and `help`) only shadow
# the builtin inside the generated code's own namespace, which is harmless.
_JS_GLOBALS = frozenset(
    {
        "Array",
        "Boolean",
        "Date",
        "Error",
        "JSON",
        "Map",
        "Math",
        "Number",
        "Object",
        "Promise",
        "RegExp",
        "Set",
        "String",
        "Symbol",
        "console",
        "globalThis",
        "undefined",
    }
)
_PYTHON_NAMES = frozenset({"asyncio", "json", "math", "print"})
RESERVED_NAMES = frozenset(keyword.kwlist) | _JS_GLOBALS | _PYTHON_NAMES
# BFCL types that aren't JSON Schema; `any` drops the `type` key instead.
_BFCL_TO_JSON_TYPE = {"dict": "object", "float": "number", "tuple": "array"}

Tool = Callable[..., Awaitable[str]]


def sanitize_name(name: str) -> str:
    """Turn a BFCL function name into an identifier, e.g. `math.factorial`.

    The same name is used by every arm: OpenAI function names can't contain a
    dot, and neither can a Python or JavaScript global.

    Args:
        name: The BFCL function name.

    Returns:
        `name` with every non-word character replaced by `_`.
    """
    return re.sub(r"\W", "_", name)


def to_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a BFCL parameter schema into JSON Schema, recursively.

    Args:
        schema: A BFCL schema (`dict`, `float`, `tuple` and `any` types allowed).

    Returns:
        A copy using `object`, `number` and `array`, with no type for `any`.
    """
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            if value != "any":
                converted[key] = _BFCL_TO_JSON_TYPE.get(value, value)
        elif key == "properties" and isinstance(value, dict):
            converted[key] = {
                name: to_json_schema(spec) for name, spec in value.items()
            }
        elif key in {"items", "additionalProperties"} and isinstance(value, dict):
            converted[key] = to_json_schema(value)
        else:
            converted[key] = value
    return converted


def _type_label(spec: dict[str, Any]) -> str:
    kind = spec.get("type", "any")
    items = spec.get("items")
    if kind == "array" and isinstance(items, dict):
        return f"array of {_type_label(items)}"
    return str(kind)


def _describe_parameter(
    name: str, spec: dict[str, Any], *, required: bool, depth: int
) -> list[str]:
    """Render one parameter, and any fields nested in it, as bullet lines."""
    details = [_type_label(spec), "required" if required else "optional"]
    if "enum" in spec:
        details.append("one of " + ", ".join(json.dumps(v) for v in spec["enum"]))
    if "default" in spec:
        details.append(f"default {json.dumps(spec['default'])}")
    description = spec.get("description", "")
    indent = "  " * depth
    lines = [f"{indent}- `{name}` ({', '.join(details)}): {description}".rstrip()]
    nested = spec.get("items") if spec.get("type") == "array" else spec
    if isinstance(nested, dict) and nested.get("properties"):
        lines += _describe_parameters(nested, depth + 1)
    return lines


def _describe_parameters(schema: dict[str, Any], depth: int) -> list[str]:
    required = set(schema.get("required", []))
    return [
        line
        for name, spec in schema.get("properties", {}).items()
        for line in _describe_parameter(
            name, spec, required=name in required, depth=depth
        )
    ]


def describe_function(
    name: str, description: str, schema: dict[str, Any], language: CodeLanguage
) -> str:
    """Render one function's signature and full parameter docs.

    Code arms get the same information the baseline's native tool schemas
    carry (types, enums, defaults, descriptions), only as text.

    Args:
        name: The sanitized function name.
        description: The function's description.
        schema: Its parameters, already converted to JSON Schema.
        language: The calling convention of the signature.

    Returns:
        A `- await name(params): description` line followed by one indented
        line per parameter.
    """
    required = set(schema.get("required", []))
    optional_suffix = "=..." if language == "python" else "?"
    params = ", ".join(
        param if param in required else f"{param}{optional_suffix}"
        for param in schema.get("properties", {})
    )
    if language == "javascript" and params:
        params = f"{{{params}}}"
    header = f"- `await {name}({params})`: {description}".rstrip()
    return "\n".join([header, *_describe_parameters(schema, depth=1)])


def _check_arguments(
    name: str, arguments: dict[str, Any], params: Iterable[str], required: list[str]
) -> None:
    """Reject arguments a real function with this signature wouldn't accept.

    Raises:
        TypeError: On an unknown parameter or a missing required one.
    """
    unknown = sorted(set(arguments) - set(params))
    if unknown:
        msg = f"{name}() got unexpected argument(s): {', '.join(unknown)}"
        raise TypeError(msg)
    missing = [param for param in required if param not in arguments]
    if missing:
        msg = f"{name}() missing required argument(s): {', '.join(missing)}"
        raise TypeError(msg)


# Says outright that there is no data to wait for: given a bare echo, models
# kept reasoning, or wrote code to compute the requested numbers themselves.
SIMULATED_NOTE = (
    "Simulated function: the call is recorded and complete; no data is returned."
)


def echo_result(name: str, arguments: dict[str, Any]) -> str:
    """The deterministic result every simulated function returns.

    Args:
        name: The function's BFCL name.
        arguments: The call's normalized arguments.

    Returns:
        JSON with sorted keys, identical for identical calls in every arm.
    """
    canonical = json.dumps(arguments, sort_keys=True)
    digest = hashlib.sha256(f"{name}:{canonical}".encode()).hexdigest()[:12]
    return json.dumps(
        {
            "arguments": arguments,
            "function": name,
            "note": SIMULATED_NOTE,
            "result_id": digest,
            "status": "ok",
        },
        sort_keys=True,
    )


class BfclToolBridge:
    """The single tool surface every arm uses for one BFCL entry.

    Duck-types `McpToolBridge` (`callables`, `describe_tools`), so it can be
    passed straight to `IptcAgent`/`PtcAgent`, and also serves the baseline.
    Every function is simulated: it validates its arguments, waits `delay`
    seconds and returns `echo_result`. Every call is recorded under its BFCL
    name, for grading.

    Args:
        entry: The entry whose functions to serve.
        recorder: Receives one `ToolCallRecord` per call.
        delay: Seconds each valid call takes.

    Raises:
        ValueError: If a sanitized name collides with another, is too long,
            or shadows a keyword, a JavaScript global, `print` or a common
            module.
    """

    def __init__(self, entry: BfclEntry, recorder: RunRecorder, delay: float) -> None:
        self._recorder = recorder
        self._delay = delay
        self.bfcl_names: dict[str, str] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._descriptions: dict[str, str] = {}
        for function in entry.functions:
            name = self._register_name(entry.id, function["name"])
            self._schemas[name] = to_json_schema(function.get("parameters", {}))
            self._descriptions[name] = function.get("description", "")
        self.callables: dict[str, Tool] = {
            name: self._make_callable(name) for name in self.bfcl_names
        }

    def _register_name(self, entry_id: str, bfcl_name: str) -> str:
        name = sanitize_name(bfcl_name)
        problem = None
        if name in self.bfcl_names:
            problem = f"collides with {self.bfcl_names[name]!r}"
        elif len(name) > MAX_NAME_LENGTH:
            problem = f"is longer than {MAX_NAME_LENGTH} characters"
        elif name in RESERVED_NAMES:
            problem = "shadows a name the generated code needs"
        if problem:
            msg = f"{entry_id}: function {bfcl_name!r} ({name!r}) {problem}"
            raise ValueError(msg)
        self.bfcl_names[name] = bfcl_name
        return name

    def describe_tools(self, language: CodeLanguage = "python") -> str:
        """Function signatures and parameter docs, plus the code guidance.

        Args:
            language: The language of the `eval` code.

        Returns:
            Every function in `language`'s calling convention, followed by
            that language's guidance.
        """
        functions = "\n".join(
            describe_function(
                name, self._descriptions[name], self._schemas[name], language
            )
            for name in self.bfcl_names
        )
        return f"{functions}\n\n{EVAL_CODE_GUIDANCE[language]}"

    def openai_tools(self) -> list[dict[str, Any]]:
        """The functions as native OpenAI function tools, for the baseline."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._descriptions[name],
                    "parameters": self._schemas[name],
                },
            }
            for name in self.bfcl_names
        ]

    def _make_callable(self, name: str) -> Tool:
        schema = self._schemas[name]
        params = list(schema.get("properties", {}))
        required = list(schema.get("required", []))
        bfcl_name = self.bfcl_names[name]
        recorder, delay = self._recorder, self._delay

        def record(
            start: float, turn: int | None, arguments: str, error: str | None
        ) -> None:
            recorder.tool_calls.append(
                ToolCallRecord(
                    bfcl_name,
                    start,
                    recorder.now(),
                    ok=error is None,
                    arguments=arguments,
                    error=error,
                    turn=turn,
                )
            )

        async def call(*args: Any, **kwargs: Any) -> str:
            start, turn = recorder.now(), recorder.current_turn_index()
            try:
                bound = _bind_arguments(name, params, args, kwargs)
                _check_arguments(name, bound, params, required)
                # A JSON round trip turns tuples into lists, as BFCL's answers have.
                arguments = json.loads(json.dumps(bound, sort_keys=True))
            except (TypeError, ValueError) as exc:
                record(start, turn, _safe_json(kwargs, args), f"TypeError: {exc}")
                raise TypeError(str(exc)) from exc
            canonical = json.dumps(arguments, sort_keys=True)
            try:
                await asyncio.sleep(delay)
            except BaseException as exc:
                record(start, turn, canonical, repr(exc))
                raise
            record(start, turn, canonical, None)
            return echo_result(bfcl_name, arguments)

        call.__name__ = name
        return call


def _safe_json(kwargs: dict[str, Any], args: tuple[Any, ...]) -> str:
    """Best-effort JSON of arguments that failed validation, for the record."""
    return json.dumps({"args": list(args), "kwargs": kwargs}, default=repr)
