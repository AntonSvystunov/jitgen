from enum import Enum, auto

from jitgen.base import CodeSegment

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_HIGH_SURROGATE_RANGE = range(0xD800, 0xDC00)
_LOW_SURROGATE_RANGE = range(0xDC00, 0xE000)


class _State(Enum):
    SEEK_OBJECT_START = auto()
    SEEK_KEY_OR_END = auto()
    IN_KEY = auto()
    SEEK_COLON = auto()
    SEEK_VALUE = auto()
    IN_TARGET_VALUE = auto()
    IN_SKIP_STRING = auto()
    SKIP_STRUCT = auto()
    SKIP_SCALAR = auto()
    SEEK_COMMA_OR_END = auto()


class _StringScanner:
    """Decodes one JSON string body, one character at a time.

    Escape sequences — including a `\\uXXXX` split across several calls, and
    a surrogate pair split across two separate escapes — are tracked as
    instance state so a chunk boundary landing mid-escape is buffered rather
    than misread as a plain character or an error.
    """

    def __init__(self) -> None:
        self._escaped = False
        self._pending_escape = ""
        self._pending_high_surrogate: str | None = None

    def reset(self) -> None:
        """Clear buffered escape state so the scanner can decode a new string."""
        self._escaped = False
        self._pending_escape = ""
        self._pending_high_surrogate = None

    def step(self, char: str) -> tuple[list[str], bool]:
        """Consume one character of a string body (after the opening quote).

        Args:
            char: The next raw character of the string body.

        Returns:
            `(decoded_chars, closed)`. `decoded_chars` is usually zero or one
            characters — two only when an unpaired high surrogate is flushed
            alongside the character that follows it. `closed` is `True` when
            `char` was the unescaped quote that ends the string.
        """
        if self._pending_escape:
            return self._continue_unicode_escape(char), False
        if self._escaped:
            self._escaped = False
            return self._resolve_escape(char), False
        if char == "\\":
            self._escaped = True
            return [], False
        if char == '"':
            return self.flush_pending_surrogate(), True
        return [char], False

    def flush_pending_surrogate(self) -> list[str]:
        """Emit a held-back high surrogate that never got its pair.

        Called both when the string closes and, best-effort, when the
        stream ends while a value is still open (see
        `OpenAIToolCallSegmenter.finalize`).

        Returns:
            `[pending_char]`, or `[]` when nothing was held back.
        """
        if self._pending_high_surrogate is None:
            return []
        pending, self._pending_high_surrogate = self._pending_high_surrogate, None
        return [pending]

    def _resolve_escape(self, char: str) -> list[str]:
        if char == "u":
            self._pending_escape = "u"
            return []
        return self._emit(_SIMPLE_ESCAPES.get(char, char))

    def _continue_unicode_escape(self, char: str) -> list[str]:
        self._pending_escape += char
        if len(self._pending_escape) < 5:  # "u" + 4 hex digits
            return []
        hex_digits, self._pending_escape = self._pending_escape[1:], ""
        try:
            code_point = int(hex_digits, 16)
        except ValueError:
            return []  # malformed escape; drop it rather than raise
        return self._emit(chr(code_point))

    def _emit(self, unit: str) -> list[str]:
        code_point = ord(unit)
        if self._pending_high_surrogate is not None:
            high, self._pending_high_surrogate = self._pending_high_surrogate, None
            if code_point in _LOW_SURROGATE_RANGE:
                combined = (
                    0x10000 + (ord(high) - 0xD800) * 0x400 + (code_point - 0xDC00)
                )
                return [chr(combined)]
            return [high, unit]
        if code_point in _HIGH_SURROGATE_RANGE:
            self._pending_high_surrogate = unit
            return []
        return [unit]


class OpenAIToolCallSegmenter:
    """Extracts one JSON string property from a streaming tool-call `arguments` blob.

    OpenAI (and OpenAI-compatible) tool-call streaming delivers a function's
    arguments as successive plain-text deltas of one growing JSON object,
    e.g. `{"la` -> `{"language": "python", "co` -> `{"language": "python",
    "code": "pri` -> ... This scans that raw text and emits the *decoded*
    value of `property_name` as soon as each character of it is final,
    correctly skipping over any other properties before or after it and
    treating the property's closing quote as the end of the block.

    Feed it the raw `arguments` delta text yourself, e.g.
    `chunk.choices[0].delta.tool_calls[0].function.arguments` — this class
    has no OpenAI SDK dependency and never sees the chunk object itself.

    One instance scans one JSON object's `property_name` at a time, but it
    is safe to keep feeding it further JSON afterward — either after
    `StreamDriver` resets it on `end_of_block`, or even concatenated within
    a single `feed()` call (e.g. two tool calls' `arguments` fed back to
    back) — each new `{` restarts the scan from scratch.

    Args:
        property_name: The JSON key whose string value should be extracted.
    """

    def __init__(self, *, property_name: str) -> None:
        self.property_name = property_name
        self._state = _State.SEEK_OBJECT_START
        self._scanner = _StringScanner()
        self._key_chars: list[str] = []
        self._current_key = ""
        self._struct_depth = 0
        self._struct_in_string = False
        self._struct_scanner = _StringScanner()
        self._output: list[str] = []
        self._segments: list[CodeSegment] = []

    @property
    def inside_block(self) -> bool:
        """`True` while scanning `property_name`'s still-open string value."""
        return self._state is _State.IN_TARGET_VALUE

    def reset(self) -> None:
        """Clear buffered state so the segmenter can be reused for a new stream."""
        self._state = _State.SEEK_OBJECT_START
        self._scanner.reset()
        self._key_chars = []
        self._current_key = ""
        self._struct_depth = 0
        self._struct_in_string = False
        self._struct_scanner.reset()
        self._output = []

    def feed(self, chunk: str) -> list[CodeSegment]:
        """Append `chunk` and return any newly extractable code segments.

        Args:
            chunk: The next increment of raw `arguments` text.

        Returns:
            Code segments that became extractable as a result of `chunk`, in
            order; empty when nothing is ready yet.
        """
        self._segments = []
        for char in chunk:
            self._advance(char)
        if self._state is _State.IN_TARGET_VALUE and self._output:
            self._segments.append(CodeSegment(text="".join(self._output)))
            self._output = []
        segments, self._segments = self._segments, []
        return segments

    def finalize(self) -> list[CodeSegment]:
        """Flush whatever is buffered at end-of-stream.

        Used when the tool call's `arguments` ended without closing
        `property_name`'s string — inference stopped early, or the model
        never reached the closing quote — so the code written so far is
        executed rather than silently discarded.

        Returns:
            Any code segments still held back, in order.
        """
        if self._state is not _State.IN_TARGET_VALUE:
            return []
        self._output.extend(self._scanner.flush_pending_surrogate())
        text = "".join(self._output)
        self._output = []
        return [CodeSegment(text=text, end_of_block=True)] if text else []

    # ── private ────────────────────────────────────────────────────────

    def _advance(self, char: str) -> None:
        """Drive the state machine over one raw character.

        A single character can cross more than one state transition (e.g. a
        scalar value's first character both starts `SKIP_SCALAR` and, if
        the value is empty, immediately ends it), so `_dispatch` is
        retried on the same character until it reports the char consumed.

        Args:
            char: The next raw character of the buffered JSON text.
        """
        while self._dispatch(char):
            pass

    def _dispatch(self, char: str) -> bool:
        """Handle `char` in the current state.

        Args:
            char: The character to process under `self._state`.

        Returns:
            `True` when `char` must be reprocessed under the state this call
            just switched to; `False` once `char` has been fully consumed.
        """
        state = self._state

        if state is _State.SEEK_OBJECT_START:
            if char == "{":
                self._state = _State.SEEK_KEY_OR_END
            return False

        if state is _State.SEEK_KEY_OR_END:
            if char.isspace():
                return False
            if char == '"':
                self._key_chars = []
                self._scanner.reset()
                self._state = _State.IN_KEY
                return False
            if char == "}":
                self._state = _State.SEEK_OBJECT_START
            return False  # lenient: ignore any other unexpected character

        if state is _State.IN_KEY:
            decoded, closed = self._scanner.step(char)
            self._key_chars.extend(decoded)
            if closed:
                self._current_key = "".join(self._key_chars)
                self._state = _State.SEEK_COLON
            return False

        if state is _State.SEEK_COLON:
            if char.isspace():
                return False
            if char == ":":
                self._state = _State.SEEK_VALUE
            return False

        if state is _State.SEEK_VALUE:
            if char.isspace():
                return False
            if char == '"':
                self._scanner.reset()
                self._output = []
                self._state = (
                    _State.IN_TARGET_VALUE
                    if self._current_key == self.property_name
                    else _State.IN_SKIP_STRING
                )
                return False
            if char in "{[":
                self._struct_depth = 1
                self._struct_in_string = False
                self._state = _State.SKIP_STRUCT
                return False
            self._state = _State.SKIP_SCALAR
            return True  # let SKIP_SCALAR see this same char

        if state is _State.IN_TARGET_VALUE:
            decoded, closed = self._scanner.step(char)
            self._output.extend(decoded)
            if closed:
                text = "".join(self._output)
                self._output = []
                self._segments.append(CodeSegment(text=text, end_of_block=True))
                self._state = _State.SEEK_COMMA_OR_END
            return False

        if state is _State.IN_SKIP_STRING:
            _, closed = self._scanner.step(char)
            if closed:
                self._state = _State.SEEK_COMMA_OR_END
            return False

        if state is _State.SKIP_STRUCT:
            if self._struct_in_string:
                _, closed = self._struct_scanner.step(char)
                if closed:
                    self._struct_in_string = False
                return False
            if char == '"':
                self._struct_in_string = True
                self._struct_scanner.reset()
                return False
            if char in "{[":
                self._struct_depth += 1
                return False
            if char in "}]":
                self._struct_depth -= 1
                if self._struct_depth == 0:
                    self._state = _State.SEEK_COMMA_OR_END
            return False

        if state is _State.SKIP_SCALAR:
            if char.isspace() or char in ",}]":
                self._state = _State.SEEK_COMMA_OR_END
                return True  # let SEEK_COMMA_OR_END see the delimiter
            return False

        if state is _State.SEEK_COMMA_OR_END:
            if char.isspace():
                return False
            if char == ",":
                self._state = _State.SEEK_KEY_OR_END
                return False
            if char in "}]":
                self._state = _State.SEEK_OBJECT_START
            return False  # lenient: ignore any other unexpected character

        return False
