import re
from collections.abc import Callable
from enum import Enum, auto
from typing import ClassVar

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
_UNICODE_ESCAPE_LENGTH = len("\\uXXXX")

_STRING_DELIMITER = re.compile(r'["\\]')
_NESTED_DELIMITER = re.compile(r'["{}\[\]]')
_SCALAR_END = re.compile(r"[\s,}\]]")
_HEX_CODE_UNIT = re.compile(r"[0-9a-fA-F]{4}")


def _is_high_surrogate(code_point: int) -> bool:
    return 0xD800 <= code_point < 0xDC00


def _is_low_surrogate(code_point: int) -> bool:
    return 0xDC00 <= code_point < 0xE000


class _StringDecoder:
    """Incrementally decodes one JSON string body, across chunk boundaries.

    An escape sequence or surrogate pair split between chunks is held back
    until it is complete, so it is never misread as plain characters.
    """

    def __init__(self) -> None:
        self._escape = ""
        self._high_surrogate = ""

    def reset(self) -> None:
        """Forget any partial escape so a new string can be decoded."""
        self._escape = ""
        self._high_surrogate = ""

    def decode(self, chunk: str, pos: int) -> tuple[str, int, bool]:
        """Decode `chunk` from `pos` up to the closing quote or the chunk's end.

        Args:
            chunk: Raw JSON text.
            pos: Index of the first string-body character to decode.

        Returns:
            `(decoded, next_pos, closed)`, where `closed` is `True` when the
            string's closing quote was consumed.
        """
        parts: list[str] = []
        while pos < len(chunk):
            char = chunk[pos]
            if self._escape:
                self._escape += char
                pos += 1
                parts.append(self._resolve_escape())
            elif char == "\\":
                self._escape = char
                pos += 1
            elif char == '"':
                parts.append(self.flush())
                return "".join(parts), pos + 1, True
            else:
                delimiter = _STRING_DELIMITER.search(chunk, pos)
                run_end = delimiter.start() if delimiter else len(chunk)
                parts.append(self.flush() + chunk[pos:run_end])
                pos = run_end
        return "".join(parts), pos, False

    def flush(self) -> str:
        """Release a high surrogate still waiting for its pair.

        Returns:
            The held-back surrogate, or `""` when there is none.
        """
        pending, self._high_surrogate = self._high_surrogate, ""
        return pending

    def _resolve_escape(self) -> str:
        escape = self._escape
        if escape[1] != "u":
            self._escape = ""
            return self.flush() + _SIMPLE_ESCAPES.get(escape[1], escape[1])
        if len(escape) < _UNICODE_ESCAPE_LENGTH:
            return ""
        self._escape = ""
        hex_digits = escape[2:]
        if not _HEX_CODE_UNIT.fullmatch(hex_digits):
            return ""  # malformed escape; drop it rather than raise
        return self._combine_surrogates(int(hex_digits, 16))

    def _combine_surrogates(self, code_point: int) -> str:
        high = self.flush()
        if high and _is_low_surrogate(code_point):
            return chr(0x10000 + (ord(high) - 0xD800) * 0x400 + (code_point - 0xDC00))
        if _is_high_surrogate(code_point):
            self._high_surrogate = chr(code_point)
            return high
        return high + chr(code_point)


class _State(Enum):
    SEEK_OBJECT = auto()
    SEEK_KEY = auto()
    IN_KEY = auto()
    SEEK_VALUE = auto()
    IN_TARGET_VALUE = auto()
    SKIP_STRING = auto()
    SKIP_NESTED = auto()
    SKIP_SCALAR = auto()


class OpenAIToolCallSegmenter:
    """Streams one string property out of a tool call's JSON `arguments`.

    Feed it the raw `arguments` deltas (`delta.tool_calls[i].function.arguments`);
    it emits the decoded value of `property_name` as soon as each character is
    final, skips every other property, and ends the block at the value's closing
    quote. Each new top-level `{` starts a fresh scan, so several objects can be
    fed back to back.

    Args:
        property_name: The JSON key whose string value holds the code.
    """

    def __init__(self, *, property_name: str) -> None:
        self.property_name = property_name
        self._decoder = _StringDecoder()
        self.reset()

    @property
    def inside_block(self) -> bool:
        """`True` while `property_name`'s string value is still open."""
        return self._state is _State.IN_TARGET_VALUE

    def reset(self) -> None:
        """Clear buffered state so the segmenter can be reused for a new stream."""
        self._decoder.reset()
        self._state = _State.SEEK_OBJECT
        self._key_parts: list[str] = []
        self._is_target_key = False
        self._nesting_depth = 0
        self._after_skipped_string = _State.SEEK_KEY

    def feed(self, chunk: str) -> list[CodeSegment]:
        """Append `chunk` and return any newly extractable code segments.

        Args:
            chunk: The next increment of raw `arguments` text.

        Returns:
            Code segments that became extractable as a result of `chunk`, in
            order; empty when nothing is ready yet.
        """
        segments: list[CodeSegment] = []
        pos = 0
        while pos < len(chunk):
            if self._state is not _State.IN_TARGET_VALUE:
                pos = self._HANDLERS[self._state](self, chunk, pos)
                continue
            text, pos, closed = self._decoder.decode(chunk, pos)
            if closed:
                self._state = _State.SEEK_KEY
                segments.append(CodeSegment(text=text, end_of_block=True))
            elif text:
                segments.append(CodeSegment(text=text))
        return segments

    def finalize(self) -> list[CodeSegment]:
        """Flush whatever is buffered when the stream ends inside the value.

        Returns:
            A final block-ending segment with any held-back text, or `[]`.
        """
        if not self.inside_block:
            return []
        text = self._decoder.flush()
        return [CodeSegment(text=text, end_of_block=True)] if text else []

    # Each handler below consumes input in its state, starting at `pos`, and
    # returns the index of the next unprocessed character.

    def _seek_object(self, chunk: str, pos: int) -> int:
        start = chunk.find("{", pos)
        if start == -1:
            return len(chunk)
        self._state = _State.SEEK_KEY
        return start + 1

    def _seek_key(self, chunk: str, pos: int) -> int:
        char = chunk[pos]
        if char == '"':
            self._key_parts = []
            self._start_string(_State.IN_KEY)
        elif char == "}":
            self._state = _State.SEEK_OBJECT
        return pos + 1  # whitespace, commas and stray characters are ignored

    def _read_key(self, chunk: str, pos: int) -> int:
        text, pos, closed = self._decoder.decode(chunk, pos)
        self._key_parts.append(text)
        if closed:
            self._is_target_key = "".join(self._key_parts) == self.property_name
            self._state = _State.SEEK_VALUE
        return pos

    def _seek_value(self, chunk: str, pos: int) -> int:
        char = chunk[pos]
        if char.isspace() or char == ":":
            return pos + 1
        if char == '"':
            if self._is_target_key:
                self._start_string(_State.IN_TARGET_VALUE)
            else:
                self._skip_string_then(_State.SEEK_KEY)
            return pos + 1
        if char in "{[":
            self._nesting_depth = 1
            self._state = _State.SKIP_NESTED
            return pos + 1
        self._state = _State.SKIP_SCALAR
        return pos

    def _skip_string(self, chunk: str, pos: int) -> int:
        _, pos, closed = self._decoder.decode(chunk, pos)
        if closed:
            self._state = self._after_skipped_string
        return pos

    def _skip_nested(self, chunk: str, pos: int) -> int:
        delimiter = _NESTED_DELIMITER.search(chunk, pos)
        if delimiter is None:
            return len(chunk)
        char = delimiter.group()
        if char == '"':
            self._skip_string_then(_State.SKIP_NESTED)
        elif char in "{[":
            self._nesting_depth += 1
        else:
            self._nesting_depth -= 1
            if self._nesting_depth == 0:
                self._state = _State.SEEK_KEY
        return delimiter.end()

    def _skip_scalar(self, chunk: str, pos: int) -> int:
        delimiter = _SCALAR_END.search(chunk, pos)
        if delimiter is None:
            return len(chunk)
        self._state = _State.SEEK_KEY
        return delimiter.start()  # let `SEEK_KEY` consume the delimiter

    def _start_string(self, state: _State) -> None:
        self._decoder.reset()
        self._state = state

    def _skip_string_then(self, resume: _State) -> None:
        self._after_skipped_string = resume
        self._start_string(_State.SKIP_STRING)

    _HANDLERS: ClassVar[dict[_State, Callable[..., int]]] = {
        _State.SEEK_OBJECT: _seek_object,
        _State.SEEK_KEY: _seek_key,
        _State.IN_KEY: _read_key,
        _State.SEEK_VALUE: _seek_value,
        _State.SKIP_STRING: _skip_string,
        _State.SKIP_NESTED: _skip_nested,
        _State.SKIP_SCALAR: _skip_scalar,
    }
