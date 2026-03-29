class JitGenError(Exception):
    """Base class for all JITGen-related errors."""


class ParsingError(JitGenError):
    """Raised when there is an error during parsing."""


class SyntaxError(ParsingError):
    """Raised when there is a syntax error in the input code."""
