import unittest
from unittest.mock import Mock, patch

from agentic.agents import IncrementalAgentSession


class FakeAlgorithm:
    def __init__(self) -> None:
        self._code_buffer = ""
        self._raw_buffer = ""
        self._inside_markers = False


class FakeAsyncSession:
    def __init__(self) -> None:
        self._algorithm = FakeAlgorithm()
        self._stdout_handler = None
        self._error_handler = None
        self.apush_error: Exception | None = None
        self.aflush_error: Exception | None = None

    def on_stdout(self, handler):
        self._stdout_handler = handler
        return handler

    def on_error(self, handler):
        self._error_handler = handler
        return handler

    async def apush(self, chunk: str) -> None:
        if self.apush_error is None:
            return

        self._error_handler(self.apush_error)
        raise self.apush_error

    async def aflush(self) -> None:
        if self.aflush_error is None:
            return

        self._error_handler(self.aflush_error)
        raise self.aflush_error


class IncrementalAgentSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_chunk_swallows_handled_session_errors(self) -> None:
        fake_session = FakeAsyncSession()
        fake_session.apush_error = ValueError("Syntax error detected")

        with patch(
            "agentic.agents.create_python_async_jitgen_session",
            return_value=fake_session,
        ):
            session = IncrementalAgentSession(Mock(), [], "question", "guidelines")

        session._last_response = ""
        session._last_observation = ""

        await session.on_chunk("<execute>!\n")

        self.assertFalse(session._should_continue_streaming)
        self.assertEqual(session._handled_session_error, fake_session.apush_error)
        self.assertIn("Syntax error detected", session._last_observation)

    async def test_on_after_step_swallows_handled_flush_errors_and_resets_state(self) -> None:
        fake_session = FakeAsyncSession()
        fake_session.aflush_error = ValueError("Flush syntax error")
        fake_session._algorithm._code_buffer = "print('x')"
        fake_session._algorithm._raw_buffer = "<execute>print('x')"
        fake_session._algorithm._inside_markers = True

        with patch(
            "agentic.agents.create_python_async_jitgen_session",
            return_value=fake_session,
        ):
            session = IncrementalAgentSession(Mock(), [], "question", "guidelines")

        session._last_response = "<execute>print('x')"
        session._last_observation = ""
        session._should_continue_streaming = True

        await session.on_after_step()

        self.assertFalse(session._should_continue_streaming)
        self.assertEqual(fake_session._algorithm._code_buffer, "")
        self.assertEqual(fake_session._algorithm._raw_buffer, "")
        self.assertFalse(fake_session._algorithm._inside_markers)
        self.assertIsNone(session._handled_session_error)
        self.assertIn("Flush syntax error", session._last_observation)


if __name__ == "__main__":
    unittest.main()