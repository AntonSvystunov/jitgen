from jitgen.base import BaseExecutor
from jitgen.session import Session

from jitgen_js.executors.quickjs import QuickJsExecutor
from jitgen_js.extractors.javascript import JavaScriptAntlrExtractor


def create_javascript_session(executor: BaseExecutor | None = None) -> Session:
    """Build a `Session` wired for JavaScript: ANTLR4 extractor + QuickJS executor.

    Unlike `create_python_session`, `JavaScriptAntlrExtractor` takes no parser
    argument — it builds its own fresh lexer/parser per parse call (see
    `JavaScriptAntlrExtractor._build_parser`), so there is no shared, cacheable
    parser object to construct once and inject here.

    Args:
        executor: Executor to use in place of a session-owned `QuickJsExecutor`.
            Pass your own to keep ownership of it — `aclose()` will not close
            an executor it was not given ownership of.

    Returns:
        A `Session` ready to have JavaScript source pushed into it.
    """
    owns_executor = executor is None
    return Session(
        extractor=JavaScriptAntlrExtractor(),
        executor=executor if executor is not None else QuickJsExecutor(),
        owns_executor=owns_executor,
    )
