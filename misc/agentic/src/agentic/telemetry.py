import os


def configure(*, enable_langsmith: bool, enable_otel: bool) -> None:
    """Set env vars that control LangSmith and OTel tracing.

    Must be called before any langchain modules are imported so that
    the tracing SDK reads the correct values at initialisation time.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_langsmith else "false"
    if not enable_otel:
        # Prevent the OTel SDK from attempting to connect to a collector.
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
