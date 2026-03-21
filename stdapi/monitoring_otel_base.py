"""Base class for OpenTelemetry monitoring."""

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from fastapi import FastAPI
    from opentelemetry.trace.span import Span


class OpenTelemetryManager:
    """Manages OpenTelemetry tracing with AWS X-ray integration."""

    @staticmethod
    def instrument(app: FastAPI) -> None:
        """Instrument FastAPI application with OpenTelemetry.

        Args:
            app: FastAPI application instance to instrument.
        """

    def flush(self) -> None:
        """Flush OpenTelemetry tracing."""

    def start_span(self, name: str, attributes: dict[str, str]) -> Span | None:
        """Start a new tracing span.

        Args:
            name: Span name.
            attributes: Key-value metadata for the span.

        Returns:
            New span, or ``None`` when tracing is disabled.
        """

    @contextmanager
    def use_span(self, span: Span | None) -> Generator[None]:  # noqa: ARG002
        """No-op context manager; active span management is handled by subclasses.

        Args:
            span: Span to activate (ignored in this base implementation).

        Yields:
            None
        """
        yield None
