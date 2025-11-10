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
    def instrument(app: "FastAPI") -> None:
        """Instrument FastAPI application with OpenTelemetry.

        Args:
            app: FastAPI application instance to instrument.
        """

    def flush(self) -> None:
        """Flush OpenTelemetry tracing."""

    def start_span(self, name: str, attributes: dict[str, str]) -> "Span | None":
        """Starts a new span with a given name and attributes.

        Args:
            name: The name assigned to the span.
            attributes: A dictionary of key-value pairs

        Returns:
            None
        """

    @contextmanager
    def use_span(self, span: "Span | None") -> "Generator[None]":  # noqa: ARG002
        """Use span.

        Args:
            span: The span to use.
        """
        yield None
