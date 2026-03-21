"""Monitoring using OpenTelemetry."""

from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagators.aws import AwsXRayPropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import use_span

from stdapi.config import SETTINGS
from stdapi.monitoring_otel_base import OpenTelemetryManager as _OpenTelemetryManager

if TYPE_CHECKING:
    from collections.abc import Generator

    from fastapi import FastAPI
    from opentelemetry.trace.span import Span


class OpenTelemetryManager(_OpenTelemetryManager):
    """Manages OpenTelemetry tracing with AWS X-ray integration."""

    __slots__ = ("_tracer_provider", "tracer")

    def __init__(self) -> None:
        """Initialize the OpenTelemetry manager."""
        resource = Resource.create(
            {"service.name": SETTINGS.otel_service_name, "service.version": "1.0.0"}
        )
        self._tracer_provider = TracerProvider(
            resource=resource, sampler=TraceIdRatioBased(SETTINGS.otel_sample_rate)
        )
        self._tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=SETTINGS.otel_exporter_endpoint),
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=200,
                export_timeout_millis=10000,
            )
        )
        trace.set_tracer_provider(self._tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        BotocoreInstrumentor().instrument()  # type: ignore[no-untyped-call]
        AioHttpClientInstrumentor().instrument()

        propagate.set_global_textmap(AwsXRayPropagator())

    @staticmethod
    def instrument(app: FastAPI) -> None:
        """Instrument FastAPI application with OpenTelemetry.

        Args:
            app: FastAPI application instance to instrument.
        """
        FastAPIInstrumentor.instrument_app(app)

    def flush(self) -> None:
        """Flush OpenTelemetry tracing."""
        self._tracer_provider.force_flush()
        self._tracer_provider.shutdown()

    def start_span(self, name: str, attributes: dict[str, str]) -> Span:
        """Start a new tracing span.

        Args:
            name: Span name.
            attributes: Key-value metadata for the span.

        Returns:
            New span instance.
        """
        return self.tracer.start_span(name, attributes=attributes)

    @contextmanager
    def use_span(self, span: Span) -> Generator[None]:  # type: ignore[override]
        """Activate *span* as the current span within this context.

        Args:
            span: Span to activate.

        Yields:
            None
        """
        with use_span(span):
            yield None
