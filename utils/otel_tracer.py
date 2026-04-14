from contextlib import contextmanager
from typing import Any, Dict

from config.settings import (
    OTEL_SERVICE_NAME,
    OTEL_EXPORTER_OTLP_ENDPOINT,
)


class OTelTracer:
    """Optional OpenTelemetry wrapper with no-op fallback when OTel is unavailable."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._tracer = None

        if not self.enabled:
            return

        try:
            from opentelemetry import trace  # type: ignore
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore

            resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
            provider = TracerProvider(resource=resource)

            endpoint = (OTEL_EXPORTER_OTLP_ENDPOINT or "").strip()
            if endpoint:
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("indic_rag")
        except Exception:
            self._tracer = None

    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None):
        attrs = attributes or {}

        if not self.enabled or self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            try:
                for k, v in attrs.items():
                    try:
                        span.set_attribute(k, v)
                    except Exception:
                        pass
                yield span
            except Exception as ex:
                try:
                    span.record_exception(ex)
                except Exception:
                    pass
                raise
