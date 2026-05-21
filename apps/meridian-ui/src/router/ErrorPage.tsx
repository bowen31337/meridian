import { useEffect } from "react";
import { useRouteError } from "react-router-dom";
import { useMeridianApi } from "../api/index.js";
import { getRouteTracer, recordRouteFailure } from "./telemetry.js";

export function ErrorPage() {
  const error = useRouteError();
  const { auditLog } = useMeridianApi();
  const message = error instanceof Error ? error.message : String(error);

  useEffect(() => {
    const tracer = getRouteTracer();
    tracer.startActiveSpan("ui.route.error", (span) => {
      recordRouteFailure(span, error, auditLog, { route: window.location.pathname });
      span.end();
    });
  }, [error, auditLog]);

  return (
    <div role="alert">
      <h1>Something went wrong</h1>
      <p>{message}</p>
    </div>
  );
}
