import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { getRouteTracer, recordRouteNavigationEvent } from "./telemetry.js";

export function NavigationTracer() {
  const location = useLocation();

  useEffect(() => {
    const tracer = getRouteTracer();
    tracer.startActiveSpan("ui.route.navigate", (span) => {
      recordRouteNavigationEvent(span, {
        name: "ui.navigation",
        route: location.pathname,
        timestamp: new Date().toISOString(),
      });
      span.end();
    });
  }, [location.pathname]);

  return <Outlet />;
}
