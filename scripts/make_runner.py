#!/usr/bin/env python
"""OTel-instrumented runner for Makefile targets.

Usage: uv run python scripts/make_runner.py --target <dev|ci|codegen|lint|test>
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

AUDIT_DIR = Path(".meridian")
AUDIT_LOG = AUDIT_DIR / "make-audit.ndjson"

_TRACER_NAME = "meridian.make"
_tracer = trace.get_tracer(_TRACER_NAME)

# (step_name, cmd, cwd_relative_to_repo_root, error_hint)
_Step = tuple[str, list[str], str | None, str]

_LINT_STEPS: list[_Step] = [
    (
        "lint:ruff-format",
        ["uv", "run", "ruff", "format", "--check", "."],
        None,
        "run 'uv run ruff format .' to auto-fix",
    ),
    (
        "lint:ruff-check",
        ["uv", "run", "ruff", "check", "."],
        None,
        "run 'uv run ruff check --fix .' to auto-fix",
    ),
    (
        "lint:pyright",
        ["uv", "run", "pyright"],
        None,
        "fix type errors shown above",
    ),
    (
        "lint:lint-imports",
        ["uv", "run", "lint-imports"],
        None,
        "fix import boundary violations shown above",
    ),
    (
        "lint:biome",
        ["packages/sdk-widget/node_modules/.bin/biome", "check", "."],
        None,
        "run 'biome check --write .' to auto-fix",
    ),
    (
        "lint:tsc",
        ["npx", "tsc", "--noEmit"],
        "packages/sdk-widget",
        "fix TypeScript type errors shown above",
    ),
]

_TEST_STEPS: list[_Step] = [
    (
        "test:pytest",
        ["uv", "run", "pytest"],
        None,
        "fix failing tests shown above",
    ),
    (
        "test:vitest",
        ["npm", "test"],
        "packages/sdk-widget",
        "fix failing tests shown above",
    ),
]

_CODEGEN_STEPS: list[_Step] = [
    (
        "codegen:openapi",
        ["uv", "run", "python", "scripts/export_openapi.py"],
        None,
        "fix openapi export errors shown above",
    ),
    (
        "codegen:sdk-ts",
        ["uv", "run", "python", "scripts/gen_sdk_ts.py"],
        None,
        "fix openapi-typescript errors shown above",
    ),
    (
        "codegen:sdk-py",
        ["uv", "run", "python", "scripts/gen_sdk_py.py"],
        None,
        "fix datamodel-codegen errors shown above",
    ),
]

_STEPS: dict[str, list[_Step]] = {
    "lint": _LINT_STEPS,
    "test": _TEST_STEPS,
    "codegen": _CODEGEN_STEPS,
    "ci": _CODEGEN_STEPS + _LINT_STEPS + _TEST_STEPS,
}


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _audit(level: str, event: str, detail: dict | None = None) -> None:
    AUDIT_DIR.mkdir(exist_ok=True)
    entry: dict = {"ts": _now(), "level": level, "event": event}
    if detail:
        entry["detail"] = detail
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_steps(target: str) -> int:
    steps = _STEPS[target]
    _audit("info", f"make.{target}.invoked", {"target": target})

    with _tracer.start_as_current_span(f"make.{target}") as span:
        span.set_attribute("make.target", target)

        if not steps:
            msg = f"make {target}: no steps configured (pipeline not yet implemented)"
            print(msg, file=sys.stderr)
            _audit("warn", f"make.{target}.noop", {"target": target, "message": msg})
            span.add_event("make.noop", {"make.target": target})
            return 0

        failed: list[tuple[str, str]] = []
        for step_name, cmd, cwd, hint in steps:
            span.add_event("make.step.start", {"make.step": step_name})
            rc = subprocess.run(cmd, cwd=cwd).returncode
            if rc != 0:
                failed.append((step_name, hint))
                span.add_event("make.step.failed", {"make.step": step_name, "make.hint": hint})

        if failed:
            stage_list = ", ".join(s for s, _ in failed)
            hints = "\n".join(f"  {s}: {h}" for s, h in failed)
            msg = f"STAGE FAILED: {stage_list}\n{hints}"
            print(f"\nerror: {msg}", file=sys.stderr)
            _audit(
                "error",
                f"make.{target}.failed",
                {
                    "target": target,
                    "failed_stages": [s for s, _ in failed],
                    "message": msg,
                },
            )
            span.set_status(Status(StatusCode.ERROR, stage_list))
            span.add_event(
                "make.failed",
                {"make.target": target, "make.failed_stages": stage_list},
            )
            return 1

        _audit("info", f"make.{target}.ok", {"target": target})
        span.add_event("make.completed", {"make.target": target})
        return 0


def _run_dev() -> int:
    daemon_dir = Path("apps/meridiand")
    ui_dir = Path("apps/meridian-ui")
    missing = [str(d) for d in [daemon_dir, ui_dir] if not d.exists()]

    if missing:
        msg = (
            f"make dev: app directories not yet created: {', '.join(missing)}\n"
            "  See docs/ARCHITECTURE.md §23 for the planned monorepo layout.\n"
            "  Expected: apps/meridiand (Python FastAPI daemon)"
            " and apps/meridian-ui (TypeScript UI)."
        )
        print(f"error: {msg}", file=sys.stderr)
        _audit("error", "make.dev.failed", {"target": "dev", "missing": missing, "message": msg})
        return 1

    _audit("info", "make.dev.invoked", {"target": "dev"})

    with _tracer.start_as_current_span("make.dev") as span:
        span.set_attribute("make.target", "dev")

        daemon = subprocess.Popen(["uv", "run", "python", "-m", "meridiand"], cwd=str(daemon_dir))
        ui = subprocess.Popen(["npm", "run", "dev"], cwd=str(ui_dir))
        procs = [daemon, ui]
        names = ["daemon", "ui"]

        span.add_event("make.dev.started", {"daemon.pid": daemon.pid, "ui.pid": ui.pid})
        print(f"Meridian dev: daemon pid={daemon.pid}  ui pid={ui.pid}  (Ctrl-C to stop)")

        exit_code = 0
        try:
            while True:
                for proc, name in zip(procs, names, strict=True):
                    rc = proc.poll()
                    if rc is not None:
                        msg = f"make dev: {name} exited with code {rc}"
                        if rc != 0:
                            print(f"error: {msg}", file=sys.stderr)
                            _audit("error", "make.dev.proc_exited", {"proc": name, "exit_code": rc})
                            span.set_status(Status(StatusCode.ERROR, msg))
                        else:
                            _audit("info", "make.dev.proc_exited", {"proc": name, "exit_code": rc})
                        span.add_event(
                            "make.dev.proc_exited",
                            {"make.dev.proc": name, "make.dev.exit_code": rc},
                        )
                        exit_code = rc
                        return exit_code
                time.sleep(0.1)
        except KeyboardInterrupt:
            _audit("info", "make.dev.interrupted", {"target": "dev"})
            span.add_event("make.dev.interrupted")
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                proc.wait()

        if exit_code == 0:
            _audit("info", "make.dev.ok", {"target": "dev"})
            span.add_event("make.completed", {"make.target": "dev"})
        return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["dev", "ci", "codegen", "lint", "test"])
    args = parser.parse_args()

    if args.target == "dev":
        sys.exit(_run_dev())
    else:
        sys.exit(_run_steps(args.target))


if __name__ == "__main__":
    main()
