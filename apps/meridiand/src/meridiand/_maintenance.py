"""Deterministic clawinfra maintenance harness.

A cron fire of ``kind == "maintenance"`` is handled here, NOT by a free-form
agent turn with shell access.  The split is deliberate (§13.5): the LLM agent
holds only capability-gated ``fs.*`` tools (it edits files), while every git /
gh / test step is run in-process by this harness as a plain subprocess.  The
agent never gets a shell, so a prompt-injected web page can never reach
``git push``; and the git plumbing is deterministic rather than hallucinated.

The repo tree to maintain is named per-cron in ``metadata.repos_root`` (so the
daemon hardcodes no user path); the rotating queue and append-only log live
under ``<storage_root>/maintenance/<root-name>.*``.

Per fire, exactly ONE repo (the head of the queue) is handled:

1. Pick + rotate the head of the queue (persisted, so a crash never re-runs the
   same repo forever).
2. Skip sensitive repos and repos with a dirty working tree (never disturb WIP).
3. ``git checkout -B alex/maint-<repo>`` off the remote default branch.
4. Run ONE edit-only agent turn (``deliver=False``): the agent reads the repo,
   makes a single focused improvement by editing files, and returns a summary.
5. If files changed: run the repo's quick test command, commit, push, and open a
   PR (draft when tests fail).  Degrade gracefully when ``gh`` is unauthenticated
   (push the branch, skip the PR, say so).
6. Restore the default branch, append a log line, and deliver a one-line status.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

# (exit_code, stdout, stderr)
CommandRunner = Callable[[list[str], "Path | None", float], Awaitable[tuple[int, str, str]]]

# Repos that must never be auto-edited: a validator key manager and a wallet.
# They are skipped from the write-flow; touch them by hand.
_DEFAULT_SKIP = frozenset({"clawkeyring", "claw-wallet"})

_DEFAULT_TEST_TIMEOUT = 90.0
_GIT_TIMEOUT = 60.0


class _Responder(Protocol):
    async def run_prompt(
        self, channel_id: str, prompt: str, *, session_id: str = "", deliver: bool = True
    ) -> str | None: ...

    async def deliver_text(self, channel_id: str, text: str) -> None: ...


def _now_date() -> str:
    return datetime.now(UTC).date().isoformat()


async def _default_runner(
    args: list[str], cwd: Path | None, timeout: float
) -> tuple[int, str, str]:
    """Run a command, never raising; returns (exit_code, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
        )
    except OSError as exc:
        return 127, "", f"failed to start {args[0]!r}: {exc}"
    try:
        raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout:.0f}s"
    return (
        proc.returncode if proc.returncode is not None else -1,
        raw_out.decode("utf-8", errors="replace"),
        raw_err.decode("utf-8", errors="replace"),
    )


def _detect_test_cmd(repo: Path) -> list[str] | None:
    """Best-effort quick test command for a repo, or None when undetectable."""
    if (repo / "pyproject.toml").exists() or (repo / "pytest.ini").exists():
        return ["python", "-m", "pytest", "-q", "-x"]
    if (repo / "Cargo.toml").exists():
        return ["cargo", "test", "--quiet"]
    if (repo / "go.mod").exists():
        return ["go", "test", "./..."]
    pkg = repo / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        scripts = data.get("scripts")
        if isinstance(scripts, dict) and "test" in scripts:
            return ["npm", "test", "--silent"]
    return None


def _commit_subject(reply: str, repo: str) -> str:
    """Derive a one-line commit subject from the agent's summary."""
    for raw in (reply or "").splitlines():
        line = raw.strip().lstrip("#").strip()
        if line:
            return f"chore(maint): {line}"[:100]
    return f"chore(maint): routine maintenance for {repo}"


_AGENT_PROMPT = """\
You are doing autonomous maintenance on a single repository.

Repository: {repo}
Absolute path: {path}

A fresh branch has already been checked out for you. Your ONLY job is to make ONE
small, focused, safe improvement by editing files with your file tools — then stop.

Pick the single most valuable bounded item you can FINISH now, e.g.:
- a roadmap / TODO / design-doc item (check README, ROADMAP, docs/),
- a clear bug or open issue,
- a missing or failing test, or a small docs gap.

Rules:
- Edit files under {path} ONLY. Use absolute paths.
- Keep the diff small and self-contained. Do NOT broadly refactor.
- Do NOT run git, do NOT commit/push — the system handles version control.
- Do NOT touch secrets, keys, CI credentials, or delete files wholesale.
- If nothing safe and small is worth doing, make NO edits.

Reply with: a one-line summary of what you changed (or "no change" and why),
then a 1-2 sentence PR description.
"""


class MaintenanceExecutor:
    """Runs one repo's maintenance per cron fire (see module docstring).

    Cron fires are dispatched sequentially by the scheduler (it awaits each fire
    before the next), so a single instance handles one repo at a time.
    """

    def __init__(
        self,
        *,
        responder: _Responder,
        storage_root: Path,
        default_channel_id: str = "",
        repos_root: Path | None = None,
        branch_prefix: str = "alex/maint",
        skip_repos: frozenset[str] = _DEFAULT_SKIP,
        runner: CommandRunner | None = None,
        test_timeout: float = _DEFAULT_TEST_TIMEOUT,
    ) -> None:
        self._responder = responder
        self._maint_dir = storage_root / "maintenance"
        self._default_channel = default_channel_id
        self._repos_root = repos_root
        self._prefix = branch_prefix
        self._skip = skip_repos
        self._run = runner or _default_runner
        self._test_timeout = test_timeout

    # -- queue state ---------------------------------------------------------

    def _state_path(self, repos_root: Path) -> Path:
        return self._maint_dir / f"{repos_root.name}.queue.json"

    def _log_path(self, repos_root: Path) -> Path:
        return self._maint_dir / f"{repos_root.name}.log"

    def _load_queue(self, repos_root: Path) -> list[str]:
        try:
            state = json.loads(self._state_path(repos_root).read_text())
            queue = state.get("queue")
            if isinstance(queue, list) and queue:
                return [str(r) for r in queue]
        except (json.JSONDecodeError, OSError):
            pass
        if not repos_root.exists():
            return []
        return sorted(p.name for p in repos_root.iterdir() if (p / ".git").exists())

    def _save_queue(self, repos_root: Path, queue: list[str]) -> None:
        self._maint_dir.mkdir(parents=True, exist_ok=True)
        self._state_path(repos_root).write_text(json.dumps({"queue": queue}))

    def _append_log(self, repos_root: Path, line: str) -> None:
        self._maint_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path(repos_root).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- git helpers ---------------------------------------------------------

    async def _git(
        self, repo: Path, *args: str, timeout: float = _GIT_TIMEOUT
    ) -> tuple[int, str, str]:
        return await self._run(["git", "-C", str(repo), *args], None, timeout)

    async def _default_branch(self, repo: Path) -> str:
        code, out, _ = await self._git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        if code == 0 and out.strip():
            return out.strip().removeprefix("origin/")
        for cand in ("main", "master"):
            code, _, _ = await self._git(repo, "rev-parse", "--verify", "--quiet", f"origin/{cand}")
            if code == 0:
                return cand
        code, out, _ = await self._git(repo, "branch", "--show-current")
        return out.strip() or "main"

    # -- main entrypoint -----------------------------------------------------

    async def __call__(self, resource: dict[str, Any]) -> dict[str, Any]:
        meta = resource.get("metadata") or {}
        channel_id = resource.get("channel_id") or meta.get("channel_id") or self._default_channel
        session_id = str(resource.get("session_id") or "")

        root_str = meta.get("repos_root")
        repos_root = Path(root_str) if root_str else self._repos_root
        if repos_root is None:
            return {"status": "skipped", "reason": "no 'repos_root' for maintenance"}
        if not channel_id:
            return {"status": "skipped", "reason": "no 'channel_id' for delivery"}

        queue = self._load_queue(repos_root)
        if not queue:
            return {"status": "skipped", "reason": "no repos in maintenance queue"}

        repo_name = queue[0]
        # Rotate immediately and persist so a failure never pins the same repo.
        self._save_queue(repos_root, queue[1:] + [repo_name])

        try:
            return await self._maintain(repo_name, repos_root, channel_id, session_id)
        except Exception as exc:  # noqa: BLE001 - harness must never raise to the scheduler
            summary = f"error: {exc}"
            self._append_log(repos_root, f"- {repo_name} — {summary} ({_now_date()})")
            await self._notify(channel_id, f"{repo_name} — {summary}")
            return {"status": "error", "error": str(exc)}

    async def _notify(self, channel_id: str, text: str) -> None:
        await self._responder.deliver_text(channel_id, f"[maint] {text}")

    async def _finish(
        self,
        repo_name: str,
        repos_root: Path,
        channel_id: str,
        summary: str,
        status: str = "completed",
    ) -> dict[str, Any]:
        self._append_log(repos_root, f"- {repo_name} — {summary} ({_now_date()})")
        await self._notify(channel_id, f"{repo_name} — {summary}")
        return {"status": status, "output": f"{repo_name} — {summary}"}

    async def _maintain(
        self, repo_name: str, repos_root: Path, channel_id: str, session_id: str
    ) -> dict[str, Any]:
        if repo_name in self._skip:
            return await self._finish(
                repo_name,
                repos_root,
                channel_id,
                "skipped: sensitive repo (manual maintenance only)",
                "skipped",
            )

        repo = repos_root / repo_name
        code, _, _ = await self._git(repo, "rev-parse", "--is-inside-work-tree")
        if code != 0:
            return await self._finish(
                repo_name, repos_root, channel_id, "skipped: not a git repo", "skipped"
            )

        code, out, _ = await self._git(repo, "status", "--porcelain")
        if code != 0 or out.strip():
            return await self._finish(
                repo_name, repos_root, channel_id, "skipped: working tree not clean", "skipped"
            )

        default = await self._default_branch(repo)
        await self._git(repo, "fetch", "origin", default, "--quiet")
        branch = f"{self._prefix}-{repo_name}"
        code, _, err = await self._git(repo, "checkout", "-B", branch, f"origin/{default}")
        if code != 0:
            code, _, err = await self._git(repo, "checkout", "-B", branch)
            if code != 0:
                return await self._finish(
                    repo_name,
                    repos_root,
                    channel_id,
                    f"skipped: cannot branch ({err.strip()[:80]})",
                    "skipped",
                )

        prompt = _AGENT_PROMPT.format(repo=repo_name, path=repo)
        reply = await self._responder.run_prompt(
            channel_id, prompt, session_id=session_id, deliver=False
        )
        reply = (reply or "").strip()

        await self._git(repo, "add", "-A")
        code, _, _ = await self._git(repo, "diff", "--cached", "--quiet")
        if code == 0:
            await self._git(repo, "checkout", default)
            await self._git(repo, "branch", "-D", branch)
            return await self._finish(repo_name, repos_root, channel_id, "no change this run")

        tests = await self._run_tests(repo)
        subject = _commit_subject(reply, repo_name)
        await self._git(repo, "commit", "-m", subject)

        summary = await self._publish(repo, branch, default, subject, reply, tests)
        await self._git(repo, "checkout", default)
        return await self._finish(repo_name, repos_root, channel_id, summary)

    async def _run_tests(self, repo: Path) -> str:
        cmd = _detect_test_cmd(repo)
        if cmd is None:
            return "none"
        code, _, _ = await self._run(cmd, repo, self._test_timeout)
        return "pass" if code == 0 else "fail"

    async def _publish(
        self, repo: Path, branch: str, default: str, subject: str, reply: str, tests: str
    ) -> str:
        code, _, err = await self._git(repo, "push", "-u", "origin", branch, "--force-with-lease")
        if code != 0:
            return f"committed locally; push failed ({err.strip()[:80]}); tests={tests}"

        auth, _, _ = await self._run(["gh", "auth", "status"], repo, 20.0)
        if auth != 0:
            return f"branch {branch} pushed; PR skipped (gh not authenticated); tests={tests}"

        body = (reply or subject) + f"\n\nTests: {tests}.\n\n_Opened by alex-chen maintenance._"
        pr_args = [
            "gh",
            "pr",
            "create",
            "--head",
            branch,
            "--base",
            default,
            "--title",
            subject,
            "--body",
            body,
        ]
        if tests == "fail":
            pr_args.append("--draft")
        code, out, err = await self._run(pr_args, repo, 30.0)
        if code != 0:
            return f"branch {branch} pushed; PR failed ({err.strip()[:80]}); tests={tests}"
        url = out.strip().splitlines()[-1] if out.strip() else ""
        draft = " (draft, tests fail)" if tests == "fail" else ""
        return f"PR opened{draft}: {url}; tests={tests}"
