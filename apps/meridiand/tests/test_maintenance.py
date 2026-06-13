"""Tests for the deterministic maintenance harness (MaintenanceExecutor).

Git / gh / test invocations are scripted through an injected runner, so no real
repository is mutated. The agent turn is faked too: the harness's change
detection is driven by the scripted ``git diff --cached`` exit code, not real IO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meridiand._maintenance import (
    MaintenanceExecutor,
    _commit_subject,
    _default_runner,
    _detect_test_cmd,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponder:
    def __init__(
        self, reply: str = "Tighten the README install steps", raise_exc: Exception | None = None
    ):
        self.reply = reply
        self.raise_exc = raise_exc
        self.prompts: list[tuple[str, str, str, bool]] = []
        self.delivered: list[tuple[str, str]] = []

    async def run_prompt(
        self, channel_id: str, prompt: str, *, session_id: str = "", deliver: bool = True
    ) -> str | None:
        self.prompts.append((channel_id, prompt, session_id, deliver))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.reply

    async def deliver_text(self, channel_id: str, text: str) -> None:
        self.delivered.append((channel_id, text))


def _key(args: list[str]) -> str:
    if args[0] == "gh":
        return "gh:auth" if args[1] == "auth" else "gh:pr"
    if args[0] == "git":
        sub = args[3]
        if sub == "rev-parse":
            return "is_repo" if "--is-inside-work-tree" in args else "verify"
        if sub == "diff":
            return "diff_cached"
        if sub == "branch":
            return "show-current" if "--show-current" in args else "branch-d"
        return sub
    return "test"


_DEFAULTS: dict[str, tuple[int, str, str]] = {
    "is_repo": (0, "", ""),
    "status": (0, "", ""),
    "symbolic-ref": (0, "origin/main", ""),
    "verify": (0, "", ""),
    "fetch": (0, "", ""),
    "checkout": (0, "", ""),
    "add": (0, "", ""),
    "diff_cached": (1, "", ""),  # changes present by default
    "commit": (0, "", ""),
    "push": (0, "", ""),
    "branch-d": (0, "", ""),
    "show-current": (0, "main", ""),
    "gh:auth": (0, "", ""),
    "gh:pr": (0, "https://github.com/clawinfra/x/pull/1\n", ""),
    "test": (0, "", ""),
}


class _FakeRunner:
    def __init__(self, **overrides: Any) -> None:
        self.calls: list[list[str]] = []
        self.overrides = overrides

    async def __call__(
        self, args: list[str], cwd: Path | None, timeout: float
    ) -> tuple[int, str, str]:
        self.calls.append(list(args))
        k = _key(args)
        resp = self.overrides.get(k, _DEFAULTS.get(k, (0, "", "")))
        return resp(args) if callable(resp) else resp

    def cmd_keys(self) -> list[str]:
        return [_key(a) for a in self.calls]

    def find(self, key: str) -> list[str] | None:
        for a in self.calls:
            if _key(a) == key:
                return a
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_root(
    tmp_path: Path, names: list[str], *, with_pyproject: list[str] | None = None
) -> Path:
    root = tmp_path / "clawinfra"
    root.mkdir()
    for n in names:
        (root / n / ".git").mkdir(parents=True)
    for n in with_pyproject or []:
        (root / n / "pyproject.toml").write_text("[tool.x]\n")
    return root


def _resource(root: Path | None, **meta: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"kind": "maintenance", "channel_id": "ch1"}
    if root is not None:
        m["repos_root"] = str(root)
    m.update(meta)
    return {"id": "cron_1", "session_id": "sess_1", "metadata": m}


def _exec(
    resp: _FakeResponder, runner: _FakeRunner, storage: Path, **kw: Any
) -> MaintenanceExecutor:
    return MaintenanceExecutor(responder=resp, storage_root=storage, runner=runner, **kw)


def _queue_on_disk(storage: Path, root: Path) -> list[str]:
    return json.loads((storage / "maintenance" / f"{root.name}.queue.json").read_text())["queue"]


def _seed_queue(storage: Path, root: Path, queue: list[str]) -> None:
    d = storage / "maintenance"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{root.name}.queue.json").write_text(json.dumps({"queue": queue}))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_opens_pr(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha", "beta"], with_pyproject=["alpha"])
    storage = tmp_path / "storage"
    resp = _FakeResponder()
    runner = _FakeRunner()
    out = await _exec(resp, runner, storage)(_resource(root))

    assert out["status"] == "completed"
    assert "PR opened" in out["output"]
    assert "pull/1" in out["output"]
    assert "tests=pass" in out["output"]

    # one edit-only agent turn, on the head repo, not delivered to the channel
    assert len(resp.prompts) == 1
    _, prompt, _, deliver = resp.prompts[0]
    assert deliver is False
    assert str(root / "alpha") in prompt

    # git plumbing
    push = runner.find("push")
    assert push is not None and "--force-with-lease" in push
    pr = runner.find("gh:pr")
    assert pr is not None and "--draft" not in pr and "main" in pr

    # rotated + logged + status delivered
    assert _queue_on_disk(storage, root) == ["beta", "alpha"]
    log = (storage / "maintenance" / f"{root.name}.log").read_text()
    assert "alpha — PR opened" in log
    assert resp.delivered and resp.delivered[0][1].startswith("[maint] alpha — PR opened")


async def test_no_change_run_deletes_branch(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    resp = _FakeResponder()
    runner = _FakeRunner(diff_cached=(0, "", ""))  # no staged changes
    out = await _exec(resp, runner, storage)(_resource(root))

    assert out["status"] == "completed"
    assert "no change this run" in out["output"]
    keys = runner.cmd_keys()
    assert "commit" not in keys
    assert "push" not in keys
    assert "branch-d" in keys  # work branch deleted


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------


async def test_sensitive_repo_skipped_without_git(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    _seed_queue(storage, root, ["claw-wallet", "alpha"])
    resp = _FakeResponder()
    runner = _FakeRunner()
    out = await _exec(resp, runner, storage)(_resource(root))

    assert out["status"] == "skipped"
    assert "sensitive repo" in out["output"]
    assert runner.calls == []  # never touched git
    assert _queue_on_disk(storage, root) == ["alpha", "claw-wallet"]


async def test_not_a_git_repo_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(is_repo=(128, "", "not a repo"))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "skipped"
    assert "not a git repo" in out["output"]


async def test_dirty_tree_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(status=(0, " M foo.py", ""))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "skipped"
    assert "working tree not clean" in out["output"]
    assert "checkout" not in runner.cmd_keys()


async def test_status_nonzero_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(status=(1, "", "fatal"))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "skipped"
    assert "working tree not clean" in out["output"]


async def test_cannot_branch_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(checkout=(1, "", "no such ref"))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "skipped"
    assert "cannot branch" in out["output"]


async def test_branch_falls_back_to_local(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"

    def checkout(args: list[str]) -> tuple[int, str, str]:
        return (1, "", "missing") if any(a.startswith("origin/") for a in args) else (0, "", "")

    runner = _FakeRunner(checkout=checkout)
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "PR opened" in out["output"]


# ---------------------------------------------------------------------------
# Default branch resolution
# ---------------------------------------------------------------------------


async def test_default_branch_via_rev_parse(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"

    def verify(args: list[str]) -> tuple[int, str, str]:
        return (0, "", "") if any("origin/main" in a for a in args) else (1, "", "")

    runner = _FakeRunner(**{"symbolic-ref": (1, "", "")}, verify=verify)
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    pr = runner.find("gh:pr")
    assert pr is not None and "main" in pr


async def test_default_branch_via_show_current(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(
        **{"symbolic-ref": (1, "", "")},
        verify=(1, "", ""),
        **{"show-current": (0, "trunk", "")},
    )
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    pr = runner.find("gh:pr")
    assert pr is not None and "trunk" in pr


# ---------------------------------------------------------------------------
# Publish branches: tests, gh auth, push/pr failures
# ---------------------------------------------------------------------------


async def test_tests_fail_opens_draft(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"], with_pyproject=["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(test=(1, "", "1 failed"))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "draft, tests fail" in out["output"]
    pr = runner.find("gh:pr")
    assert pr is not None and "--draft" in pr


async def test_gh_unauthenticated_skips_pr(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(**{"gh:auth": (1, "", "not logged in")})
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "gh not authenticated" in out["output"]
    assert runner.find("gh:pr") is None
    assert runner.find("push") is not None


async def test_push_failure_reported(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(push=(1, "", "remote rejected"))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "push failed" in out["output"]
    assert runner.find("gh:auth") is None


async def test_pr_create_failure_reported(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(**{"gh:pr": (1, "", "boom")})
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "PR failed" in out["output"]


async def test_pr_empty_stdout_url(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(**{"gh:pr": (0, "", "")})
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert "PR opened" in out["output"]


# ---------------------------------------------------------------------------
# Queue / config resolution
# ---------------------------------------------------------------------------


async def test_rotation_across_two_fires(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha", "beta"])
    storage = tmp_path / "storage"
    ex = _exec(_FakeResponder(), _FakeRunner(diff_cached=(0, "", "")), storage)
    await ex(_resource(root))
    assert _queue_on_disk(storage, root) == ["beta", "alpha"]
    await ex(_resource(root))
    assert _queue_on_disk(storage, root) == ["alpha", "beta"]


async def test_corrupt_state_falls_back_to_disk(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha", "beta"])
    storage = tmp_path / "storage"
    d = storage / "maintenance"
    d.mkdir(parents=True)
    (d / f"{root.name}.queue.json").write_text("{not json")
    runner = _FakeRunner(diff_cached=(0, "", ""))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["status"] == "completed"
    assert out["output"].startswith("alpha")  # disk order, head=alpha


async def test_empty_queue_list_falls_back_to_disk(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    _seed_queue(storage, root, [])
    runner = _FakeRunner(diff_cached=(0, "", ""))
    out = await _exec(_FakeResponder(), runner, storage)(_resource(root))
    assert out["output"].startswith("alpha")


async def test_no_repos_at_all_skipped(tmp_path: Path) -> None:
    root = tmp_path / "clawinfra"
    root.mkdir()
    storage = tmp_path / "storage"
    out = await _exec(_FakeResponder(), _FakeRunner(), storage)(_resource(root))
    assert out["status"] == "skipped"
    assert "no repos" in out["reason"]


async def test_nonexistent_repos_root_skipped(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    missing = tmp_path / "nope"  # never created
    out = await _exec(_FakeResponder(), _FakeRunner(), storage)(_resource(missing))
    assert out["status"] == "skipped"
    assert "no repos" in out["reason"]


async def test_missing_repos_root_skipped(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    out = await _exec(_FakeResponder(), _FakeRunner(), storage)(_resource(None))
    assert out["status"] == "skipped"
    assert "repos_root" in out["reason"]


async def test_repos_root_from_constructor_default(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    runner = _FakeRunner(diff_cached=(0, "", ""))
    ex = _exec(_FakeResponder(), runner, storage, repos_root=root)
    out = await ex(_resource(None))  # no repos_root in metadata
    assert out["output"].startswith("alpha")


async def test_missing_channel_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    res = {
        "id": "c",
        "session_id": "s",
        "metadata": {"kind": "maintenance", "repos_root": str(root)},
    }
    out = await _exec(_FakeResponder(), _FakeRunner(), storage)(res)
    assert out["status"] == "skipped"
    assert "channel_id" in out["reason"]


async def test_channel_from_top_level_field(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    resp = _FakeResponder()
    res = {
        "id": "c",
        "session_id": "s",
        "channel_id": "ch_top",
        "metadata": {"kind": "maintenance", "repos_root": str(root)},
    }
    runner = _FakeRunner(diff_cached=(0, "", ""))
    await _exec(resp, runner, storage)(res)
    assert resp.delivered and resp.delivered[0][0] == "ch_top"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


async def test_agent_failure_reported_not_raised(tmp_path: Path) -> None:
    root = _make_root(tmp_path, ["alpha"])
    storage = tmp_path / "storage"
    resp = _FakeResponder(raise_exc=RuntimeError("model down"))
    out = await _exec(resp, _FakeRunner(), storage)(_resource(root))
    assert out["status"] == "error"
    assert "model down" in out["error"]
    log = (storage / "maintenance" / f"{root.name}.log").read_text()
    assert "alpha — error" in log
    assert resp.delivered and "error" in resp.delivered[0][1]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_detect_test_cmd_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    assert _detect_test_cmd(tmp_path) == ["python", "-m", "pytest", "-q", "-x"]


def test_detect_test_cmd_pytest_ini(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("")
    assert _detect_test_cmd(tmp_path)[1:3] == ["-m", "pytest"]


def test_detect_test_cmd_cargo(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("")
    assert _detect_test_cmd(tmp_path) == ["cargo", "test", "--quiet"]


def test_detect_test_cmd_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("")
    assert _detect_test_cmd(tmp_path) == ["go", "test", "./..."]


def test_detect_test_cmd_npm(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    assert _detect_test_cmd(tmp_path) == ["npm", "test", "--silent"]


def test_detect_test_cmd_npm_without_test_script(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"build": "x"}}))
    assert _detect_test_cmd(tmp_path) is None


def test_detect_test_cmd_corrupt_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{not json")
    assert _detect_test_cmd(tmp_path) is None


def test_detect_test_cmd_none(tmp_path: Path) -> None:
    assert _detect_test_cmd(tmp_path) is None


def test_commit_subject_from_reply() -> None:
    assert _commit_subject("# Fix the parser\nmore", "alpha") == "chore(maint): Fix the parser"


def test_commit_subject_fallback_when_empty() -> None:
    assert _commit_subject("   \n  ", "alpha") == "chore(maint): routine maintenance for alpha"


# ---------------------------------------------------------------------------
# _default_runner (real subprocess)
# ---------------------------------------------------------------------------


async def test_default_runner_success() -> None:
    code, out, _ = await _default_runner(["echo", "hi"], None, 5.0)
    assert code == 0
    assert out.strip() == "hi"


async def test_default_runner_nonzero() -> None:
    code, _, _ = await _default_runner(["false"], None, 5.0)
    assert code == 1


async def test_default_runner_timeout() -> None:
    code, _, err = await _default_runner(["sleep", "5"], None, 0.05)
    assert code == 124
    assert "timed out" in err


async def test_default_runner_missing_binary() -> None:
    code, _, err = await _default_runner(["__no_such_binary_xyz__"], None, 5.0)
    assert code == 127
    assert "failed to start" in err
