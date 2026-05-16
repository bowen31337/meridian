"""Grammar tests: parse() and parse_set() for dotted-name capability strings."""
from __future__ import annotations

import pytest

from sdk_capabilities import Capability, CapabilityParseError, parse, parse_set


# ---------------------------------------------------------------------------
# Successful parses
# ---------------------------------------------------------------------------

class TestParseSuccess:
    def test_bare_namespace_name(self) -> None:
        assert parse("exec.shell") == Capability("exec", "shell", None)

    def test_with_param(self) -> None:
        assert parse("fs.read[/home/*]") == Capability("fs", "read", "/home/*")

    def test_net_fetch_host(self) -> None:
        assert parse("net.fetch[api.example.com]") == Capability("net", "fetch", "api.example.com")

    def test_secret_read_with_slash_in_param(self) -> None:
        assert parse("secret.read[vault/my-secret]") == Capability("secret", "read", "vault/my-secret")

    def test_underscore_in_namespace(self) -> None:
        c = parse("my_ns.my_name")
        assert c.namespace == "my_ns"
        assert c.name == "my_name"

    def test_param_with_hyphen(self) -> None:
        assert parse("acp.outbound[agent-42]").param == "agent-42"

    def test_param_with_underscore(self) -> None:
        assert parse("memory.read[mem_id]").param == "mem_id"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert parse("  exec.shell  ") == Capability("exec", "shell", None)

    def test_digits_in_namespace_after_first_char(self) -> None:
        c = parse("ns1.name2")
        assert c.namespace == "ns1"
        assert c.name == "name2"


class TestCapabilityStr:
    def test_str_no_param(self) -> None:
        assert str(Capability("exec", "shell", None)) == "exec.shell"

    def test_str_with_param(self) -> None:
        assert str(Capability("fs", "read", "/home/*")) == "fs.read[/home/*]"

    def test_str_roundtrip(self) -> None:
        texts = [
            "exec.shell",
            "fs.read[/home/*]",
            "net.fetch[api.example.com]",
            "secret.read[vault/name]",
        ]
        for text in texts:
            assert str(parse(text)) == text


# ---------------------------------------------------------------------------
# Parse failures
# ---------------------------------------------------------------------------

class TestParseFailure:
    def test_empty_string(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("   ")

    def test_no_dot_separator(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("exec_shell")

    def test_uppercase_namespace(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("Exec.shell")

    def test_uppercase_name(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("exec.Shell")

    def test_namespace_starts_with_digit(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("1exec.shell")

    def test_name_starts_with_digit(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("exec.1shell")

    def test_unclosed_bracket(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("fs.read[/home/*")

    def test_empty_brackets(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("fs.read[]")

    def test_nested_brackets(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("fs.read[[/home]]")

    def test_three_dot_segments(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("a.b.c")

    def test_trailing_dot(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse("fs.")

    def test_leading_dot(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse(".read")

    def test_just_a_dot(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse(".")

    def test_error_message_includes_input(self) -> None:
        with pytest.raises(CapabilityParseError) as exc_info:
            parse("BAD_INPUT")
        assert "BAD_INPUT" in str(exc_info.value)

    def test_error_has_text_attribute(self) -> None:
        with pytest.raises(CapabilityParseError) as exc_info:
            parse("bad")
        assert exc_info.value.text == "bad"

    def test_error_has_reason_attribute(self) -> None:
        with pytest.raises(CapabilityParseError) as exc_info:
            parse("bad")
        assert exc_info.value.reason


# ---------------------------------------------------------------------------
# parse_set
# ---------------------------------------------------------------------------

class TestParseSet:
    def test_empty_iterable(self) -> None:
        assert parse_set([]) == frozenset()

    def test_multiple_capabilities(self) -> None:
        result = parse_set(["exec.shell", "fs.read[/home/*]", "net.listen"])
        assert result == frozenset([
            Capability("exec", "shell"),
            Capability("fs", "read", "/home/*"),
            Capability("net", "listen"),
        ])

    def test_deduplicates(self) -> None:
        result = parse_set(["exec.shell", "exec.shell"])
        assert result == frozenset([Capability("exec", "shell")])

    def test_raises_on_invalid_entry(self) -> None:
        with pytest.raises(CapabilityParseError):
            parse_set(["exec.shell", "BAD"])


# ---------------------------------------------------------------------------
# All 20 canonical system capabilities from the spec
# ---------------------------------------------------------------------------

class TestAllKnownCapabilities:
    @pytest.mark.parametrize("text,expected", [
        ("fs.read[glob]",          Capability("fs",      "read",    "glob")),
        ("fs.write[glob]",         Capability("fs",      "write",   "glob")),
        ("fs.delete[glob]",        Capability("fs",      "delete",  "glob")),
        ("net.fetch[host]",        Capability("net",     "fetch",   "host")),
        ("net.listen",             Capability("net",     "listen",  None)),
        ("exec.shell",             Capability("exec",    "shell",   None)),
        ("exec.sudo",              Capability("exec",    "sudo",    None)),
        ("exec.pty",               Capability("exec",    "pty",     None)),
        ("kb.read[scope]",         Capability("kb",      "read",    "scope")),
        ("kb.write[scope]",        Capability("kb",      "write",   "scope")),
        ("memory.read[mem_id]",    Capability("memory",  "read",    "mem_id")),
        ("memory.write[mem_id]",   Capability("memory",  "write",   "mem_id")),
        ("agent.spawn[agent_id]",  Capability("agent",   "spawn",   "agent_id")),
        ("agent.cancel",           Capability("agent",   "cancel",  None)),
        ("secret.read[vault/name]",Capability("secret",  "read",    "vault/name")),
        ("hook.invoke[name]",      Capability("hook",    "invoke",  "name")),
        ("channel.send[chan_id]",  Capability("channel", "send",    "chan_id")),
        ("channel.receive[chan_id]",Capability("channel","receive", "chan_id")),
        ("acp.outbound[target]",   Capability("acp",     "outbound","target")),
        ("acp.inbound[target]",    Capability("acp",     "inbound", "target")),
    ])
    def test_parses_correctly(self, text: str, expected: Capability) -> None:
        assert parse(text) == expected
