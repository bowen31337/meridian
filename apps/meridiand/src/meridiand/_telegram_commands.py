"""Shared Telegram bot command menu (openclaw-style slash commands).

Single source of truth for the slash-command set so the registered menu and
the handler never drift:
  * the driver registers it with Telegram via ``setMyCommands`` (the chat's
    menu button then lists the commands and / triggers autocomplete);
  * the AgentResponder uses it to render ``/help`` and to decide which messages
    to handle locally instead of forwarding to the LLM.
"""

from __future__ import annotations

from typing import Any

# (command, description) — description is shown in Telegram's command menu and
# in /help. Telegram caps descriptions at 256 chars; keep them short.
BOT_COMMANDS: list[tuple[str, str]] = [
    ("start", "What I can do and how to talk to me"),
    ("help", "Show the list of commands"),
    ("new", "Start a fresh conversation (clear short-term context)"),
    ("remember", "Save a durable fact, e.g. /remember I live in Sydney"),
    ("whoami", "Show your paired identity on this channel"),
]

_INTRO = (
    "Hi, I'm your personal assistant. Message me normally and I'll help.\n\n"
    "You can also use slash commands — type / to open the menu:\n\n"
)


def help_text() -> str:
    """The body of /help: one line per command."""
    return "\n".join(f"/{name} — {desc}" for name, desc in BOT_COMMANDS)


def start_text() -> str:
    """The /start greeting, including the command list."""
    return _INTRO + help_text()


def set_my_commands_payload() -> dict[str, Any]:
    """Payload for the Telegram Bot API setMyCommands method."""
    return {"commands": [{"command": name, "description": desc} for name, desc in BOT_COMMANDS]}
