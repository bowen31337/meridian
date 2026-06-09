// Tutorial content for Meridian. Sourced from docs/PRD.md, docs/ARCHITECTURE.md,
// CLAUDE.md and the meridian-cli command surface.

export type NavItem = { id: string; index: string; label: string };

export const NAV: NavItem[] = [
  { id: "intro", index: "00", label: "Orientation" },
  { id: "model", index: "01", label: "Mental model" },
  { id: "quickstart", index: "02", label: "Quickstart" },
  { id: "tracks", index: "03", label: "Two front doors" },
  { id: "concepts", index: "04", label: "Core concepts" },
  { id: "reference", index: "05", label: "Command index" },
];

export const PRINCIPLES = [
  {
    k: "I",
    title: "The session is the truth",
    body: "Every session is an append-only NDJSON event log. Relational tables and the live session phase are just projections you can rebuild. Nothing is lost; everything replays.",
    tag: "append-only",
  },
  {
    k: "II",
    title: "The harness is cattle",
    body: "No session state lives in the worker loop. Any harness can wake(session_id) and resume from the log — crash mid-run, restart, and pick up exactly where you left off.",
    tag: "stateless",
  },
  {
    k: "III",
    title: "One dispatch surface",
    body: "Built-in tools, MCP servers, HTTP tools, containers, subprocesses — every executable action flows through the Sandbox's execute(name, input) → result. One door, audited.",
    tag: "execute()",
  },
  {
    k: "IV",
    title: "Capabilities by intersection",
    body: "Tools declare the caps they require; agents declare the caps they're granted. Dispatch enforces the intersection. No upward escalation, ever.",
    tag: "least-privilege",
  },
];

export type Step = {
  n: string;
  title: string;
  note: string;
  code: string;
  lang?: string;
};

export const QUICKSTART: Step[] = [
  {
    n: "01",
    title: "Install the workspace",
    note: "Python is managed by uv; the UI side by pnpm. uv auto-syncs the workspace on first use.",
    code: `# from the repo root
uv sync                 # provisions the Python workspace
pnpm install            # provisions the TS / UI side`,
  },
  {
    n: "02",
    title: "Point a model at it",
    note: "Declare providers in ~/.meridian/config.yml. Secrets use secret_ref:// indirection through a Vault — never inline keys.",
    code: `# ~/.meridian/config.yml
providers:
  anthropic:
    mode: oauth          # use a Claude Pro/Max subscription
  # or: mode: api_key, key: secret_ref://vault/default/anthropic

routing:
  default: anthropic/claude-opus-4`,
  },
  {
    n: "03",
    title: "Wake the daemon",
    note: "meridiand binds 127.0.0.1:7432 by default. It serves the /v1 API and Meridian extensions under /v1/x.",
    code: `python -m meridiand            # or: make dev  (daemon + UI)

meridian meridianconfig validate   # sanity-check your config`,
  },
  {
    n: "04",
    title: "Say hello",
    note: "Create an agent, open a session, send a turn, and stream the event log to your terminal.",
    code: `meridian agents create --data '{"name":"scout","model":"anthropic/claude-opus-4"}'
meridian sessions create --data '{"agent_id":"<id>","input":"What can you do?"}'
meridian meridianrun <session_id>   # live TTY stream of the session`,
  },
];

export type TrackStep = { label: string; detail: string; code?: string };
export type Track = {
  key: string;
  kicker: string;
  title: string;
  blurb: string;
  accent: "signal" | "teal";
  steps: TrackStep[];
};

export const TRACKS: Track[] = [
  {
    key: "coding",
    kicker: "Front door A",
    title: "The coding agent",
    accent: "signal",
    blurb:
      "At the keyboard, Meridian is a dev-loop agent. It runs in your repo, dispatches tools through the sandbox, and streams every token, tool call and phase change straight to your terminal — or to a full-screen TUI.",
    steps: [
      {
        label: "Scaffold the project workspace",
        detail:
          "Initialise the uv workspace so the agent has a structured place to read and write code.",
        code: "meridian workspace-init --root .",
      },
      {
        label: "Grant a capability-scoped agent",
        detail:
          "Define which tools the agent may touch. Grants intersect with each tool's required caps at dispatch time.",
        code: `meridian agents create --data '{
  "name": "dev",
  "model": "anthropic/claude-opus-4",
  "grants": ["fs.read", "fs.write", "shell.exec"]
}'`,
      },
      {
        label: "Open a coding session",
        detail: "The session is the truth — an append-only log you can replay or fork later.",
        code: `meridian sessions create --data '{
  "agent_id": "<id>",
  "input": "Add retry logic to the HTTP client and run the tests"
}'`,
      },
      {
        label: "Watch it work",
        detail:
          "meridianrun renders streaming tokens, inlined tool calls, collapsed thinking blocks and colour-coded phase transitions. Ctrl-C detaches without killing the run.",
        code: "meridian meridianrun <session_id>",
      },
      {
        label: "Or drive the full TUI",
        detail:
          "A keyboard-first terminal UI for browsing sessions, approving tool calls and steering the loop.",
        code: "meridian meridiantui",
      },
    ],
  },
  {
    key: "assistant",
    kicker: "Front door B",
    title: "The personal assistant",
    accent: "teal",
    blurb:
      "Away from the keyboard, the same daemon answers on the channels you already use. A Telegram DM, a Slack mention, or a webhook can wake a sleeping session — and the agent remembers you across all of them.",
    steps: [
      {
        label: "Attach a channel",
        detail:
          "CLI, Telegram, Slack, Discord, WhatsApp, iMessage, web and webhook all land on one gateway. One session is the truth; each channel is just a viewer.",
        code: `meridian channels create --data '{
  "kind": "telegram",
  "agent_id": "<id>",
  "credentials": "secret_ref://vault/default/telegram"
}'`,
      },
      {
        label: "Give it a memory",
        detail:
          "A MemoryStore lets the assistant carry context between conversations; a UserProfile holds who you are and how you like to be helped.",
        code: `meridian memory-stores create --data '{"name":"daily"}'
meridian user-profiles create --data '{"name":"bowen","tz":"Australia/Sydney"}'`,
      },
      {
        label: "Let things wake it up",
        detail:
          "Register a webhook so an external event — a PR, a calendar ping, an alert — can resume a session and act on your behalf.",
        code: `meridian webhooks create --data '{
  "event": "github.pull_request",
  "agent_id": "<id>"
}'`,
      },
      {
        label: "Schedule recurring work",
        detail:
          "A cron entry runs the agent on a timetable — your morning brief, an end-of-day summary into the MemoryStore.",
        code: `meridian cron create --data '{
  "schedule": "0 8 * * *",
  "agent_id": "<id>",
  "input": "Summarise today's calendar and unread threads"
}'`,
      },
      {
        label: "Stay observable",
        detail:
          "Every turn from every channel is the same append-only event log. Replay it, audit it, or follow it live.",
        code: "meridian sessions list && meridian meridianrun <session_id>",
      },
    ],
  },
];

export type Concept = {
  title: string;
  ref: string;
  body: string;
};

export const CONCEPTS: Concept[] = [
  {
    title: "Model Router",
    ref: "provider-polymorphic",
    body: "Anthropic (api_key + oauth), OpenAI, OpenRouter and Ollama in v1, configured declaratively in YAML. Routing rules pick a provider per request; swap models without touching agent code.",
  },
  {
    title: "Skill Forge",
    ref: "self-improving",
    body: "Distills durable, reusable skills out of real agent trajectories using the agentskills.io standard. Good runs compound into capabilities the next session inherits.",
  },
  {
    title: "Environment Manager",
    ref: "7 backends",
    body: "Sandbox execution across local, Docker, SSH, Modal, Daytona, Vercel and Singularity. The agent's code runs where you point it; dispatch stays identical.",
  },
  {
    title: "Vaults & secret_ref",
    ref: "no inline keys",
    body: "Credentials live behind secret_ref://vault/{id}/{key} indirection. Providers and channels reference secrets; the values never enter config files or the event log.",
  },
  {
    title: "Deterministic replay",
    ref: "the log is enough",
    body: "Because the session is an append-only log, any historical run replays exactly. Reproduce a bug, fork from a checkpoint, or audit what an agent actually did.",
  },
  {
    title: "Hot config reload",
    ref: "validate-then-swap",
    body: "POST /v1/x/config/reload or send SIGHUP. The daemon validates the new config, then atomically swaps it in — no dropped sessions, no restart.",
  },
];

export type Command = {
  group: string;
  cmd: string;
  desc: string;
};

export const COMMANDS: Command[] = [
  {
    group: "core",
    cmd: "meridian agents <list|get|create|update|delete>",
    desc: "Manage capability-scoped agents.",
  },
  {
    group: "core",
    cmd: "meridian sessions <list|get|create|update|delete>",
    desc: "Open and inspect the append-only session log.",
  },
  {
    group: "core",
    cmd: "meridian sessions archive <id>",
    desc: "Archive a session (restore brings it back).",
  },
  {
    group: "core",
    cmd: "meridian meridianrun <session_id>",
    desc: "Stream a session's events to the terminal, live.",
  },
  {
    group: "core",
    cmd: "meridian meridiantui",
    desc: "Launch the full-screen keyboard-first TUI.",
  },
  {
    group: "assistant",
    cmd: "meridian channels create --data '{...}'",
    desc: "Attach Telegram / Slack / Discord / web / webhook.",
  },
  {
    group: "assistant",
    cmd: "meridian memory-stores ...",
    desc: "Cross-conversation memory for the assistant.",
  },
  {
    group: "assistant",
    cmd: "meridian user-profiles ...",
    desc: "Who you are and how you like to be helped.",
  },
  { group: "assistant", cmd: "meridian webhooks ...", desc: "Let external events wake a session." },
  { group: "assistant", cmd: "meridian cron ...", desc: "Run an agent on a schedule." },
  { group: "platform", cmd: "meridian skills ...", desc: "Manage forged + authored skills." },
  {
    group: "platform",
    cmd: "meridian environments ...",
    desc: "Configure sandbox execution backends.",
  },
  { group: "platform", cmd: "meridian vaults ...", desc: "Manage secrets behind secret_ref://." },
  {
    group: "platform",
    cmd: "meridian files <list|get|create|update|delete>",
    desc: "Daemon-managed files.",
  },
  {
    group: "platform",
    cmd: "meridian files upload <path>",
    desc: "Upload a binary blob to the daemon.",
  },
  { group: "platform", cmd: "meridian hooks ...", desc: "Pre/post dispatch hooks and verdicts." },
  {
    group: "platform",
    cmd: "meridian imports <openclaw|hermes> <path>",
    desc: "Import data from OpenClaw or Hermes.",
  },
  {
    group: "config",
    cmd: "meridian meridianconfig validate",
    desc: "Validate ~/.meridian/config.yml.",
  },
  {
    group: "config",
    cmd: "meridian meridianconfig migrate",
    desc: "Migrate an older config to the current schema.",
  },
  {
    group: "config",
    cmd: "meridian workspace-init --root .",
    desc: "Initialise the uv workspace at the repo root.",
  },
];

export const COMMAND_GROUPS = ["all", "core", "assistant", "platform", "config"] as const;
