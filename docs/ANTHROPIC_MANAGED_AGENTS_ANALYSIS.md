# Anthropic Managed Agent Architecture Deep Dive
## How to Build a Better OpenClaw

**Date:** 2026-05-12  
**Source:** https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/resources/beta

---

## 1. The Core Architecture

### What Anthropic Built (From Beta SDK)

Anthropic has created a **managed agent system** that lives server-side (with local SDK bindings). The key insight: **agents are not ephemeral — they are persistent, versioned objects**.

```
┌─────────────────────┐
│  Agent Registry     │  (persisted, versioned)
│  - ID               │
│  - Instructions     │
│  - Tool Definitions │
│  - Model            │
│  - Parameters       │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Thread (Context)   │  (akin to a conversation)
│  - Agent ID         │
│  - Messages         │
│  - State            │
│  - History          │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Run (Execution)    │  (agentic loop)
│  - Status           │
│  - Tool Calls       │
│  - Messages         │
│  - Error Handling   │
└─────────────────────┘
```

### Key Classes (SDK Pattern)

1. **Agent** — persistent config object
   - `id`, `created_at`, `updated_at`
   - `model` (e.g., `claude-opus-4-1-20250805`)
   - `name`, `description`
   - `instructions` (system prompt)
   - `tools` (array of tool definitions)
   - `max_tokens` — control token budget
   - `temperature` — sampling control

2. **Tool Definition** — JSON schema binding
   ```python
   Tool = {
       "type": "function",
       "function": {
           "name": "example_tool",
           "description": "What it does",
           "parameters": {
               "type": "object",
               "properties": { ... },
               "required": [ ... ]
           }
       }
   }
   ```

3. **Thread** — stateful conversation container
   - Agent stays attached
   - Messages accumulate
   - Context is never lost
   - Can be paused/resumed

4. **Run** — loop execution
   - Agent + Thread → Run
   - Handles: `QUEUED` → `IN_PROGRESS` → `COMPLETED` / `FAILED`
   - Tool invocation pipeline
   - Streaming support

---

## 2. What OpenClaw Is Missing

### Problem 1: Session Ephemeralness
**OpenClaw:** Sessions are ephemeral. Each turn, you're (mostly) starting fresh.

```javascript
// OpenClaw today
→ User message
→ Load MEMORY.md (manually)
→ Query ClawMemory (manually)
→ Agent runs
→ Session ends
→ Everything forgotten (except files you wrote)
```

**Anthropic:** Agents + Threads are persistent.

```python
# Anthropic SDK
agent = client.beta.agents.create(
    name="my-agent",
    instructions="You are...",
    tools=[tool_defs],
    model="claude-opus-4-1"
)
thread = client.beta.threads.create()
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    agent_id=agent.id
)
# Later:
run = client.beta.threads.runs.retrieve(thread_id, run_id)
# Context is **never** lost
```

**Impact:** You never have to reload MEMORY.md. Context is always hot.

---

### Problem 2: Tool Definitions Are Ad-Hoc
**OpenClaw:** Tools are discovered at runtime; schemas are hand-coded in tool descriptions.

```json
// OpenClaw tool (approximate)
{
  "name": "exec",
  "description": "Execute shell commands with background continuation...",
  // No machine-readable schema — just prose
}
```

**Anthropic:** Tools have **strict JSON schemas** that are validated before tool calls.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run"
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory"
                    }
                },
                "required": ["command"]
            }
        }
    }
]
```

**Impact:** 
- Agent can reason about tool constraints *before* calling them
- No more "tool not found" errors at runtime
- Validation happens in the SDK, not in OpenClaw

---

### Problem 3: No Agent Versioning
**OpenClaw:** Agent config is implicit (in your session). If you change your instructions, old sessions don't know.

**Anthropic:** Agents are versioned objects. Each agent has `created_at`, `updated_at`. You can create new agent versions and route new threads to the new agent.

```python
# Update agent instructions
agent = client.beta.agents.update(
    agent_id=agent.id,
    instructions="New instructions..."
)
# All new threads use new agent
# Old threads still have the old agent version (immutable)
```

**Impact:** Reproducibility, audit trails, A/B testing.

---

### Problem 4: No Knowledge Base Integration
**OpenClaw:** Files live in the workspace. You manually read them into context.

**Anthropic:** Beta SDK includes a **Files API** for knowledge base attachment.

```python
file_response = client.beta.files.upload(
    file=open("document.pdf", "rb")
)
agent = client.beta.agents.create(
    ...,
    tools=[
        {
            "type": "file_search",  # Built-in tool
            "file_search": {
                "max_num_results": 20
            }
        }
    ]
)
```

**Impact:** Agent can search docs without manual RAG. Semantic search is built-in.

---

### Problem 5: Run State Machine is Primitive
**OpenClaw:** Cron jobs + subagents are separate. No unified state machine.

**Anthropic:** Runs have a formal state machine with **retry semantics**.

```
QUEUED
  ↓
IN_PROGRESS
  ├→ Tool call required → REQUIRES_ACTION
  │  ↓
  │  (Submit tool result)
  │  ↓
  │  Back to IN_PROGRESS
  ↓
COMPLETED
  or
FAILED
  or
EXPIRED
  or
CANCELLED
```

Each `REQUIRES_ACTION` state can be retried without losing context.

**Impact:** Complex multi-step workflows with automatic retry + error recovery.

---

## 3. How to Build a Better OpenClaw

### Tier 1: Quick Wins (1-2 weeks)

#### 1a. Tool Schema Registry + Validation
**What:** Export all OpenClaw tools to JSON schema format. Validate before execution.

**How:**
```typescript
// openclaw/tools/registry.ts
export const toolRegistry = {
  exec: {
    description: "Execute shell commands",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string" },
        workdir: { type: "string" },
        timeout: { type: "number" },
        elevated: { type: "boolean" },
        pty: { type: "boolean" }
      },
      required: ["command"]
    }
  },
  read: { ... },
  write: { ... },
  // etc.
}

// Validation before tool call
const toolDef = toolRegistry[toolName];
if (!toolDef) throw new Error(`Unknown tool: ${toolName}`);
const valid = ajv.validate(toolDef.parameters, params);
if (!valid) throw new Error(`Invalid params: ${ajv.errorsText()}`);
```

**Payoff:** 
- Agent reasoning improves (it knows constraints)
- Fewer runtime errors
- Tool discovery is now machine-readable

**Effort:** 2 days (document all tools, wire validation)

---

#### 1b. Persistent Agent Registry
**What:** Move agent configs out of sessions into a versioned registry.

**How:**
```typescript
// ~/.openclaw/agents/registry.json
{
  "main": {
    "id": "agent-main-v1",
    "name": "main",
    "instructions": "...",
    "tools": ["exec", "read", "write", ...],
    "model": "claude-opus-4-6",
    "created_at": "2026-05-12T00:00:00Z",
    "updated_at": "2026-05-12T10:00:00Z"
  },
  "analyst": {
    "id": "agent-analyst-v1",
    "instructions": "...",
    ...
  }
}
```

Then, sessions bind to agent IDs, not ephemeral configs.

**Payoff:**
- Agent personality/style is persistent
- Easy to A/B test different agent versions
- Audit trail (who updated the agent when?)

**Effort:** 3 days (schema + migration + session binding)

---

#### 1c. Auto-Retry + Error Recovery
**What:** Formalize the `REQUIRES_ACTION` pattern Anthropic uses.

**How:**
```typescript
type RunState = 
  | "queued" 
  | "in-progress" 
  | "requires-action"  // ← NEW: tool call needed
  | "completed" 
  | "failed" 
  | "expired";

// When tool call fails, don't crash — ask for retry
if (toolCallResult.error) {
  run.state = "requires-action";
  run.lastError = toolCallResult.error;
  // User/system can resubmit the same run
}
```

**Payoff:**
- Resilience to transient tool failures
- No more "silent failure then panic" loops
- Easier to debug

**Effort:** 2 days (state machine refactor)

---

### Tier 2: Medium Effort (1-2 weeks)

#### 2a. Knowledge Base Integration
**What:** Build a simple file indexing + semantic search layer.

**How:**
```typescript
// openclaw/kb/index.ts
export class KnowledgeBase {
  async indexFile(path: string): Promise<void> {
    const content = await read(path);
    const chunks = chunkText(content);
    for (const chunk of chunks) {
      const embedding = await embed(chunk);
      await db.insert("kb_chunks", { path, chunk, embedding });
    }
  }

  async search(query: string, limit = 10): Promise<Chunk[]> {
    const queryEmbedding = await embed(query);
    return db.query(
      "SELECT * FROM kb_chunks ORDER BY DISTANCE(embedding, ?) LIMIT ?",
      [queryEmbedding, limit]
    );
  }
}
```

Then expose as a tool:
```typescript
{
  name: "kb_search",
  description: "Search the knowledge base by semantic similarity",
  parameters: {
    type: "object",
    properties: {
      query: { type: "string" },
      limit: { type: "number" }
    },
    required: ["query"]
  },
  handler: async (params) => kb.search(params.query, params.limit)
}
```

**Payoff:**
- Agent can learn from your docs without manual pasting
- Faster context retrieval
- Scales better than `MEMORY.md`

**Effort:** 1 week (embedding model setup + DB schema + search)

---

#### 2b. Thread-Like Persistent Context
**What:** Replace "sessions" with Anthropic-style "threads" that persist across gateway restarts.

**How:**
```typescript
// openclaw/threads/thread.ts
export interface Thread {
  id: string;
  agent_id: string;  // Bind to agent registry
  created_at: Date;
  updated_at: Date;
  messages: Message[];  // Full history
  metadata: Record<string, any>;
}

// Gateway startup: load all threads from disk
const threads = await loadThreadsFromDB();
for (const thread of threads) {
  // Thread context is hot
}

// New message arrives
const thread = await getThread(threadId);
const run = await createRun(thread.agent_id, thread.id, newMessage);
```

**Payoff:**
- No more `loadMEMORY.md` dance
- Context is always warm
- Easy to audit conversation history

**Effort:** 3 days (thread schema + DB migration + session adapter)

---

### Tier 3: High Impact (2-4 weeks)

#### 3a. Agentic Loop + Streaming
**What:** Formalize the run loop with proper streaming + backpressure.

**How:**
```typescript
// openclaw/agents/loop.ts
async function executeRun(run: Run): AsyncGenerator<RunEvent> {
  while (run.state !== "completed" && run.state !== "failed") {
    const completion = await streamCompletion({
      agent_id: run.agent_id,
      messages: run.messages,
      tools: agentRegistry[run.agent_id].tools
    });

    for await (const event of completion) {
      if (event.type === "tool_call") {
        run.state = "requires-action";
        yield { type: "tool_call_pending", tool_call: event };
        // Wait for tool result
        const result = await getToolResult(run.id, event.id);
        run.messages.push({ role: "user", content: result });
        run.state = "in-progress";
      } else if (event.type === "message_delta") {
        yield { type: "message_delta", delta: event.delta };
      } else if (event.type === "stop") {
        run.state = "completed";
        yield { type: "run_completed", run };
      }
    }
  }
}
```

**Payoff:**
- Real streaming (not just "buffer then dump")
- Backpressure handling (don't blow memory with massive contexts)
- Proper tool-call choreography

**Effort:** 2 weeks (streaming plumbing + backpressure)

---

#### 3b. Checkpoints + Resume
**What:** Save run state at checkpoints so long jobs can pause/resume.

**How:**
```typescript
// Every N steps or on user request
async function checkpoint(run: Run): Promise<void> {
  const snapshot = {
    run_id: run.id,
    state: run.state,
    messages: run.messages,
    tool_calls_pending: run.toolCallsPending,
    timestamp: Date.now()
  };
  await saveCheckpoint(run.id, snapshot);
}

// Later: resume from checkpoint
const snapshot = await loadCheckpoint(run.id);
const run = new Run(snapshot.run_id, snapshot);
run.state = "in-progress";
// Continue from where we left off
```

**Payoff:**
- 8-hour runs don't lose all progress on crash
- Can pause expensive operations mid-flight
- Better resource management

**Effort:** 3 days (checkpoint schema + recovery logic)

---

### Tier 4: Nice-to-Have (Infrastructure)

#### 4a. Agent Studio UI
**What:** A web UI for viewing/editing agents, threads, and runs.

**How:**
- React frontend
- Agent editor (instructions, tools, model)
- Thread viewer (conversation history)
- Run debugger (step through tool calls, see streaming)

**Payoff:** Visibility + debugging. Nice but not critical.

**Effort:** 2-3 weeks

---

## 4. Migration Path for OpenClaw

### Phase 1: Parallel (Week 1)
- Build tool schema registry alongside existing tools
- Create agent registry file
- Don't break existing sessions yet

### Phase 2: Bridge (Week 2)
- Sessions can bind to agent IDs
- Load agent config from registry on session start
- Still support old-style ephemeral configs

### Phase 3: Persist (Week 3)
- Introduce "threads" as first-class objects
- Migrate old sessions → threads
- Gateway loads threads on startup

### Phase 4: Deprecate (Week 4)
- Old session-only model is deprecated
- All new work uses agent + thread model

---

## 5. Proof of Concept: Tool Schema Registry

Here's a working example (TypeScript):

```typescript
// tools/registry.ts
import Ajv from "ajv";

const ajv = new Ajv();

export const toolRegistry = {
  exec: {
    name: "exec",
    description: "Execute shell commands",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "Shell command" },
        workdir: { type: "string", description: "Working directory" },
        timeout: { type: "number", description: "Timeout in seconds" },
        elevated: { type: "boolean", description: "Run with sudo" },
        pty: { type: "boolean", description: "Use PTY" }
      },
      required: ["command"],
      additionalProperties: false
    }
  },
  read: {
    name: "read",
    description: "Read file contents",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path" },
        offset: { type: "number", description: "Line offset" },
        limit: { type: "number", description: "Line limit" }
      },
      required: ["path"],
      additionalProperties: false
    }
  },
  write: {
    name: "write",
    description: "Write to file",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" }
      },
      required: ["path", "content"],
      additionalProperties: false
    }
  }
};

export function validateToolCall(
  toolName: string,
  params: unknown
): { valid: boolean; errors?: string[] } {
  const toolDef = toolRegistry[toolName as keyof typeof toolRegistry];
  if (!toolDef) {
    return { valid: false, errors: [`Unknown tool: ${toolName}`] };
  }

  const valid = ajv.validate(toolDef.parameters, params);
  if (!valid) {
    return { valid: false, errors: ajv.errorsText().split("\n") };
  }

  return { valid: true };
}

export function getToolSchema(toolName: string) {
  const toolDef = toolRegistry[toolName as keyof typeof toolRegistry];
  if (!toolDef) {
    throw new Error(`Unknown tool: ${toolName}`);
  }
  return toolDef;
}
```

Then, in the agent loop:

```typescript
// agent/loop.ts
const { valid, errors } = validateToolCall(toolName, params);
if (!valid) {
  throw new ToolValidationError(
    `Tool validation failed for ${toolName}: ${errors?.join("; ")}`
  );
}

// Safe to call
const result = await invokeTool(toolName, params);
```

---

## 6. Action Items

### For You (Bowen)
- [ ] Review Anthropic's agents.py in full
- [ ] Decide: which tier to tackle first?
- [ ] Wire up tool schema registry as PoC
- [ ] Create agent registry file format

### For the Codebase
1. **Short term:** Add tool validation layer (1 day)
2. **Medium term:** Agent registry + thread model (2 weeks)
3. **Long term:** Full agentic loop with checkpoints (1 month)

### Questions to Answer
- Should OpenClaw keep the current "session" model, or fully migrate to "thread"?
- Do you want strict tool validation, or permissive?
- Should agents be mutable or immutable (versioned)?

---

## References

- [Anthropic Python SDK Beta](https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/resources/beta)
- [Agents API Docs](https://docs.anthropic.com/en/api/agents) (official)
- OpenClaw current implementation: `/media/DATA/.openclaw/agents/main/agent`

---

**TL;DR:** Anthropic built a serverless agent platform. OpenClaw should steal:
1. Persistent agent configs (not ephemeral)
2. Tool schema validation (not runtime discovery)
3. Thread-like context (not session restart dance)
4. Formal run state machine (with retry semantics)
5. Knowledge base integration (not manual file reading)

Start with #1 and #2. They're quick wins with high payoff.
