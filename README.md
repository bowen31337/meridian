# Meridian

An extensible agent platform with persistent context, tool schema validation, and knowledge base integration — inspired by Anthropic's managed agents architecture.

## Architecture

```
Agent Registry (persistent configs)
    ↓
Thread (stateful context)
    ↓
Run (formal state machine)
    ↓
Tool Execution (schema-validated)
    ↓
Knowledge Base (semantic search)
```

### Key Concepts

- **Agents**: Persistent, versioned configurations (not ephemeral)
- **Threads**: Stateful conversation containers that survive restarts
- **Runs**: Formal state machine (`QUEUED` → `IN_PROGRESS` → `REQUIRES_ACTION` → `COMPLETED`)
- **Tools**: JSON schema-validated with pre-execution validation
- **Knowledge Base**: Semantic search over documents

## Quick Start

```bash
npm install
npm run dev
```

## Core Modules

### `src/agents/`
- `base.js` - Base agent class
- `registry.js` - Persistent agent registry (create, update, list agents)

### `src/tools/`
- `registry.js` - Tool registry
- `schema-registry.js` - JSON schema validation for all tools

### `src/threads/`
- `thread.js` - Thread model and persistent storage

### `src/runs/`
- `run.js` - Run state machine with formal state transitions

### `src/kb/`
- `knowledge-base.js` - Knowledge base with keyword indexing

### `src/memory/`
- `store.js` - Persistent memory storage

## Usage

### Create an Agent

```javascript
const AgentRegistry = require('./src/agents/registry');

const registry = new AgentRegistry();
await registry.load();

const agent = await registry.create({
  name: 'assistant',
  description: 'My personal assistant',
  instructions: 'You are a helpful assistant...',
  tools: ['exec', 'read', 'write', 'web_search', 'kb_search'],
  model: 'claude-opus-4-6'
});
```

### Create a Thread

```javascript
const { Thread, ThreadStore } = require('./src/threads/thread');

const store = new ThreadStore();
const thread = new Thread({
  id: 'thread-1',
  agent_id: agent.id,
  messages: []
});

thread.addMessage('user', 'Hello!');
await store.save(thread);
```

### Execute a Run

```javascript
const { Run, RUN_STATUS } = require('./src/runs/run');

const run = new Run({
  id: 'run-1',
  thread_id: 'thread-1',
  agent_id: agent.id
});

run.transition(RUN_STATUS.IN_PROGRESS);
// ... agent processes
run.transition(RUN_STATUS.REQUIRES_ACTION);
run.addToolCall('exec', { command: 'ls' }, 'tool-call-1');
// ... tool executes
run.submitToolResult('tool-call-1', { stdout: '...' });
run.transition(RUN_STATUS.COMPLETED);
```

### Validate Tool Calls

```javascript
const { validateToolCall } = require('./src/tools/schema-registry');

const { valid, errors } = validateToolCall('exec', { command: 'ls' });
if (!valid) {
  console.error('Tool validation failed:', errors);
}
```

### Search Knowledge Base

```javascript
const KB = require('./src/kb/knowledge-base');

const kb = new KB();
await kb.initialize();
await kb.indexFile('./docs/guide.md');

const results = kb.search('how to deploy', limit: 10);
console.log(results);
```

## Testing

```bash
npm test
npm run test:watch
npm run test:coverage
```

## Architecture Decisions

This implementation is based on [Anthropic's managed agents architecture](https://docs.anthropic.com/en/api/agents):

1. **Persistent Agents** — Agents are not ephemeral; they're versioned objects
2. **Schema-Validated Tools** — All tools must conform to JSON schema
3. **Stateful Threads** — Context survives across session boundaries
4. **Formal Runs** — State machine prevents invalid transitions
5. **Knowledge Base** — Built-in semantic search over docs

## Roadmap

- [ ] Real embedding-based semantic search (replace TF-IDF)
- [ ] Checkpoint/resume for long-running operations
- [ ] Agent Studio UI for debugging
- [ ] Real-time streaming via Server-Sent Events
- [ ] File-based knowledge base (PDF, DOCX support)

## Contributing

See CONTRIBUTING.md

## License

MIT
