# Meridian Architecture

## System Design
Agent Runtime → Tool Registry → Persistence Layer

## Core Components
1. **Agent Runtime** - Session management, conversation state, tool orchestration
2. **Tool Registry** - Dynamic tool loading, execution sandboxing
3. **Memory System** - Long-term memory, daily logs, vector integration
4. **Scheduler** - Cron scheduling, task management
5. **Channel Adapters** - Telegram, Discord, Signal integration

## Extensibility Points
- Tool development via registry
- Channel integration via adapters
- Memory backend swapping
- Custom execution policies
