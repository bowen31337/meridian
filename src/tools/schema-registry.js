/**
 * Tool Schema Registry with JSON Schema validation
 * Based on Anthropic's managed agents pattern
 */

const Ajv = require('ajv');
const ajv = new Ajv();

const TOOL_REGISTRY = {
  exec: {
    name: 'exec',
    description: 'Execute shell commands',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Shell command to execute' },
        workdir: { type: 'string', description: 'Working directory' },
        timeout: { type: 'number', description: 'Timeout in seconds' },
        elevated: { type: 'boolean', description: 'Run with elevated privileges' },
        pty: { type: 'boolean', description: 'Use pseudo-terminal' }
      },
      required: ['command'],
      additionalProperties: false
    }
  },

  read: {
    name: 'read',
    description: 'Read file contents',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path' },
        offset: { type: 'number', description: 'Line offset' },
        limit: { type: 'number', description: 'Line limit' }
      },
      required: ['path'],
      additionalProperties: false
    }
  },

  write: {
    name: 'write',
    description: 'Write to file',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path' },
        content: { type: 'string', description: 'File content' }
      },
      required: ['path', 'content'],
      additionalProperties: false
    }
  },

  web_search: {
    name: 'web_search',
    description: 'Search the web',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        count: { type: 'number', description: 'Number of results' }
      },
      required: ['query'],
      additionalProperties: false
    }
  },

  kb_search: {
    name: 'kb_search',
    description: 'Search knowledge base by semantic similarity',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        limit: { type: 'number', description: 'Max results' }
      },
      required: ['query'],
      additionalProperties: false
    }
  }
};

function validateToolCall(toolName, params) {
  const toolDef = TOOL_REGISTRY[toolName];
  if (!toolDef) {
    return { valid: false, errors: [`Unknown tool: ${toolName}`] };
  }

  const valid = ajv.validate(toolDef.parameters, params);
  if (!valid) {
    return { valid: false, errors: ajv.errorsText().split('\n') };
  }

  return { valid: true };
}

function getToolSchema(toolName) {
  const toolDef = TOOL_REGISTRY[toolName];
  if (!toolDef) throw new Error(`Unknown tool: ${toolName}`);
  return toolDef;
}

function listTools() {
  return Object.values(TOOL_REGISTRY).map(tool => ({
    name: tool.name,
    description: tool.description
  }));
}

module.exports = {
  TOOL_REGISTRY,
  validateToolCall,
  getToolSchema,
  listTools
};
