/**
 * Agent Registry - Persistent versioned agent configurations
 * Based on Anthropic's agents pattern
 */

const fs = require('fs').promises;
const path = require('path');

class AgentRegistry {
  constructor(registryPath = './agents-registry.json') {
    this.registryPath = registryPath;
    this.agents = new Map();
    this.initialized = false;
  }

  async load() {
    try {
      const data = await fs.readFile(this.registryPath, 'utf-8');
      const agents = JSON.parse(data);
      this.agents = new Map(Object.entries(agents));
      this.initialized = true;
    } catch {
      this.agents = new Map();
      this.initialized = true;
    }
  }

  async save() {
    const agentsObj = Object.fromEntries(this.agents);
    await fs.writeFile(this.registryPath, JSON.stringify(agentsObj, null, 2));
  }

  async create(config) {
    const agentId = `agent-${config.name}-v1`;
    const agent = {
      id: agentId,
      name: config.name,
      description: config.description || '',
      instructions: config.instructions || '',
      tools: config.tools || [],
      model: config.model || 'claude-opus-4-6',
      max_tokens: config.max_tokens || 4096,
      temperature: config.temperature || 0.7,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    this.agents.set(agentId, agent);
    await this.save();
    return agent;
  }

  async get(agentId) {
    return this.agents.get(agentId);
  }

  async update(agentId, updates) {
    const agent = this.agents.get(agentId);
    if (!agent) throw new Error(`Agent not found: ${agentId}`);
    
    const updated = {
      ...agent,
      ...updates,
      updated_at: new Date().toISOString()
    };
    this.agents.set(agentId, updated);
    await this.save();
    return updated;
  }

  async list() {
    return Array.from(this.agents.values());
  }

  async delete(agentId) {
    this.agents.delete(agentId);
    await this.save();
  }
}

module.exports = AgentRegistry;
