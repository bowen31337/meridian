class Agent {
  constructor(config = {}) {
    this.name = config.name || 'Anonymous Agent';
    this.memory = config.memory || {};
    this.conversationHistory = [];
  }
  async execute(prompt) {
    this.conversationHistory.push({
      role: 'user',
      content: prompt,
      timestamp: new Date().toISOString()
    });
    return { role: 'assistant', content: `Response from ${this.name}` };
  }
}
module.exports = Agent;
