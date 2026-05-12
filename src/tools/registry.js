class ToolRegistry {
  constructor() { this.tools = new Map(); }
  register(tool) {
    if (!tool.name || !tool.execute) throw new Error('Tool must have name and execute');
    this.tools.set(tool.name, tool);
  }
}
module.exports = ToolRegistry;
