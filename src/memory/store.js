const fs = require('fs').promises;
const path = require('path');

class MemoryStore {
  constructor(baseDir = './memory') { this.baseDir = baseDir; }
  async loadMemory(key) {
    try {
      const data = await fs.readFile(path.join(this.baseDir, `${key}.json`), 'utf-8');
      return JSON.parse(data);
    } catch { return null; }
  }
}
module.exports = MemoryStore;
