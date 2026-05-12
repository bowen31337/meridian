const MemoryStore = require('../../src/memory/store');
describe('MemoryStore', () => {
  test('should return null for missing', async () => {
    const store = new MemoryStore();
    const result = await store.loadMemory('nonexistent');
    expect(result).toBeNull();
  });
});
