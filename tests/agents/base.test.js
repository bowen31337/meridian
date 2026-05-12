const Agent = require('../../src/agents/base');
describe('Agent', () => {
  test('should create agent', () => {
    const agent = new Agent({ name: 'Test' });
    expect(agent.name).toBe('Test');
  });
});
