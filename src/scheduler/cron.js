class CronScheduler {
  constructor() { this.jobs = new Map(); }
  addJob(jobId, cronExpr, callback) {
    this.jobs.set(jobId, { cronExpr, callback });
  }
}
module.exports = CronScheduler;
