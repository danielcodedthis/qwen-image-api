// A tiny event bus, so modules can react to each other without importing each other.
//   "status"  every /status poll       (data: status JSON, or null when offline)
//   "config"  /config loaded           (data: config JSON)
//   "job"     a job finished           (data: result item)
//   "route"   the view changed         (data: {view, id})

const handlers = {};

export function on(name, fn) { (handlers[name] ||= []).push(fn); }

export function emit(name, data) {
  for (const fn of handlers[name] || []) {
    try { fn(data); } catch (e) { console.error(`handler for "${name}" failed`, e); }
  }
}
