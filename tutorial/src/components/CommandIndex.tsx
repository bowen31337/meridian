import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { COMMANDS, COMMAND_GROUPS } from "../content";

export function CommandIndex() {
  const [group, setGroup] = useState<(typeof COMMAND_GROUPS)[number]>("all");
  const [q, setQ] = useState("");

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return COMMANDS.filter((c) => group === "all" || c.group === group).filter(
      (c) =>
        needle === "" ||
        c.cmd.toLowerCase().includes(needle) ||
        c.desc.toLowerCase().includes(needle),
    );
  }, [group, q]);

  return (
    <div>
      <div className="mb-7 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap gap-2">
          {COMMAND_GROUPS.map((g) => (
            <button
              key={g}
              type="button"
              onClick={() => setGroup(g)}
              className={`mono rounded-full border px-3.5 py-1.5 text-[0.7rem] uppercase tracking-widest transition-colors ${
                group === g
                  ? "border-signal/50 bg-signal/10 text-signal"
                  : "border-line text-muted hover:text-paper-dim"
              }`}
            >
              {g}
            </button>
          ))}
        </div>
        <div className="relative">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="filter commands…"
            className="mono w-full rounded-full border border-line bg-ink-2/60 px-4 py-2 text-sm text-paper outline-none placeholder:text-faint focus:border-signal/50 md:w-64"
          />
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-line">
        {rows.map((c, i) => (
          <motion.div
            key={c.cmd}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: Math.min(i * 0.025, 0.4) }}
            className="grid grid-cols-1 items-baseline gap-1 border-b border-line px-5 py-4 transition-colors last:border-b-0 hover:bg-ink-2/50 md:grid-cols-[1.3fr_1fr] md:gap-6"
          >
            <code className="mono text-[0.8rem] text-signal-soft">{c.cmd}</code>
            <span className="text-sm text-muted">{c.desc}</span>
          </motion.div>
        ))}
        {rows.length === 0 && (
          <div className="mono px-5 py-8 text-center text-sm text-faint">
            no commands match “{q}”
          </div>
        )}
      </div>
    </div>
  );
}
