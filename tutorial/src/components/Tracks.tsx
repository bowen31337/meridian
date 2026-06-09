import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { TRACKS } from "../content";
import { CodeBlock } from "./CodeBlock";

export function Tracks() {
  const [active, setActive] = useState(0);
  const track = TRACKS[active];
  const accent = track.accent === "signal" ? "text-signal" : "text-teal";
  const accentBg = track.accent === "signal" ? "bg-signal" : "bg-teal";

  return (
    <div>
      <div className="relative mb-12 flex w-full max-w-md gap-1 rounded-full border border-line bg-ink-2/60 p-1">
        {TRACKS.map((t, i) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setActive(i)}
            className="relative flex-1 rounded-full px-4 py-2.5 text-sm font-medium transition-colors"
          >
            {active === i && (
              <motion.span
                layoutId="track-pill"
                className={`absolute inset-0 rounded-full ${
                  t.accent === "signal" ? "bg-signal/15" : "bg-teal/15"
                } ring-1 ${t.accent === "signal" ? "ring-signal/40" : "ring-teal/40"}`}
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
              />
            )}
            <span
              className={`relative z-10 ${
                active === i ? (t.accent === "signal" ? "text-signal" : "text-teal") : "text-muted"
              }`}
            >
              {t.title}
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={track.key}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="grid gap-12 lg:grid-cols-[0.85fr_1.15fr]"
        >
          <div>
            <span className={`tick-label ${accent}`}>{track.kicker}</span>
            <h3 className="display mt-3 text-3xl font-light text-paper md:text-4xl">
              {track.title}
            </h3>
            <p className="mt-5 leading-relaxed text-paper-dim">{track.blurb}</p>
          </div>

          <ol className="relative space-y-7 border-l border-line pl-8">
            {track.steps.map((s, i) => (
              <motion.li
                key={s.label}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.08 * i + 0.1, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                className="relative"
              >
                <span
                  className={`absolute -left-[2.35rem] top-1 flex h-5 w-5 items-center justify-center rounded-full ${accentBg} mono text-[0.6rem] font-bold text-ink`}
                >
                  {i + 1}
                </span>
                <h4 className="text-[0.98rem] font-semibold text-paper">{s.label}</h4>
                <p className="mt-1 text-sm leading-relaxed text-muted">{s.detail}</p>
                {s.code && <CodeBlock code={s.code} className="mt-3" />}
              </motion.li>
            ))}
          </ol>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
