import { useEffect, useRef, useState } from "react";

export type TermLine = { kind: "cmd" | "out" | "ok" | "warn" | "dim"; text: string };

const COLOR: Record<TermLine["kind"], string> = {
  cmd: "text-paper",
  out: "text-paper-dim",
  ok: "text-teal-soft",
  warn: "text-signal-soft",
  dim: "text-muted",
};

/** A terminal panel that types its lines out once it scrolls into view. */
export function Terminal({
  lines,
  title = "meridian",
  className = "",
  autoplay = true,
}: {
  lines: TermLine[];
  title?: string;
  className?: string;
  autoplay?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [started, setStarted] = useState(!autoplay);
  const [visibleCount, setVisibleCount] = useState(autoplay ? 0 : lines.length);
  const [typed, setTyped] = useState("");

  useEffect(() => {
    if (!autoplay || started) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setStarted(true);
            io.disconnect();
          }
        }
      },
      { threshold: 0.4 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [autoplay, started]);

  useEffect(() => {
    if (!started || visibleCount >= lines.length) return;
    const line = lines[visibleCount];
    // Output lines drop in instantly; command lines type character by character.
    if (line.kind !== "cmd") {
      const t = setTimeout(() => setVisibleCount((c) => c + 1), 120);
      return () => clearTimeout(t);
    }
    if (typed.length < line.text.length) {
      const t = setTimeout(() => setTyped(line.text.slice(0, typed.length + 1)), 22);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setVisibleCount((c) => c + 1);
      setTyped("");
    }, 280);
    return () => clearTimeout(t);
  }, [started, visibleCount, typed, lines]);

  const done = visibleCount >= lines.length;

  return (
    <div
      ref={ref}
      className={`glass overflow-hidden rounded-xl shadow-[0_30px_80px_-40px_rgba(0,0,0,0.9)] ${className}`}
    >
      <div className="flex items-center gap-2 border-b border-line bg-ink-2/60 px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-rose/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-signal/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-teal/80" />
        <span className="mono ml-3 text-[0.7rem] tracking-widest text-muted">{title}</span>
      </div>
      <div className="mono max-h-[420px] overflow-auto p-5 text-[0.82rem] leading-relaxed scrollbar-none">
        {lines.slice(0, visibleCount).map((l, i) => (
          <Line key={i} line={l} />
        ))}
        {!done && started && (
          <div className="flex">
            {lines[visibleCount]?.kind === "cmd" && <span className="mr-2 text-signal">›</span>}
            <span className={COLOR[lines[visibleCount]?.kind ?? "out"]}>
              {lines[visibleCount]?.kind === "cmd" ? typed : ""}
            </span>
            <span className="caret" />
          </div>
        )}
      </div>
    </div>
  );
}

function Line({ line }: { line: TermLine }) {
  return (
    <div className="flex whitespace-pre-wrap">
      {line.kind === "cmd" && <span className="mr-2 shrink-0 text-signal">›</span>}
      <span className={COLOR[line.kind]}>{line.text}</span>
    </div>
  );
}
