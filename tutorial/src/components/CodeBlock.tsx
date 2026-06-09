import { useState } from "react";

/** Lightweight, dependency-free shell highlighting. */
function tint(line: string) {
  const trimmed = line.trimStart();
  if (trimmed.startsWith("#")) {
    return <span className="text-faint italic">{line}</span>;
  }
  // split off leading comment token, color the command verb
  const parts = line.split(/(\s+)/);
  let coloredFirst = false;
  return parts.map((p, i) => {
    if (p.trim() === "") return <span key={i}>{p}</span>;
    if (!coloredFirst) {
      coloredFirst = true;
      if (p === "meridian" || p === "python" || p === "uv" || p === "pnpm") {
        return (
          <span key={i} className="text-signal">
            {p}
          </span>
        );
      }
    }
    if (p.startsWith("--") || p.startsWith("-")) {
      return (
        <span key={i} className="text-teal-soft">
          {p}
        </span>
      );
    }
    if (p.startsWith("'") || p.startsWith('"') || p.endsWith("'") || p.endsWith('"')) {
      return (
        <span key={i} className="text-paper-dim">
          {p}
        </span>
      );
    }
    return (
      <span key={i} className="text-paper">
        {p}
      </span>
    );
  });
}

export function CodeBlock({ code, className = "" }: { code: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const lines = code.split("\n");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div
      className={`group relative overflow-hidden rounded-lg border border-line bg-ink-2/70 ${className}`}
    >
      <button
        type="button"
        onClick={copy}
        className="absolute right-2.5 top-2.5 z-10 rounded-md border border-line bg-ink/80 px-2.5 py-1 text-[0.62rem] uppercase tracking-widest text-muted opacity-0 transition-opacity hover:text-signal group-hover:opacity-100"
      >
        {copied ? "copied" : "copy"}
      </button>
      <pre className="mono overflow-x-auto p-4 text-[0.78rem] leading-relaxed scrollbar-none">
        <code>
          {lines.map((l, i) => (
            <div key={i} className="whitespace-pre">
              {tint(l)}
              {l === "" ? " " : ""}
            </div>
          ))}
        </code>
      </pre>
    </div>
  );
}
