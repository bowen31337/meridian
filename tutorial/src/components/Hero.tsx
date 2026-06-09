import gsap from "gsap";
import { useEffect, useRef } from "react";
import { type TermLine, Terminal } from "./Terminal";

const HERO_LINES: TermLine[] = [
  { kind: "cmd", text: "python -m meridiand" },
  { kind: "dim", text: "meridiand 0.4 — local-first agent control plane" },
  { kind: "ok", text: "✓ listening on 127.0.0.1:7432  · /v1  · /v1/x" },
  { kind: "cmd", text: 'meridian sessions create --data \'{"agent_id":"scout"}\'' },
  { kind: "out", text: "sess_01H9 created · phase: idle → running" },
  { kind: "cmd", text: "meridian meridianrun sess_01H9" },
  { kind: "dim", text: "Assistant: On it — one session, every front door." },
  { kind: "ok", text: "✓ phase: running → done" },
];

export function Hero({ onJump }: { onJump: (id: string) => void }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
      tl.from("[data-hero='kick']", { y: 16, opacity: 0, duration: 0.8 })
        .from(
          "[data-hero='word']",
          { yPercent: 115, opacity: 0, duration: 1.05, stagger: 0.09 },
          "-=0.4",
        )
        .from("[data-hero='sub']", { y: 20, opacity: 0, duration: 0.9 }, "-=0.6")
        .from("[data-hero='cta'] > *", { y: 14, opacity: 0, duration: 0.7, stagger: 0.1 }, "-=0.5")
        .from("[data-hero='term']", { y: 40, opacity: 0, duration: 1.1 }, "-=0.8")
        .from("[data-hero='meta'] > *", { opacity: 0, duration: 0.8, stagger: 0.12 }, "-=0.6");
    }, root);
    return () => ctx.revert();
  }, []);

  return (
    <header
      ref={root}
      className="relative mx-auto flex min-h-screen w-full max-w-6xl flex-col justify-center px-6 pb-16 pt-28"
    >
      <div className="grid items-center gap-14 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <div data-hero="kick" className="mb-7 flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-signal shadow-[0_0_10px_2px_rgba(246,166,35,0.6)]" />
            <span className="tick-label">A field guide to Meridian</span>
          </div>

          <h1 className="display text-paper">
            <span className="block overflow-hidden">
              <span data-hero="word" className="block text-[clamp(3rem,9vw,7rem)] font-light">
                One daemon.
              </span>
            </span>
            <span className="block overflow-hidden">
              <span
                data-hero="word"
                className="block text-[clamp(3rem,9vw,7rem)] font-light italic text-signal"
              >
                Two front doors.
              </span>
            </span>
          </h1>

          <p
            data-hero="sub"
            className="mt-8 max-w-xl text-lg leading-relaxed text-paper-dim md:text-xl"
          >
            Meridian is a self-hostable, local-first runtime for LLM agents. The same process is
            your <span className="text-paper">coding agent</span> at the keyboard and your{" "}
            <span className="text-paper">personal-assistant gateway</span> everywhere else. One
            session is the truth; everything else is a viewer.
          </p>

          <div data-hero="cta" className="mt-10 flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={() => onJump("quickstart")}
              className="group relative overflow-hidden rounded-full bg-signal px-7 py-3 text-sm font-medium text-ink transition-transform hover:scale-[1.03]"
            >
              Start in 4 commands
            </button>
            <button
              type="button"
              onClick={() => onJump("model")}
              className="signal-link text-sm font-medium"
            >
              Read the mental model →
            </button>
          </div>

          <dl data-hero="meta" className="mono mt-12 flex flex-wrap gap-x-10 gap-y-3 text-xs">
            {[
              ["local-first", "no cloud, your laptop"],
              ["API-compatible", "Anthropic managed-agents shape"],
              ["model-agnostic", "Anthropic · OpenAI · Ollama"],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="text-signal">{k}</dt>
                <dd className="mt-1 text-muted">{v}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div data-hero="term">
          <Terminal lines={HERO_LINES} title="~/meridian" />
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 hidden -translate-x-1/2 md:block">
        <span className="tick-label animate-pulse">scroll ↓</span>
      </div>
    </header>
  );
}
