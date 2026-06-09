import { NAV } from "../content";

export function Rail({
  active,
  progress,
  onJump,
}: {
  active: string;
  progress: number;
  onJump: (id: string) => void;
}) {
  return (
    <nav
      aria-label="Sections"
      className="pointer-events-none fixed left-0 top-0 z-40 hidden h-screen w-[clamp(140px,15vw,210px)] flex-col justify-center pl-7 lg:flex"
    >
      {/* the meridian line */}
      <div className="absolute left-7 top-0 h-full w-px bg-line" />
      <div
        className="absolute left-7 top-0 w-px bg-gradient-to-b from-signal via-signal to-teal"
        style={{ height: `${progress * 100}%` }}
      />
      <div
        className="absolute left-7 h-2 w-2 -translate-x-1/2 rounded-full bg-signal shadow-[0_0_12px_2px_rgba(246,166,35,0.6)]"
        style={{ top: `calc(${progress * 100}% - 4px)` }}
      />

      <ul className="pointer-events-auto flex flex-col gap-7 pl-6">
        {NAV.map((item) => {
          const on = item.id === active;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onJump(item.id)}
                className="group flex items-center gap-3 text-left"
              >
                <span
                  className={`mono text-[0.62rem] tabular-nums transition-colors ${
                    on ? "text-signal" : "text-faint group-hover:text-muted"
                  }`}
                >
                  {item.index}
                </span>
                <span
                  className={`text-[0.82rem] transition-all duration-500 ${
                    on
                      ? "translate-x-0 text-paper"
                      : "-translate-x-1 text-muted group-hover:translate-x-0 group-hover:text-paper-dim"
                  }`}
                >
                  {item.label}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
