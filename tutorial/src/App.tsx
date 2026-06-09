import Lenis from "lenis";
import { useEffect, useRef, useState } from "react";
import { CodeBlock } from "./components/CodeBlock";
import { CommandIndex } from "./components/CommandIndex";
import { Hero } from "./components/Hero";
import { Rail } from "./components/Rail";
import { Reveal } from "./components/Reveal";
import { Section } from "./components/Section";
import { Tracks } from "./components/Tracks";
import { CONCEPTS, NAV, PRINCIPLES, QUICKSTART } from "./content";

export default function App() {
  const lenisRef = useRef<Lenis | null>(null);
  const [active, setActive] = useState("intro");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const lenis = new Lenis({ lerp: 0.1, wheelMultiplier: 1 });
    lenisRef.current = lenis;
    let raf = 0;
    const loop = (t: number) => {
      lenis.raf(t);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    const onScroll = () => {
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(1, window.scrollY / max) : 0);

      const mid = window.scrollY + window.innerHeight * 0.38;
      let current = NAV[0].id;
      for (const item of NAV) {
        const el = document.getElementById(item.id);
        if (el && el.offsetTop <= mid) current = item.id;
      }
      setActive(current);
    };
    lenis.on("scroll", onScroll);
    onScroll();

    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
    };
  }, []);

  const jump = (id: string) => {
    const el = document.getElementById(id);
    if (el && lenisRef.current) lenisRef.current.scrollTo(el, { offset: -40 });
  };

  return (
    <div className="lg:pl-[clamp(140px,15vw,210px)]">
      <Rail active={active} progress={progress} onJump={jump} />

      <Hero onJump={jump} />

      {/* 01 — mental model */}
      <Section
        id="model"
        index="01"
        kicker="how to think about it"
        title={
          <>
            Four ideas hold the
            <br />
            whole system together.
          </>
        }
      >
        <div className="grid gap-5 sm:grid-cols-2">
          {PRINCIPLES.map((p, i) => (
            <Reveal key={p.k} delay={i * 0.08}>
              <article className="card-glow glass h-full rounded-2xl p-7 transition-all duration-500">
                <div className="mb-5 flex items-center justify-between">
                  <span className="display text-3xl italic text-signal">{p.k}</span>
                  <span className="mono rounded-full border border-line px-2.5 py-1 text-[0.6rem] uppercase tracking-widest text-muted">
                    {p.tag}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-paper">{p.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted">{p.body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* 02 — quickstart */}
      <Section
        id="quickstart"
        index="02"
        kicker="zero to running"
        title={
          <>
            From clone to first
            <br />
            reply, four steps.
          </>
        }
      >
        <div className="space-y-px">
          {QUICKSTART.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.06}>
              <div className="grid gap-6 border-t border-line py-8 md:grid-cols-[auto_0.9fr_1.1fr] md:gap-10">
                <span className="display text-4xl font-light text-faint md:text-5xl">{s.n}</span>
                <div>
                  <h3 className="text-lg font-semibold text-paper">{s.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{s.note}</p>
                </div>
                <CodeBlock code={s.code} />
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* 03 — tracks */}
      <Section
        id="tracks"
        index="03"
        kicker="one binary, two surfaces"
        title={
          <>
            Same daemon. Pick the
            <br />
            door that fits the moment.
          </>
        }
      >
        <Tracks />
      </Section>

      {/* 04 — concepts */}
      <Section
        id="concepts"
        index="04"
        kicker="what makes it SOTA"
        title={
          <>
            The machinery behind
            <br />
            the two front doors.
          </>
        }
      >
        <div className="grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
          {CONCEPTS.map((c, i) => (
            <Reveal key={c.title} delay={(i % 3) * 0.07}>
              <article className="group h-full bg-ink p-7 transition-colors hover:bg-ink-2">
                <div className="mono mb-4 text-[0.62rem] uppercase tracking-[0.25em] text-teal">
                  {c.ref}
                </div>
                <h3 className="text-base font-semibold text-paper">{c.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted">{c.body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* 05 — command index */}
      <Section id="reference" index="05" kicker="keep this open" title={<>The command index.</>}>
        <CommandIndex />
      </Section>

      <Footer onJump={jump} />
    </div>
  );
}

function Footer({ onJump }: { onJump: (id: string) => void }) {
  return (
    <footer className="relative mx-auto w-full max-w-5xl px-6 pb-20 pt-10">
      <div className="hairline mb-12 h-px" />
      <div className="flex flex-col gap-10 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="display text-3xl font-light text-paper">
            Meridian<span className="text-signal">.</span>
          </div>
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted">
            One persistent, versioned, capability-scoped control plane — reachable through whichever
            surface fits the moment.
          </p>
        </div>
        <nav className="flex flex-wrap gap-x-8 gap-y-2">
          {NAV.slice(1).map((n) => (
            <button
              key={n.id}
              type="button"
              onClick={() => onJump(n.id)}
              className="signal-link text-sm text-paper-dim"
            >
              {n.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="mono mt-12 flex flex-col gap-2 text-[0.7rem] text-faint md:flex-row md:justify-between">
        <span>docs/ · PRD.md · ARCHITECTURE.md · TOOL_AUTHOR_GUIDE.md</span>
        <span>self-hosted · local-first · the session is the truth</span>
      </div>
    </footer>
  );
}
