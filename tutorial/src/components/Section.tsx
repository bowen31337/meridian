import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

export function Section({
  id,
  index,
  kicker,
  title,
  children,
}: {
  id: string;
  index: string;
  kicker: string;
  title: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="relative mx-auto w-full max-w-5xl scroll-mt-24 px-6 py-24 md:py-32">
      <Reveal>
        <div className="mb-3 flex items-center gap-4">
          <span className="mono text-[0.7rem] tracking-[0.3em] text-signal">{index}</span>
          <span className="h-px flex-1 bg-line" />
          <span className="tick-label">{kicker}</span>
        </div>
        <h2 className="display max-w-3xl text-balance text-4xl font-light text-paper md:text-6xl">
          {title}
        </h2>
      </Reveal>
      <div className="mt-12">{children}</div>
    </section>
  );
}
