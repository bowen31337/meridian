/**
 * Tests for the seven built-in Meridian widget components.
 *
 * Each widget is rendered through the WidgetRegistry pipeline so that
 * the OTel span, audit log, and onError paths are exercised alongside
 * the component rendering itself.
 */

import { cleanup, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AuditLog } from "../audit.js";
import { WidgetRegistry } from "../pipeline.js";
import type { ContentBlockCanvasOp } from "../types.js";
import { ALL_WIDGETS } from "../widgets/registry.js";
import { CodeWidget } from "../widgets/code.js";
import { FormWidget } from "../widgets/form.js";
import { ImageWidget } from "../widgets/image.js";
import { MarkdownWidget } from "../widgets/markdown.js";
import { ProgressWidget } from "../widgets/progress.js";
import { TableWidget } from "../widgets/table.js";
import { TextWidget } from "../widgets/text.js";

// ---------------------------------------------------------------------------
// OTel mock
// ---------------------------------------------------------------------------
const mockSpan = {
  addEvent: vi.fn(),
  setStatus: vi.fn(),
  recordException: vi.fn(),
  end: vi.fn(),
};

vi.mock("@opentelemetry/api", () => ({
  trace: {
    getTracer: () => ({
      startActiveSpan: (_name: string, _opts: unknown, fn: (span: typeof mockSpan) => unknown) =>
        fn(mockSpan),
    }),
  },
  SpanStatusCode: { UNSET: 0, OK: 1, ERROR: 2 },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAuditLog(): AuditLog & { entries: unknown[] } {
  const entries: unknown[] = [];
  return { write: (e) => entries.push(e), entries };
}

function makeBlock(
  widget_kind: string,
  props: Record<string, unknown>,
  op: ContentBlockCanvasOp["canvas_op"]["op"] = "set",
): ContentBlockCanvasOp {
  return {
    type: "canvas_op",
    canvas_op: {
      op,
      widget_id: "test-widget",
      widget_kind,
      props,
      sequence: 1,
      session_id: "sess-test",
      timestamp: "2026-05-17T00:00:00Z",
    },
  };
}

// ---------------------------------------------------------------------------
// ALL_WIDGETS registration
// ---------------------------------------------------------------------------

describe("ALL_WIDGETS", () => {
  it("contains all seven built-in widget kinds", () => {
    const kinds = ALL_WIDGETS.map((w) => w.manifest.kind).sort();
    expect(kinds).toEqual([
      "meridian.code",
      "meridian.form",
      "meridian.image",
      "meridian.markdown",
      "meridian.progress",
      "meridian.table",
      "meridian.text",
    ]);
  });

  it("can be registered without duplicate errors", () => {
    const registry = new WidgetRegistry();
    expect(() => {
      for (const w of ALL_WIDGETS) registry.register(w);
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Shared registry for widget-level tests
// ---------------------------------------------------------------------------

let registry: WidgetRegistry;
let auditLog: ReturnType<typeof makeAuditLog>;

beforeEach(() => {
  registry = new WidgetRegistry();
  for (const w of ALL_WIDGETS) registry.register(w);
  auditLog = makeAuditLog();
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// meridian.text
// ---------------------------------------------------------------------------

describe("TextWidget (meridian.text)", () => {
  it("has the correct manifest kind", () => {
    expect(TextWidget.manifest.kind).toBe("meridian.text");
  });

  it("renders the text prop", () => {
    render(registry.renderCanvasOp(makeBlock("meridian.text", { text: "hello world" }), { auditLog }));
    expect(screen.getByText("hello world")).toBeTruthy();
  });

  it("fails WIDGET_PROPS_INVALID when text is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.text", {}), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
    expect((errors[0] as { code: string }).code).toBe("WIDGET_PROPS_INVALID");
  });

  it("fails WIDGET_PROPS_INVALID when text is not a string", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.text", { text: 42 }), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.markdown
// ---------------------------------------------------------------------------

describe("MarkdownWidget (meridian.markdown)", () => {
  it("has the correct manifest kind", () => {
    expect(MarkdownWidget.manifest.kind).toBe("meridian.markdown");
  });

  it("renders the content prop", () => {
    render(
      registry.renderCanvasOp(makeBlock("meridian.markdown", { content: "# Title" }), { auditLog }),
    );
    expect(screen.getByText("# Title")).toBeTruthy();
  });

  it("sets data-format=markdown on the container", () => {
    const { container } = render(
      registry.renderCanvasOp(makeBlock("meridian.markdown", { content: "foo" }), { auditLog }),
    );
    expect(container.querySelector("[data-format='markdown']")).toBeTruthy();
  });

  it("fails WIDGET_PROPS_INVALID when content is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.markdown", {}), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.form
// ---------------------------------------------------------------------------

describe("FormWidget (meridian.form)", () => {
  it("has the correct manifest kind", () => {
    expect(FormWidget.manifest.kind).toBe("meridian.form");
  });

  it("renders each field label", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.form", {
          fields: [
            { name: "email", label: "Email address", type: "email" },
            { name: "age", label: "Age", type: "number" },
          ],
        }),
        { auditLog },
      ),
    );
    expect(screen.getByLabelText("Email address")).toBeTruthy();
    expect(screen.getByLabelText("Age")).toBeTruthy();
  });

  it("renders an optional title", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.form", {
          title: "Contact Us",
          fields: [{ name: "msg", label: "Message", type: "textarea" }],
        }),
        { auditLog },
      ),
    );
    expect(screen.getByText("Contact Us")).toBeTruthy();
  });

  it("renders a checkbox field", () => {
    const { container } = render(
      registry.renderCanvasOp(
        makeBlock("meridian.form", {
          fields: [{ name: "agree", label: "Agree?", type: "checkbox", value: true }],
        }),
        { auditLog },
      ),
    );
    const checkbox = container.querySelector("input[type='checkbox']") as HTMLInputElement;
    expect(checkbox).toBeTruthy();
    expect(checkbox.defaultChecked).toBe(true);
  });

  it("fails WIDGET_PROPS_INVALID when fields is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.form", {}), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.code
// ---------------------------------------------------------------------------

describe("CodeWidget (meridian.code)", () => {
  it("has the correct manifest kind", () => {
    expect(CodeWidget.manifest.kind).toBe("meridian.code");
  });

  it("renders the code prop inside <code>", () => {
    const { container } = render(
      registry.renderCanvasOp(
        makeBlock("meridian.code", { code: "console.log('hi')", language: "javascript" }),
        { auditLog },
      ),
    );
    const code = container.querySelector("code");
    expect(code?.textContent).toBe("console.log('hi')");
  });

  it("sets data-language attribute on <pre>", () => {
    const { container } = render(
      registry.renderCanvasOp(
        makeBlock("meridian.code", { code: "x = 1", language: "python" }),
        { auditLog },
      ),
    );
    const pre = container.querySelector("pre");
    expect(pre?.getAttribute("data-language")).toBe("python");
  });

  it("renders filename when provided", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.code", { code: "x = 1", filename: "main.py" }),
        { auditLog },
      ),
    );
    expect(screen.getByTestId("code-filename").textContent).toBe("main.py");
  });

  it("omits filename element when not provided", () => {
    render(
      registry.renderCanvasOp(makeBlock("meridian.code", { code: "x = 1" }), { auditLog }),
    );
    expect(screen.queryByTestId("code-filename")).toBeNull();
  });

  it("fails WIDGET_PROPS_INVALID when code is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.code", {}), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.image
// ---------------------------------------------------------------------------

describe("ImageWidget (meridian.image)", () => {
  it("has the correct manifest kind", () => {
    expect(ImageWidget.manifest.kind).toBe("meridian.image");
  });

  it("renders an <img> with the src", () => {
    const { container } = render(
      registry.renderCanvasOp(
        makeBlock("meridian.image", { src: "https://example.com/img.png", alt: "test image" }),
        { auditLog },
      ),
    );
    const img = container.querySelector("img") as HTMLImageElement;
    expect(img.src).toContain("example.com");
    expect(img.alt).toBe("test image");
  });

  it("renders caption when provided", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.image", {
          src: "https://example.com/x.png",
          caption: "A nice picture",
        }),
        { auditLog },
      ),
    );
    expect(screen.getByTestId("image-caption").textContent).toBe("A nice picture");
  });

  it("omits figcaption when caption is not provided", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.image", { src: "https://example.com/x.png" }),
        { auditLog },
      ),
    );
    expect(screen.queryByTestId("image-caption")).toBeNull();
  });

  it("fails WIDGET_PROPS_INVALID when src is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.image", { alt: "no src" }), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.table
// ---------------------------------------------------------------------------

describe("TableWidget (meridian.table)", () => {
  it("has the correct manifest kind", () => {
    expect(TableWidget.manifest.kind).toBe("meridian.table");
  });

  it("renders column headers", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.table", {
          columns: ["Name", "Score"],
          rows: [["Alice", 95], ["Bob", 87]],
        }),
        { auditLog },
      ),
    );
    expect(screen.getByText("Name")).toBeTruthy();
    expect(screen.getByText("Score")).toBeTruthy();
  });

  it("renders data rows", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.table", {
          columns: ["Name", "Score"],
          rows: [["Alice", 95], ["Bob", 87]],
        }),
        { auditLog },
      ),
    );
    expect(screen.getByText("Alice")).toBeTruthy();
    expect(screen.getByText("87")).toBeTruthy();
  });

  it("renders a caption when provided", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.table", {
          columns: ["X"],
          rows: [],
          caption: "Test results",
        }),
        { auditLog },
      ),
    );
    expect(screen.getByText("Test results")).toBeTruthy();
  });

  it("fails WIDGET_PROPS_INVALID when columns is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.table", { rows: [] }), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// meridian.progress
// ---------------------------------------------------------------------------

describe("ProgressWidget (meridian.progress)", () => {
  it("has the correct manifest kind", () => {
    expect(ProgressWidget.manifest.kind).toBe("meridian.progress");
  });

  it("renders a <progress> element with value and max", () => {
    const { container } = render(
      registry.renderCanvasOp(
        makeBlock("meridian.progress", { value: 40, max: 80 }),
        { auditLog },
      ),
    );
    const bar = container.querySelector("progress") as HTMLProgressElement;
    expect(bar.value).toBe(40);
    expect(bar.max).toBe(80);
  });

  it("defaults max to 100 when omitted", () => {
    const { container } = render(
      registry.renderCanvasOp(makeBlock("meridian.progress", { value: 50 }), { auditLog }),
    );
    const bar = container.querySelector("progress") as HTMLProgressElement;
    expect(bar.max).toBe(100);
  });

  it("renders a label when provided", () => {
    render(
      registry.renderCanvasOp(
        makeBlock("meridian.progress", { value: 3, max: 10, label: "Step 3 of 10" }),
        { auditLog },
      ),
    );
    expect(screen.getByText("Step 3 of 10")).toBeTruthy();
  });

  it("fails WIDGET_PROPS_INVALID when value is missing", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.progress", { max: 100 }), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });

  it("fails WIDGET_PROPS_INVALID when value is negative", () => {
    const errors: unknown[] = [];
    registry.renderCanvasOp(makeBlock("meridian.progress", { value: -1 }), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    expect(errors).toHaveLength(1);
  });
});
