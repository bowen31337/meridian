import type { CanvasOp } from "@meridian/sdk-widget";
import { describe, expect, it } from "vitest";
import { applyCanvasOp, toContentBlock } from "./canvas-state.js";

function makeOp(overrides: Partial<CanvasOp> = {}): CanvasOp {
  return {
    op: "set",
    widget_id: "w1",
    widget_kind: "meridian.text",
    props: { text: "hello" },
    sequence: 1,
    session_id: "sess1",
    timestamp: "2026-05-21T00:00:00Z",
    ...overrides,
  };
}

describe("applyCanvasOp — set", () => {
  it("creates a new widget entry", () => {
    const state = applyCanvasOp(new Map(), makeOp({ op: "set" }));
    expect(state.size).toBe(1);
    expect(state.get("w1")?.props).toEqual({ text: "hello" });
  });

  it("overwrites existing props", () => {
    let state = applyCanvasOp(new Map(), makeOp({ op: "set", props: { text: "first" } }));
    state = applyCanvasOp(state, makeOp({ op: "set", props: { text: "second" }, sequence: 2 }));
    expect(state.get("w1")?.props).toEqual({ text: "second" });
    expect(state.size).toBe(1);
  });

  it("stores session_id, widget_kind, sequence, and timestamp", () => {
    const state = applyCanvasOp(new Map(), makeOp());
    const entry = state.get("w1");
    expect(entry?.session_id).toBe("sess1");
    expect(entry?.widget_kind).toBe("meridian.text");
    expect(entry?.sequence).toBe(1);
    expect(entry?.timestamp).toBe("2026-05-21T00:00:00Z");
  });
});

describe("applyCanvasOp — patch", () => {
  it("merges props into the existing entry", () => {
    let state = applyCanvasOp(
      new Map(),
      makeOp({ op: "set", props: { text: "hi", extra: "keep" } }),
    );
    state = applyCanvasOp(state, makeOp({ op: "patch", props: { text: "updated" }, sequence: 2 }));
    expect(state.get("w1")?.props).toEqual({ text: "updated", extra: "keep" });
  });

  it("creates a new entry when widget does not exist yet", () => {
    const state = applyCanvasOp(new Map(), makeOp({ op: "patch", props: { text: "new" } }));
    expect(state.get("w1")?.props).toEqual({ text: "new" });
  });

  it("updates sequence and timestamp", () => {
    let state = applyCanvasOp(new Map(), makeOp({ op: "set", sequence: 1, timestamp: "t1" }));
    state = applyCanvasOp(state, makeOp({ op: "patch", sequence: 5, timestamp: "t5" }));
    expect(state.get("w1")?.sequence).toBe(5);
    expect(state.get("w1")?.timestamp).toBe("t5");
  });
});

describe("applyCanvasOp — append", () => {
  it("concatenates array props", () => {
    let state = applyCanvasOp(
      new Map(),
      makeOp({ widget_kind: "meridian.table", op: "set", props: { columns: ["A"], rows: [[1]] } }),
    );
    state = applyCanvasOp(
      state,
      makeOp({
        widget_kind: "meridian.table",
        op: "append",
        props: { rows: [[2]] },
        sequence: 2,
      }),
    );
    const entry = state.get("w1");
    expect((entry?.props as { rows: unknown[] }).rows).toEqual([[1], [2]]);
    expect((entry?.props as { columns: unknown[] }).columns).toEqual(["A"]);
  });

  it("replaces non-array props", () => {
    let state = applyCanvasOp(new Map(), makeOp({ op: "set", props: { text: "old" } }));
    state = applyCanvasOp(state, makeOp({ op: "append", props: { text: "new" }, sequence: 2 }));
    expect(state.get("w1")?.props).toEqual({ text: "new" });
  });

  it("creates a new entry when widget does not exist yet", () => {
    const state = applyCanvasOp(new Map(), makeOp({ op: "append", props: { text: "first" } }));
    expect(state.get("w1")?.props).toEqual({ text: "first" });
  });
});

describe("applyCanvasOp — clear", () => {
  it("removes the widget from state", () => {
    let state = applyCanvasOp(new Map(), makeOp({ op: "set" }));
    state = applyCanvasOp(state, makeOp({ op: "clear", sequence: 2 }));
    expect(state.size).toBe(0);
    expect(state.has("w1")).toBe(false);
  });

  it("is a no-op when widget does not exist", () => {
    const empty = new Map();
    const state = applyCanvasOp(empty, makeOp({ op: "clear" }));
    expect(state.size).toBe(0);
  });
});

describe("toContentBlock", () => {
  it("returns a ContentBlockCanvasOp with op=set", () => {
    const state = applyCanvasOp(new Map(), makeOp({ op: "set" }));
    const entry = state.get("w1");
    if (!entry) throw new Error("expected entry for w1");
    const block = toContentBlock(entry);
    expect(block.type).toBe("canvas_op");
    expect(block.canvas_op.op).toBe("set");
    expect(block.canvas_op.widget_id).toBe("w1");
    expect(block.canvas_op.widget_kind).toBe("meridian.text");
    expect(block.canvas_op.props).toEqual({ text: "hello" });
    expect(block.canvas_op.session_id).toBe("sess1");
  });

  it("preserves sequence and timestamp from the entry", () => {
    const state = applyCanvasOp(
      new Map(),
      makeOp({ sequence: 7, timestamp: "2026-01-01T00:00:00Z" }),
    );
    const entry = state.get("w1");
    if (!entry) throw new Error("expected entry for w1");
    const block = toContentBlock(entry);
    expect(block.canvas_op.sequence).toBe(7);
    expect(block.canvas_op.timestamp).toBe("2026-01-01T00:00:00Z");
  });
});
