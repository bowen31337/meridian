import type { CanvasOp, ContentBlockCanvasOp } from "@meridian/sdk-widget";

export interface CanvasWidgetEntry {
  readonly widget_id: string;
  readonly widget_kind: string;
  readonly props: Record<string, unknown>;
  readonly sequence: number;
  readonly session_id: string;
  readonly timestamp: string;
}

export type CanvasWidgetMap = ReadonlyMap<string, CanvasWidgetEntry>;

export function applyCanvasOp(state: CanvasWidgetMap, op: CanvasOp): CanvasWidgetMap {
  const next = new Map(state);
  switch (op.op) {
    case "set": {
      next.set(op.widget_id, {
        widget_id: op.widget_id,
        widget_kind: op.widget_kind,
        props: { ...op.props },
        sequence: op.sequence,
        session_id: op.session_id,
        timestamp: op.timestamp,
      });
      break;
    }
    case "patch": {
      const existing = next.get(op.widget_id);
      next.set(op.widget_id, {
        widget_id: op.widget_id,
        widget_kind: op.widget_kind,
        props: { ...(existing?.props ?? {}), ...op.props },
        sequence: op.sequence,
        session_id: op.session_id,
        timestamp: op.timestamp,
      });
      break;
    }
    case "append": {
      const existing = next.get(op.widget_id);
      const base = existing?.props ?? {};
      const merged: Record<string, unknown> = { ...base };
      for (const [key, val] of Object.entries(op.props)) {
        const cur = merged[key];
        if (Array.isArray(cur) && Array.isArray(val)) {
          merged[key] = [...cur, ...(val as unknown[])];
        } else {
          merged[key] = val;
        }
      }
      next.set(op.widget_id, {
        widget_id: op.widget_id,
        widget_kind: op.widget_kind,
        props: merged,
        sequence: op.sequence,
        session_id: op.session_id,
        timestamp: op.timestamp,
      });
      break;
    }
    case "clear": {
      next.delete(op.widget_id);
      break;
    }
  }
  return next;
}

export function toContentBlock(entry: CanvasWidgetEntry): ContentBlockCanvasOp {
  return {
    type: "canvas_op",
    canvas_op: {
      op: "set",
      widget_id: entry.widget_id,
      widget_kind: entry.widget_kind,
      props: entry.props,
      sequence: entry.sequence,
      session_id: entry.session_id,
      timestamp: entry.timestamp,
    },
  };
}
