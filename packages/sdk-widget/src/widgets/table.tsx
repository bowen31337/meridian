import React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface TableProps {
  /** Column header labels. */
  readonly columns: readonly string[];
  /** Data rows; each row is an array of cell values aligned to columns. */
  readonly rows: ReadonlyArray<readonly unknown[]>;
  /** Optional caption rendered above the table. */
  readonly caption?: string;
}

const MANIFEST = {
  kind: "meridian.table",
  version: "1.0.0",
  displayName: "Table",
  description: "Row/column data table with optional caption.",
  propsSchema: {
    type: "object",
    required: ["columns", "rows"],
    properties: {
      columns: {
        type: "array",
        items: { type: "string" },
        minItems: 1,
      },
      rows: {
        type: "array",
        items: { type: "array" },
      },
      caption: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function TableWidgetImpl({ props }: WidgetProps<TableProps>): React.ReactElement {
  const columns = props.columns as string[];
  const rows = props.rows as unknown[][];
  const caption = props.caption as string | undefined;

  return (
    <table
      data-widget-kind="meridian.table"
      style={{ borderCollapse: "collapse", width: "100%" }}
    >
      {caption != null && <caption style={{ textAlign: "left", marginBottom: 4 }}>{caption}</caption>}
      <thead>
        <tr>
          {columns.map((col) => (
            <th
              key={col}
              scope="col"
              style={{ textAlign: "left", padding: "6px 8px", borderBottom: "2px solid currentColor" }}
            >
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIdx) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: row order is stable within a canvas op
          <tr key={rowIdx}>
            {columns.map((col, colIdx) => (
              <td
                key={col}
                style={{ padding: "6px 8px", borderBottom: "1px solid currentColor" }}
              >
                {row[colIdx] !== undefined && row[colIdx] !== null ? String(row[colIdx]) : ""}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export const TableWidget = defineWidget<TableProps>(TableWidgetImpl, MANIFEST);
