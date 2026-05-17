import type { WidgetComponent } from "../contract.js";
import { CodeWidget } from "./code.js";
import { FormWidget } from "./form.js";
import { ImageWidget } from "./image.js";
import { MarkdownWidget } from "./markdown.js";
import { ProgressWidget } from "./progress.js";
import { TableWidget } from "./table.js";
import { TextWidget } from "./text.js";

/** All built-in Meridian widget components.
 *
 * Register them at app startup:
 * ```ts
 * import { defaultRegistry } from "@meridian/sdk-widget";
 * import { ALL_WIDGETS } from "@meridian/sdk-widget/widgets";
 *
 * for (const widget of ALL_WIDGETS) {
 *   defaultRegistry.register(widget);
 * }
 * ```
 */
export const ALL_WIDGETS: readonly WidgetComponent[] = [
  CodeWidget,
  FormWidget,
  ImageWidget,
  MarkdownWidget,
  ProgressWidget,
  TableWidget,
  TextWidget,
];
