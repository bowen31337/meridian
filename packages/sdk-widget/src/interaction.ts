import React from "react";
import type { CanvasInteraction } from "./types.js";

/** Callback type for submitting a canvas interaction to the harness. */
export type OnInteraction = (interaction: CanvasInteraction) => Promise<void>;

/**
 * React context carrying the host-provided OnInteraction handler.
 * The host (LiveCanvasPanel) wraps canvas widgets with this context so they
 * can submit form-submit and button-click interactions back to the harness.
 */
export const InteractionContext = React.createContext<OnInteraction | null>(null);

/** Returns the OnInteraction callback from context, or null when not provided. */
export function useInteraction(): OnInteraction | null {
  return React.useContext(InteractionContext);
}
