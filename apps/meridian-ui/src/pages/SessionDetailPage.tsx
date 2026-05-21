import React from "react";
import { useParams } from "react-router-dom";
import { LiveCanvasPanel } from "../canvas/index.js";

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  if (!id) return <p>Session not found.</p>;
  return <LiveCanvasPanel sessionId={id} />;
}
