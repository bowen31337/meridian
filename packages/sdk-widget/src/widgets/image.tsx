import type React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface ImageProps {
  /** Image URL or data URI. */
  readonly src: string;
  /** Alternative text for accessibility. */
  readonly alt?: string;
  /** Optional caption rendered below the image. */
  readonly caption?: string;
}

const MANIFEST = {
  kind: "meridian.image",
  version: "1.0.0",
  displayName: "Image",
  description: "Embedded image with optional alt text and caption.",
  propsSchema: {
    type: "object",
    required: ["src"],
    properties: {
      src: { type: "string" },
      alt: { type: "string" },
      caption: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function ImageWidgetImpl({ props }: WidgetProps<ImageProps>): React.ReactElement {
  const src = props.src as string;
  const alt = (props.alt as string | undefined) ?? "";
  const caption = props.caption as string | undefined;

  return (
    <figure
      data-widget-kind="meridian.image"
      style={{ margin: 0, display: "inline-block", maxWidth: "100%" }}
    >
      <img src={src} alt={alt} style={{ display: "block", maxWidth: "100%", height: "auto" }} />
      {caption != null && (
        <figcaption
          data-testid="image-caption"
          style={{ fontSize: "0.875rem", marginTop: 4, opacity: 0.75 }}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

export const ImageWidget = defineWidget<ImageProps>(ImageWidgetImpl, MANIFEST);
