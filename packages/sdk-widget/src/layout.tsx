import type React from "react";

// ---------------------------------------------------------------------------
// Shared prop interfaces
// ---------------------------------------------------------------------------

interface FlexContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** CSS gap (number = px). */
  readonly gap?: number | string;
  /** CSS align-items. */
  readonly align?: React.CSSProperties["alignItems"];
  /** CSS justify-content. */
  readonly justify?: React.CSSProperties["justifyContent"];
  /** Enable flex-wrap. */
  readonly wrap?: boolean;
  readonly children?: React.ReactNode;
}

interface GridContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Number of equal-width columns, or a CSS grid-template-columns value. */
  readonly columns?: number | string;
  /** Number of equal-height rows, or a CSS grid-template-rows value. */
  readonly rows?: number | string;
  readonly gap?: number | string;
  readonly columnGap?: number | string;
  readonly rowGap?: number | string;
  readonly children?: React.ReactNode;
}

interface SpacerProps {
  /**
   * Fixed size for the spacer. When omitted the spacer flexes to fill
   * remaining space (`flex: 1`).
   */
  readonly size?: number | string;
  /**
   * Which axis the size applies to.
   * "horizontal" → width only; "vertical" → height only; "both" (default) → both.
   */
  readonly axis?: "horizontal" | "vertical" | "both";
}

interface DividerProps extends React.HTMLAttributes<HTMLHRElement> {
  readonly orientation?: "horizontal" | "vertical";
  readonly thickness?: number | string;
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/**
 * Horizontal flex container.
 *
 * @example
 * <Row gap={8} align="center">
 *   <Icon /> <Label />
 * </Row>
 */
export function Row({
  gap,
  align = "stretch",
  justify = "flex-start",
  wrap = false,
  style,
  children,
  ...rest
}: FlexContainerProps): React.ReactElement {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        alignItems: align,
        justifyContent: justify,
        flexWrap: wrap ? "wrap" : "nowrap",
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
}

/**
 * Vertical flex container.
 *
 * @example
 * <Col gap={16}>
 *   <Header /> <Body />
 * </Col>
 */
export function Col({
  gap,
  align = "stretch",
  justify = "flex-start",
  wrap = false,
  style,
  children,
  ...rest
}: FlexContainerProps): React.ReactElement {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap,
        alignItems: align,
        justifyContent: justify,
        flexWrap: wrap ? "wrap" : "nowrap",
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
}

/**
 * Vertical stack with a default 8 px gap between children.
 * Shorthand for `<Col gap={8}>`.
 */
export function Stack({ gap = 8, ...rest }: FlexContainerProps): React.ReactElement {
  return <Col gap={gap} {...rest} />;
}

/**
 * CSS Grid container.
 *
 * @example
 * <Grid columns={3} gap={16}>
 *   <Card /> <Card /> <Card />
 * </Grid>
 */
export function Grid({
  columns,
  rows,
  gap,
  columnGap,
  rowGap,
  style,
  children,
  ...rest
}: GridContainerProps): React.ReactElement {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: typeof columns === "number" ? `repeat(${columns}, 1fr)` : columns,
        gridTemplateRows: typeof rows === "number" ? `repeat(${rows}, 1fr)` : rows,
        gap,
        columnGap,
        rowGap,
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
}

/**
 * Flexible spacer for use inside Row / Col / Stack.
 * Without a `size` it expands to fill remaining space (`flex: 1`).
 *
 * @example
 * <Row>
 *   <Logo />
 *   <Spacer />       {/* pushes Nav to the right *\/}
 *   <Nav />
 * </Row>
 */
export function Spacer({ size, axis = "both" }: SpacerProps): React.ReactElement {
  const style: React.CSSProperties =
    size === undefined
      ? { flexGrow: 1, flexShrink: 1, minWidth: 0, minHeight: 0 }
      : {
          width: axis === "vertical" ? undefined : size,
          height: axis === "horizontal" ? undefined : size,
          flexShrink: 0,
        };

  return <div aria-hidden="true" style={style} />;
}

/**
 * Visual separator rendered as an `<hr>`.
 * The host provides color via CSS inheritance or the `style` prop.
 *
 * @example
 * <Col>
 *   <Section />
 *   <Divider />
 *   <Section />
 * </Col>
 */
export function Divider({
  orientation = "horizontal",
  thickness = 1,
  style,
  ...rest
}: DividerProps): React.ReactElement {
  const base: React.CSSProperties =
    orientation === "horizontal"
      ? { width: "100%", height: thickness, border: "none", flexShrink: 0 }
      : { width: thickness, height: "100%", border: "none", alignSelf: "stretch", flexShrink: 0 };

  // biome-ignore lint/a11y/noRedundantRoles: explicit role makes it testable via getAttribute
  return <hr role="separator" style={{ ...base, ...style }} {...rest} />;
}
