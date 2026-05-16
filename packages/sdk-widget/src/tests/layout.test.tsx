import { render } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { Col, Divider, Grid, Row, Spacer, Stack } from "../layout.js";

describe("Row", () => {
  it("renders a div with flex-direction: row", () => {
    const { container } = render(<Row>child</Row>);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.display).toBe("flex");
    expect(div.style.flexDirection).toBe("row");
  });

  it("forwards gap and align props", () => {
    const { container } = render(<Row gap={8} align="center" />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.gap).toBe("8px");
    expect(div.style.alignItems).toBe("center");
  });

  it("sets flex-wrap when wrap=true", () => {
    const { container } = render(<Row wrap>x</Row>);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.flexWrap).toBe("wrap");
  });
});

describe("Col", () => {
  it("renders a div with flex-direction: column", () => {
    const { container } = render(<Col>child</Col>);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.flexDirection).toBe("column");
  });
});

describe("Stack", () => {
  it("defaults to 8px gap", () => {
    const { container } = render(<Stack>x</Stack>);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.gap).toBe("8px");
    expect(div.style.flexDirection).toBe("column");
  });

  it("accepts a custom gap", () => {
    const { container } = render(<Stack gap={16}>x</Stack>);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.gap).toBe("16px");
  });
});

describe("Grid", () => {
  it("renders a CSS grid", () => {
    const { container } = render(<Grid columns={3} gap={16} />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.display).toBe("grid");
    expect(div.style.gridTemplateColumns).toBe("repeat(3, 1fr)");
    expect(div.style.gap).toBe("16px");
  });

  it("accepts a custom column template string", () => {
    const { container } = render(<Grid columns="200px 1fr" />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.gridTemplateColumns).toBe("200px 1fr");
  });
});

describe("Spacer", () => {
  it("renders with flex:1 when no size is given", () => {
    const { container } = render(<Spacer />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.flexGrow).toBe("1");
  });

  it("renders a fixed horizontal size", () => {
    const { container } = render(<Spacer size={24} axis="horizontal" />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.width).toBe("24px");
    expect(div.style.height).toBe("");
  });

  it("renders a fixed vertical size", () => {
    const { container } = render(<Spacer size={12} axis="vertical" />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.style.height).toBe("12px");
    expect(div.style.width).toBe("");
  });

  it("is hidden from accessibility tree", () => {
    const { container } = render(<Spacer />);
    const div = container.firstChild as HTMLDivElement;
    expect(div.getAttribute("aria-hidden")).toBe("true");
  });
});

describe("Divider", () => {
  it("renders an hr with role separator", () => {
    const { container } = render(<Divider />);
    const hr = container.firstChild as HTMLHRElement;
    expect(hr.tagName).toBe("HR");
    expect(hr.getAttribute("role")).toBe("separator");
  });

  it("defaults to horizontal orientation with 1px height", () => {
    const { container } = render(<Divider />);
    const hr = container.firstChild as HTMLHRElement;
    expect(hr.style.width).toBe("100%");
    expect(hr.style.height).toBe("1px");
  });

  it("renders vertically", () => {
    const { container } = render(<Divider orientation="vertical" />);
    const hr = container.firstChild as HTMLHRElement;
    expect(hr.style.width).toBe("1px");
    expect(hr.style.height).toBe("100%");
  });

  it("accepts custom thickness", () => {
    const { container } = render(<Divider thickness={2} />);
    const hr = container.firstChild as HTMLHRElement;
    expect(hr.style.height).toBe("2px");
  });
});
