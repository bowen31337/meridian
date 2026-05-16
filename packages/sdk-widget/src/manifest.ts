import Ajv from "ajv";
import type { ValidateFunction } from "ajv";

/**
 * Static declaration every widget kind must provide.
 * Consumed by the WidgetRegistry for props validation and discovery.
 */
export interface WidgetManifest {
  /** Globally unique kind identifier, e.g. "meridian.text" or "acme.chart". */
  readonly kind: string;
  /** Semver string of this widget implementation. */
  readonly version: string;
  readonly displayName: string;
  readonly description?: string;
  /** JSON Schema (draft-07) that CanvasOp.props must satisfy. */
  readonly propsSchema: Record<string, unknown>;
}

export interface PropsValidationResult {
  readonly valid: boolean;
  readonly errors?: readonly string[];
}

// Module-level Ajv instance; schemas are compiled once per kind@version.
const _ajv = new Ajv({ allErrors: true, strict: false });
const _validatorCache = new Map<string, ValidateFunction>();

function _getValidator(manifest: WidgetManifest): ValidateFunction {
  const key = `${manifest.kind}@${manifest.version}`;
  let v = _validatorCache.get(key);
  if (v === undefined) {
    v = _ajv.compile(manifest.propsSchema);
    _validatorCache.set(key, v);
  }
  return v;
}

/** Validates a props bag against the manifest's propsSchema. Returns all errors when invalid. */
export function validateProps(
  manifest: WidgetManifest,
  props: Record<string, unknown>,
): PropsValidationResult {
  const validate = _getValidator(manifest);
  const valid = validate(props) as boolean;
  if (!valid) {
    const errors = (validate.errors ?? []).map(
      (e) => `${e.instancePath || "(root)"} ${e.message ?? "invalid"}`,
    );
    return { valid: false, errors };
  }
  return { valid: true };
}
