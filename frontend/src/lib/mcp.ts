import type { McpServer, McpTool } from "./types";

/** The name a tool has in tool-call data: `<server>__<tool>` (e.g. `fusedtrackdata__view_rows`). */
export const qualifiedToolName = (server: Pick<McpServer, "name">, tool: Pick<McpTool, "name">) => `${server.name}__${tool.name}`;

export const MCP_NAME_RE = /^[A-Za-z0-9_-]{1,40}$/;

export interface ToolParam {
  name: string;
  /** Compact type, e.g. `integer`, `string | null`, `array<string>`. */
  type: string;
  required: boolean;
  description: string;
  enumValues: string[] | null;
  defaultValue: string | null;
}

type Schema = Record<string, unknown>;
const isObj = (v: unknown): v is Schema => typeof v === "object" && v !== null && !Array.isArray(v);

/** One-line type of a JSON Schema fragment (handles `type` arrays, anyOf/oneOf, arrays, enums). */
function typeOf(s: unknown): string {
  if (!isObj(s)) return "any";
  const t = s.type;
  if (Array.isArray(t)) return t.map(String).join(" | ");
  if (typeof t === "string") {
    if (t === "array") return `array<${typeOf(s.items)}>`;
    return t;
  }
  const alt = (s.anyOf ?? s.oneOf) as unknown;
  if (Array.isArray(alt)) return [...new Set(alt.map(typeOf))].join(" | ");
  if (Array.isArray(s.enum)) return "enum";
  if (isObj(s.properties)) return "object";
  return "any";
}

/** The top-level arguments of a tool, from its input schema. */
export function toolParams(tool: McpTool): ToolParam[] {
  const props = isObj(tool.inputSchema.properties) ? tool.inputSchema.properties : {};
  const required = new Set(Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : []);
  return Object.entries(props).map(([name, raw]) => {
    const s = isObj(raw) ? raw : {};
    const alt = (s.anyOf ?? s.oneOf) as unknown;
    const enumRaw = Array.isArray(s.enum) ? s.enum : Array.isArray(alt) ? alt.flatMap((a) => (isObj(a) && Array.isArray(a.enum) ? a.enum : [])) : [];
    return {
      name,
      type: typeOf(s),
      required: required.has(name),
      description: typeof s.description === "string" ? s.description : "",
      enumValues: enumRaw.length ? enumRaw.map(String) : null,
      defaultValue: s.default === undefined ? null : JSON.stringify(s.default),
    };
  });
}

/** "KEY=value" per line -> object (blank lines and `#` comments skipped). Returns the first bad line, if any. */
export function parseEnv(text: string): { env: Record<string, string>; error: string | null } {
  const env: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const i = line.indexOf("=");
    if (i < 1) return { env, error: `“${line}” is not KEY=value.` };
    env[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { env, error: null };
}

export const parseArgs = (text: string) => text.split("\n").map((l) => l.trim()).filter(Boolean);
