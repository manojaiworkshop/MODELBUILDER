import type { Block } from "./types";

/** A user-created rectangular container in the Model stack (e.g. "Transformer Block 1"). Purely a
 * canvas-arrangement concept — it never reaches the backend; the Spec view and everything sent to
 * the server sees only the flat `blocks` array these group members already live in. */
export interface BlockGroup {
  id: string;
  label: string;
}

/** blockId -> id of the group it's currently inside; a block with no entry is top-level. Kept
 * contiguous by construction: every store action that changes membership also repositions the
 * block in the flat array, so a group's members are always a single unbroken run. */
export type GroupMembership = Record<string, string>;

export type StackItem =
  | { kind: "block"; block: Block }
  | { kind: "group"; group: BlockGroup; blocks: Block[] };

/** Walks the flat block list once, folding any run of consecutive blocks that share a group id
 * into one `StackItem`. If membership ever weren't contiguous (it shouldn't be, see above), the
 * same group would just render as two separate boxes rather than crash — an acceptable degrade. */
export function layoutStack(blocks: Block[], groupOf: GroupMembership, groups: BlockGroup[]): StackItem[] {
  const byId = new Map(groups.map((g) => [g.id, g]));
  const items: StackItem[] = [];
  let i = 0;
  while (i < blocks.length) {
    const gid = groupOf[blocks[i].id];
    const g = gid ? byId.get(gid) : undefined;
    if (!g) {
      items.push({ kind: "block", block: blocks[i] });
      i += 1;
      continue;
    }
    const run: Block[] = [blocks[i]];
    let j = i + 1;
    while (j < blocks.length && groupOf[blocks[j].id] === gid) {
      run.push(blocks[j]);
      j += 1;
    }
    items.push({ kind: "group", group: g, blocks: run });
    i = j;
  }
  return items;
}

/** Everything about a block except its (irrelevant-to-shape) id, for comparing two blocks' settings. */
function withoutId(b: Block): string {
  const o: Record<string, unknown> = { ...b };
  delete o.id;
  return JSON.stringify(o);
}

/** The pieces of a "one shared class, looped `count` times" rendering — or `null` when that
 * wouldn't be an honest picture of the computation (fewer than 2 repeats, the repeats aren't
 * identical two-block [Attention, FFN] groups, or something else sits between them). */
export interface LoopPlan {
  before: Block[];
  attn: Block;
  ffn: Block;
  count: number;
  after: Block[];
}

export function loopPlan(blocks: Block[], groupOf: GroupMembership, groups: BlockGroup[]): LoopPlan | null {
  const byId = new Set(groups.map((g) => g.id));
  interface Run { start: number; end: number; attn: Block; ffn: Block }
  const runs: Run[] = [];
  let i = 0;
  while (i < blocks.length - 1) {
    const gid = groupOf[blocks[i].id];
    const isPair =
      !!gid && byId.has(gid) &&
      blocks[i].type === "attention" && blocks[i + 1].type === "ffn" &&
      groupOf[blocks[i + 1].id] === gid &&
      groupOf[blocks[i + 2]?.id] !== gid; // the group must end here — a 3rd member would make this not a plain pair
    if (isPair) {
      runs.push({ start: i, end: i + 1, attn: blocks[i], ffn: blocks[i + 1] });
      i += 2;
    } else {
      i += 1;
    }
  }
  if (runs.length < 2) return null;

  let bestStart = 0, bestLen = 1, curStart = 0, curLen = 1;
  for (let k = 1; k < runs.length; k++) {
    const adjacent = runs[k].start === runs[k - 1].end + 1;
    const identical = adjacent && withoutId(runs[k].attn) === withoutId(runs[k - 1].attn) && withoutId(runs[k].ffn) === withoutId(runs[k - 1].ffn);
    if (identical) {
      curLen += 1;
    } else {
      curStart = k;
      curLen = 1;
    }
    if (curLen > bestLen) {
      bestLen = curLen;
      bestStart = curStart;
    }
  }
  if (bestLen < 2) return null;
  const first = runs[bestStart];
  const last = runs[bestStart + bestLen - 1];
  return { before: blocks.slice(0, first.start), attn: first.attn, ffn: first.ffn, count: bestLen, after: blocks.slice(last.end + 1) };
}

/** A label not already used by any existing group — so two boxes never end up sharing one name
 * (which would make their action buttons' aria-labels ambiguous). Tries `base` as-is first, then
 * `base 2`, `base 3`, … */
export function uniqueLabel(base: string, groups: BlockGroup[]): string {
  const used = new Set(groups.map((g) => g.label));
  if (!used.has(base)) return base;
  let n = 2;
  while (used.has(`${base} ${n}`)) n += 1;
  return `${base} ${n}`;
}

/** One-time seeding for freshly loaded blocks (a preset, or "reset"): finds adjacent
 * [Attention, FFN] pairs and wraps each in its own new group, so presets look organized
 * immediately without needing to hand-author group data alongside every preset. Never called
 * again afterwards — from then on, `groups`/`groupOf` are the only source of truth. */
export function seedGroups(blocks: Block[]): { groups: BlockGroup[]; groupOf: GroupMembership } {
  const groups: BlockGroup[] = [];
  const groupOf: GroupMembership = {};
  let i = 0;
  let n = 0;
  while (i < blocks.length - 1) {
    if (blocks[i].type === "attention" && blocks[i + 1].type === "ffn") {
      n += 1;
      const id = `seed-${n}-${blocks[i].id}`;
      groups.push({ id, label: `Transformer Block ${n}` });
      groupOf[blocks[i].id] = id;
      groupOf[blocks[i + 1].id] = id;
      i += 2;
    } else {
      i += 1;
    }
  }
  return { groups, groupOf };
}
