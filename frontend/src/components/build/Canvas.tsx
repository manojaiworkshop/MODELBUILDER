"use client";
import {
  closestCenter, DndContext, DragOverlay, KeyboardSensor, PointerSensor, TouchSensor, useDroppable, useSensor, useSensors,
  type DragEndEvent, type DragOverEvent, type DragStartEvent,
} from "@dnd-kit/core";
import { SortableContext, sortableKeyboardCoordinates, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { ArrowDown, Boxes, Check, Copy, GripVertical, MousePointerClick, Pencil, Plus, Trash2, Ungroup, X, Lock } from "lucide-react";
import { useState } from "react";
import { CATALOG } from "@/lib/blocks";
import { PRESETS } from "@/lib/presets";
import { isRunning, useStudio } from "@/lib/store";
import { layoutStack, type BlockGroup } from "@/lib/transformerGroups";
import type { Block } from "@/lib/types";
import { BlockCard } from "./BlockCard";
import { Palette, type PaletteKind } from "./Palette";
import { Button, cn } from "../ui/kit";

function EndZone({ active, empty }: { active: boolean; empty: boolean }) {
  const { setNodeRef, isOver } = useDroppable({ id: "canvas-end" });
  const setBlocks = useStudio((s) => s.setBlocks);
  return (
    <div
      ref={setNodeRef}
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border-2 border-dashed text-center transition-colors",
        empty ? "min-h-40 px-4 py-8" : "min-h-14 px-3 py-3",
        isOver ? "border-accent bg-accent-soft" : active ? "border-border-strong" : "border-border",
      )}
    >
      {empty ? (
        <>
          <MousePointerClick className="mb-2 text-muted" size={22} />
          <p className="text-sm font-medium">Drag blocks here to build your transformer</p>
          <p className="mt-1 text-xs text-muted">Start with an Embedding, stack layers, finish with an LM Head — or load a preset.</p>
          <div className="mt-3 flex flex-wrap justify-center gap-2">
            {PRESETS.filter((p) => p.id !== "empty").slice(0, 3).map((p) => (
              <Button key={p.id} size="sm" onClick={() => setBlocks(p.build())}>{p.name}</Button>
            ))}
          </div>
        </>
      ) : (
        <p className="text-xs text-muted">{isOver ? "Release to add at the end" : "Drop here to add at the end"}</p>
      )}
    </div>
  );
}

/** Persistent "append to this group" strip at the bottom of every Transformer Block box — the
 * reliable way to drop a block INTO a group (dropping on the box's own header/padding instead
 * reorders it among the top-level stack, since that's not aimed at any specific member). */
function GroupDropZone({ groupId, active, empty }: { groupId: string; active: boolean; empty: boolean }) {
  const { setNodeRef, isOver } = useDroppable({ id: `gempty:${groupId}` });
  return (
    <div
      ref={setNodeRef}
      className={cn(
        "flex items-center justify-center gap-1.5 rounded-lg border border-dashed text-[11px] transition-colors",
        empty ? "min-h-12 py-3" : "py-1.5",
        isOver ? "border-accent bg-accent-soft text-accent" : active ? "border-accent/40 text-muted" : "border-transparent text-transparent",
      )}
    >
      <Plus size={12} /> {empty ? "Drop a block here" : "Drop to add to this block"}
    </div>
  );
}

function Connector() {
  return (
    <div className="flex h-6 items-center justify-center text-border-strong" aria-hidden>
      <ArrowDown size={14} />
    </div>
  );
}

/** A solid rectangular container any block can be dragged into or out of — the "Transformer
 * Block" box from the classic architecture diagram. Draggable as a whole unit (grip in the
 * header) to reorder among top-level items, and hosts its own nested sortable list for members. */
function GroupBox({ group, blocks, locked, overId, dragActive }: { group: BlockGroup; blocks: Block[]; locked: boolean; overId: string | null; dragActive: boolean }) {
  const allBlocks = useStudio((s) => s.blocks);
  const inspect = useStudio((s) => s.inspect);
  const duplicateGroup = useStudio((s) => s.duplicateGroup);
  const removeGroup = useStudio((s) => s.removeGroup);
  const ungroup = useStudio((s) => s.ungroup);
  const renameGroup = useStudio((s) => s.renameGroup);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(group.label);

  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } = useSortable({
    id: `g:${group.id}`,
    disabled: locked,
  });

  const remove = () => {
    if (!confirm(`Remove this whole ${group.label} (${blocks.length} block${blocks.length === 1 ? "" : "s"})?`)) return;
    removeGroup(group.id);
  };
  const commitRename = () => {
    const label = draft.trim();
    if (label) renameGroup(group.id, label);
    setEditing(false);
  };

  const card = (b: Block) => {
    const index = allBlocks.findIndex((x) => x.id === b.id);
    const issues = (inspect?.issues ?? []).filter((iss) => iss.blockId === b.id);
    const shape = inspect && inspect.shapes.length === allBlocks.length ? inspect.shapes.find((s) => s.blockId === b.id) : undefined;
    return (
      <BlockCard
        key={b.id}
        block={b}
        index={index}
        locked={locked}
        issues={issues}
        shape={shape}
        dropBefore={dragActive && overId === `b:${b.id}`}
      />
    );
  };

  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Translate.toString(transform), transition, opacity: isDragging ? 0.4 : 1 }}
      className={cn(
        "rounded-xl border-2 bg-accent-soft/25 p-2.5 transition-colors",
        overId === `g:${group.id}` ? "border-accent" : "border-accent/40",
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-2 px-1">
        {editing ? (
          <div className="flex min-w-0 flex-1 items-center gap-1">
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitRename();
                if (e.key === "Escape") setEditing(false);
              }}
              className="min-w-0 flex-1 rounded border border-accent/50 bg-surface px-1.5 py-0.5 text-xs font-semibold text-accent outline-none"
            />
            <button aria-label="Save name" onClick={commitRename} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-accent">
              <Check size={13} />
            </button>
            <button aria-label="Cancel rename" onClick={() => setEditing(false)} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-danger">
              <X size={13} />
            </button>
          </div>
        ) : (
          <button
            ref={setActivatorNodeRef}
            {...attributes}
            {...listeners}
            disabled={locked}
            title="Drag to reorder this whole block"
            className={cn("flex min-w-0 flex-1 touch-none items-center gap-1.5 text-left text-xs font-semibold text-accent", locked ? "cursor-not-allowed opacity-60" : "cursor-grab active:cursor-grabbing")}
          >
            <GripVertical size={13} className="shrink-0 opacity-60" />
            <Boxes size={13} className="shrink-0" />
            <span className="truncate">{group.label}</span>
            <span className="shrink-0 text-[10px] font-normal text-accent/60">× {blocks.length}</span>
          </button>
        )}
        {!editing && (
          <div className="flex shrink-0 items-center gap-0.5">
            <button aria-label={`Rename ${group.label}`} title="Rename this block" disabled={locked} onClick={() => { setDraft(group.label); setEditing(true); }} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-accent disabled:opacity-30">
              <Pencil size={13} />
            </button>
            <button aria-label={`Duplicate ${group.label}`} title="Duplicate this whole block" disabled={locked} onClick={() => duplicateGroup(group.id)} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-accent disabled:opacity-30">
              <Copy size={13} />
            </button>
            <button aria-label={`Ungroup ${group.label}`} title="Dissolve this container, keep its blocks" disabled={locked} onClick={() => ungroup(group.id)} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-accent disabled:opacity-30">
              <Ungroup size={13} />
            </button>
            <button aria-label={`Remove ${group.label}`} title="Remove this whole block" disabled={locked} onClick={remove} className="rounded p-1 text-accent/70 hover:bg-surface hover:text-danger disabled:opacity-30">
              <Trash2 size={13} />
            </button>
          </div>
        )}
      </div>
      <SortableContext items={blocks.map((b) => `b:${b.id}`)} strategy={verticalListSortingStrategy}>
        <div className="flex flex-col">
          {blocks.map((b, i) => (
            <div key={b.id}>
              {card(b)}
              {i < blocks.length - 1 && <Connector />}
            </div>
          ))}
        </div>
      </SortableContext>
      <div className={blocks.length ? "mt-1.5" : "mt-0"}>
        <GroupDropZone groupId={group.id} active={dragActive} empty={blocks.length === 0} />
      </div>
    </div>
  );
}

export function Canvas() {
  const blocks = useStudio((s) => s.blocks);
  const groups = useStudio((s) => s.groups);
  const groupOf = useStudio((s) => s.groupOf);
  const addBlockTo = useStudio((s) => s.addBlockTo);
  const addTransformerBlock = useStudio((s) => s.addTransformerBlock);
  const moveBlockTo = useStudio((s) => s.moveBlockTo);
  const moveGroupTo = useStudio((s) => s.moveGroupTo);
  const inspect = useStudio((s) => s.inspect);
  const locked = useStudio((s) => isRunning(s.status));
  const [dragKind, setDragKind] = useState<PaletteKind | null>(null);
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 180, tolerance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const firstMemberIdOfGroup = (gid: string) => blocks.find((b) => groupOf[b.id] === gid)?.id ?? null;

  const flatIndexFor = (id: string): number => {
    if (id === "canvas-end") return blocks.length;
    if (id.startsWith("gempty:")) {
      const gid = id.slice(7);
      let last = -1;
      for (let k = 0; k < blocks.length; k++) if (groupOf[blocks[k].id] === gid) last = k;
      return last >= 0 ? last + 1 : blocks.length;
    }
    if (id.startsWith("g:")) {
      const idx = blocks.findIndex((b) => groupOf[b.id] === id.slice(2));
      return idx < 0 ? blocks.length : idx;
    }
    if (id.startsWith("b:")) {
      const idx = blocks.findIndex((b) => b.id === id.slice(2));
      return idx < 0 ? blocks.length : idx;
    }
    return blocks.length;
  };

  const resolveDrop = (id: string): { groupId: string | null; before: string | null } => {
    if (id === "canvas-end") return { groupId: null, before: null };
    if (id.startsWith("gempty:")) return { groupId: id.slice(7), before: null };
    if (id.startsWith("g:")) return { groupId: null, before: firstMemberIdOfGroup(id.slice(2)) };
    if (id.startsWith("b:")) {
      const blockId = id.slice(2);
      return { groupId: groupOf[blockId] ?? null, before: blockId };
    }
    return { groupId: null, before: null };
  };

  const onStart = (e: DragStartEvent) => {
    const id = String(e.active.id);
    if (id.startsWith("palette:")) setDragKind(id.slice(8) as PaletteKind);
    else setDragId(id);
  };
  const onOver = (e: DragOverEvent) => setOverId(e.over ? String(e.over.id) : null);
  const clear = () => {
    setDragKind(null);
    setDragId(null);
    setOverId(null);
  };
  const onEnd = (e: DragEndEvent) => {
    const active = String(e.active.id);
    const over = e.over ? String(e.over.id) : null;
    clear();
    if (!over || active === over) return;

    if (active.startsWith("palette:")) {
      const kind = active.slice(8) as PaletteKind;
      if (kind === "transformer") addTransformerBlock(flatIndexFor(over));
      else {
        const { groupId, before } = resolveDrop(over);
        addBlockTo(kind, groupId, before);
      }
      return;
    }

    if (active.startsWith("g:")) {
      const groupId = active.slice(2);
      let anchor: string | null;
      if (over.startsWith("b:")) {
        const targetId = over.slice(2);
        const targetGid = groupOf[targetId];
        if (targetGid === groupId) return; // hovering our own member — no-op
        anchor = targetGid ? firstMemberIdOfGroup(targetGid) : targetId;
      } else {
        anchor = resolveDrop(over).before;
      }
      moveGroupTo(groupId, anchor);
      return;
    }

    if (active.startsWith("b:")) {
      const blockId = active.slice(2);
      const { groupId, before } = resolveDrop(over);
      moveBlockTo(blockId, groupId, before);
    }
  };

  const issuesFor = (id: string) => (inspect?.issues ?? []).filter((i) => i.blockId === id);
  const global = (inspect?.issues ?? []).filter((i) => !i.blockId && i.level !== "info");
  const infos = (inspect?.issues ?? []).filter((i) => !i.blockId && i.level === "info");
  const shapeOf = (id: string) => (inspect && inspect.shapes.length === blocks.length ? inspect.shapes.find((s) => s.blockId === id) : undefined);
  const dragActive = !!dragKind || !!dragId;
  const ghost = dragKind
    ? dragKind === "transformer" ? "Transformer Block" : CATALOG[dragKind].title
    : dragId
      ? (() => {
          if (dragId.startsWith("g:")) return groups.find((g) => g.id === dragId.slice(2))?.label ?? "";
          const b = blocks.find((x) => `b:${x.id}` === dragId);
          return b ? CATALOG[b.type].title : "";
        })()
      : "";

  const stack = layoutStack(blocks, groupOf, groups);
  const outerItems = stack.map((item) => (item.kind === "block" ? `b:${item.block.id}` : `g:${item.group.id}`));

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragStart={onStart} onDragOver={onOver} onDragEnd={onEnd} onDragCancel={clear}>
      <div className="@container flex min-h-0 flex-1">
        <div className="flex w-20 shrink-0 flex-col items-center border-r border-border py-3">
          <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted">Blocks</h2>
          <div className="flex flex-1 items-center">
            <Palette locked={locked} />
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {locked && (
            <div className="mb-3 flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs text-muted">
              <Lock size={13} /> Editing is locked while training. Stop the run to change the stack.
            </div>
          )}
          {global.map((i, k) => (
            <div key={k} className={cn("mb-2 rounded-lg border px-3 py-2 text-xs", i.level === "error" ? "border-danger/40 bg-danger/5 text-danger" : "border-warn/40 bg-warn/5 text-warn")}>
              {i.message}
            </div>
          ))}
          <SortableContext items={outerItems} strategy={verticalListSortingStrategy}>
            <div className="flex flex-col">
              {stack.map((item, i) => (
                <div key={item.kind === "block" ? item.block.id : item.group.id}>
                  {item.kind === "block" ? (
                    <BlockCard
                      block={item.block}
                      index={blocks.findIndex((b) => b.id === item.block.id)}
                      locked={locked}
                      issues={issuesFor(item.block.id)}
                      shape={shapeOf(item.block.id)}
                      dropBefore={dragActive && overId === `b:${item.block.id}`}
                    />
                  ) : (
                    <GroupBox group={item.group} blocks={item.blocks} locked={locked} overId={overId} dragActive={dragActive} />
                  )}
                  {i < stack.length - 1 && <Connector />}
                </div>
              ))}
            </div>
          </SortableContext>
          <div className={blocks.length ? "mt-3" : ""}>
            <EndZone active={dragActive} empty={blocks.length === 0} />
          </div>
          {infos.map((i, k) => (
            <p key={k} className="mt-3 rounded-lg bg-accent-soft px-3 py-2 text-xs text-accent">{i.message}</p>
          ))}
        </div>
      </div>
      <DragOverlay dropAnimation={{ duration: 160 }}>
        {ghost ? <div className="rounded-lg border border-accent bg-surface px-3 py-2 text-sm font-medium shadow-card">{ghost}</div> : null}
      </DragOverlay>
    </DndContext>
  );
}
