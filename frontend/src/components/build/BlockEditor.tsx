"use client";
import { CATALOG, type Field as FieldDef } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import type { Block } from "@/lib/types";
import { Field, NumberInput, SelectInput, Toggle } from "../ui/kit";

/** Parameter form for one block, generated from the catalog. */
export function BlockEditor({ block, disabled }: { block: Block; disabled?: boolean }) {
  const update = useStudio((s) => s.updateBlock);
  const def = CATALOG[block.type];
  const fields = def.fields.filter((f) => !f.when || f.when(block));
  const val = (f: FieldDef) => (block as unknown as Record<string, unknown>)[f.key];

  if (!fields.length) return <p className="text-xs text-muted">No parameters — {def.blurb.toLowerCase()}</p>;

  return (
    <div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2.5 @md:grid-cols-3">
        {fields.map((f) => {
          const set = (v: unknown) => update(block.id, { [f.key]: v });
          if (f.kind === "bool") {
            return (
              <Field key={f.key} label={f.label} help={f.help} inline>
                <Toggle checked={!!val(f)} onChange={set} disabled={disabled} label={f.label} />
              </Field>
            );
          }
          return (
            <Field key={f.key} label={f.label} help={f.help}>
              {f.kind === "select" ? (
                <SelectInput value={String(val(f))} onChange={set} options={f.options} disabled={disabled} />
              ) : (
                <NumberInput value={Number(val(f))} onChange={set} min={f.min} max={f.max} step={f.kind === "float" ? f.step : (f.step ?? 1)} float={f.kind === "float"} disabled={disabled} />
              )}
            </Field>
          );
        })}
      </div>
      <p className="mt-3 text-[11px] leading-snug text-muted">
        <span className="font-medium text-fg/80">From transformer_rs:</span> {def.origin}
      </p>
    </div>
  );
}
