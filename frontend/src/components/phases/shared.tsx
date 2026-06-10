import type { ComponentProps, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { RunProgress } from "@/components/run/RunProgress";
import { RunControls } from "@/components/run/RunControls";
import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import type { ActionStep, Gallery, WsEvent } from "@/lib/types";

/**
 * Shared bundle threaded from PhasePanel down to every sub-panel. The hooks
 * (`useProject` / `useRun`) are called exactly once in PhasePanel — sub-panels
 * stay pure-prop so we never open a second WebSocket connection.
 */
export interface PanelProps {
  gallery: Gallery | null;
  runStatus: string;
  runEvents: WsEvent[];
  llmBuffer: string;
  run: (steps: ActionStep[]) => Promise<void>;
  refresh: () => Promise<void> | void;
  resetRun: () => void;
  onSelectModel?: () => void;
  onAddRefs?: () => void;
}

interface ActionButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  /** Picks up the restyled Button variants (red / blue / yellow / outline / ghost). */
  variant?: ComponentProps<typeof Button>["variant"];
  icon?: ReactNode;
}

/**
 * A single action button. Disabled while a run is in flight so the user can't
 * stack overlapping runs. Icons auto-size to h-5 w-5 via the Button primitive.
 */
export function ActionButton({
  label,
  onClick,
  disabled,
  variant = "outline",
  icon,
}: ActionButtonProps) {
  return (
    <Button size="default" variant={variant} onClick={onClick} disabled={disabled}>
      {icon}
      {label}
    </Button>
  );
}

/**
 * Build a click handler that starts a one-step run. `args` is only attached
 * when non-empty, keeping the wire payload minimal.
 */
export function makeRunHandler(
  run: PanelProps["run"],
  name: string,
  args?: Record<string, unknown>
) {
  return () => {
    const step: ActionStep =
      args && Object.keys(args).length > 0 ? { name, args } : { name };
    void run([step]);
  };
}

interface PanelShellProps {
  title: string;
  description: ReactNode;
  actions: ReactNode;
  /** Extra controls rendered between the description and the action row. */
  extra?: ReactNode;
  panel: PanelProps;
}

/**
 * Consistent layout for the action-driven phases: header, action row,
 * live run progress, run controls, then the gallery.
 */
export function PanelShell({
  title,
  description,
  actions,
  extra,
  panel,
}: PanelShellProps) {
  const { gallery, runStatus, runEvents, llmBuffer, refresh, resetRun } = panel;
  return (
    <div className="space-y-6 p-6">
      <header className="space-y-2">
        <h2 className="font-black uppercase tracking-tight text-2xl md:text-3xl text-foreground">
          {title}
        </h2>
        <p className="font-medium text-foreground/70">{description}</p>
      </header>

      {extra}

      <div className="flex flex-wrap items-center gap-3">{actions}</div>

      <RunProgress events={runEvents} llmBuffer={llmBuffer} status={runStatus} />

      <RunControls status={runStatus} onRefresh={refresh} onReset={resetRun} />

      <GalleryGrid
        groups={gallery?.groups ?? []}
        refImages={gallery?.ref_images}
      />
    </div>
  );
}
