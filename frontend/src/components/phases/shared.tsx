import type { ComponentProps, ReactNode } from "react";
import { History } from "lucide-react";

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
  /** Open this phase's read-only history (pass/round/batch browser). */
  onViewHistory?: () => void;
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
 * The filled gradient variants (`default` / `primary` / `red`) are the phase's
 * MAIN "推进" action — rendered one tier larger than the supporting buttons so
 * the visual hierarchy is obvious at a glance.
 */
const PRIMARY_VARIANTS = new Set(["default", "primary", "red"]);

/**
 * A single action button. Disabled while a run is in flight so the user can't
 * stack overlapping runs. Buttons are split into two visual tiers:
 *
 *   • primary (filled gradient): taller `h-12`, wider min-width — the lead action.
 *   • secondary (white / outline / ghost): `h-11`, uniform smaller min-width.
 *
 * Within a tier every button shares the same height AND minimum width, so a row
 * of them lines up tidily instead of each hugging its label. Across tiers the
 * size + fill difference makes "which one is the main action" unmistakable.
 */
export function ActionButton({
  label,
  onClick,
  disabled,
  variant = "outline",
  icon,
}: ActionButtonProps) {
  const isPrimary = PRIMARY_VARIANTS.has(variant ?? "outline");
  return (
    <Button
      size={isPrimary ? "default" : "sm"}
      variant={variant}
      onClick={onClick}
      disabled={disabled}
      className={isPrimary ? "min-w-[9.5rem]" : "min-w-[7.5rem]"}
    >
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
  const { gallery, runStatus, runEvents, llmBuffer, refresh, resetRun, onViewHistory } = panel;
  const hasRun = runStatus !== "idle" || runEvents.length > 0;
  return (
    <div className="space-y-6 p-6">
      <header className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <h2
            className="text-3xl font-black tracking-tight text-foreground md:text-4xl"
            style={{ fontFamily: "Nunito, sans-serif" }}
          >
            {title}
          </h2>
          <p className="font-medium text-muted">{description}</p>
        </div>
        {onViewHistory && (
          <Button variant="soft" size="sm" onClick={onViewHistory} className="shrink-0">
            <History />
            查看历史
          </Button>
        )}
      </header>

      {extra}

      <section className="space-y-2">
        <SectionLabel>操作 · Actions</SectionLabel>
        <div className="flex flex-wrap items-center gap-3">{actions}</div>
      </section>

      {hasRun && (
        <section className="space-y-2">
          <SectionLabel>运行状态 · Status</SectionLabel>
          <RunProgress events={runEvents} llmBuffer={llmBuffer} status={runStatus} />
          <RunControls status={runStatus} onRefresh={refresh} onReset={resetRun} />
        </section>
      )}

      <section className="space-y-2">
        <SectionLabel>结果 · Results</SectionLabel>
        <GalleryGrid
          groups={gallery?.groups ?? []}
          refImages={gallery?.ref_images}
          trigger={gallery?.trigger}
        />
      </section>
    </div>
  );
}

/** A small divider label that makes each region's purpose unambiguous. */
function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 pb-1 text-xs font-bold uppercase tracking-widest text-muted">
      <span className="h-2 w-2 shrink-0 rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED]" />
      {children}
    </div>
  );
}
