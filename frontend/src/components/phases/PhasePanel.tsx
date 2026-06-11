import { useEffect } from "react";

import { useProject } from "@/hooks/useProject";
import { useRun } from "@/hooks/useRun";
import { useAppStore } from "@/store/app-store";
import type { Phase } from "@/lib/types";

import type { PanelProps } from "./shared";
import { InitPanel } from "./InitPanel";
import { ModelSelectPanel } from "./ModelSelectPanel";
import { StyleRefinePanel } from "./StyleRefinePanel";
import { BatchT2IPanel } from "./BatchT2IPanel";
import { BatchI2IPanel } from "./BatchI2IPanel";
import { CompletedPanel } from "./CompletedPanel";

interface PhasePanelProps {
  onSelectModel?: () => void;
  onAddRefs?: () => void;
}

function renderPhase(phase: Phase, panel: PanelProps) {
  switch (phase) {
    case "INIT":
      return <InitPanel {...panel} />;
    case "MODEL_SELECT":
      return <ModelSelectPanel {...panel} />;
    case "STYLE_REFINE":
      return <StyleRefinePanel {...panel} />;
    case "BATCH_T2I":
      return <BatchT2IPanel {...panel} />;
    case "BATCH_I2I":
      return <BatchI2IPanel {...panel} />;
    case "COMPLETED":
      return <CompletedPanel {...panel} />;
    default: {
      // Exhaustiveness guard: a new Phase will fail tsc here.
      const _exhaustive: never = phase;
      return _exhaustive;
    }
  }
}

export function PhasePanel({ onSelectModel, onAddRefs }: PhasePanelProps) {
  const { detail, gallery, refresh } = useProject();
  const { runStatus, runEvents, llmBuffer, run, resetRun } = useRun();
  const setViewing = useAppStore((s) => s.setViewing);

  // Pull fresh detail + gallery once each time a run finishes successfully so
  // the panel reflects new artifacts. `refresh` is a stable useCallback; keep
  // the dep array to [runStatus, refresh] to avoid a render loop.
  useEffect(() => {
    if (runStatus === "done") void refresh();
  }, [runStatus, refresh]);

  if (!detail) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-sm text-muted">
        加载中…
      </div>
    );
  }

  const panel: PanelProps = {
    gallery,
    runStatus,
    runEvents,
    llmBuffer,
    run,
    refresh,
    resetRun,
    onSelectModel,
    onAddRefs,
    // Only phases that actually persist history get a "查看历史" entry. INIT /
    // COMPLETED have no pass/round/batch slices to browse.
    onViewHistory: HISTORY_PHASES.has(detail.state.phase)
      ? () => setViewing({ phase: detail.state.phase })
      : undefined,
  };

  return renderPhase(detail.state.phase, panel);
}

const HISTORY_PHASES = new Set<Phase>([
  "MODEL_SELECT",
  "STYLE_REFINE",
  "BATCH_T2I",
  "BATCH_I2I",
]);
