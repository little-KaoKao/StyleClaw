import { useEffect } from "react";

import { useProject } from "@/hooks/useProject";
import { useRun } from "@/hooks/useRun";
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

  // Pull fresh detail + gallery once each time a run finishes successfully so
  // the panel reflects new artifacts. `refresh` is a stable useCallback; keep
  // the dep array to [runStatus, refresh] to avoid a render loop.
  useEffect(() => {
    if (runStatus === "done") void refresh();
  }, [runStatus, refresh]);

  if (!detail) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-sm text-muted-foreground">
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
  };

  return renderPhase(detail.state.phase, panel);
}
