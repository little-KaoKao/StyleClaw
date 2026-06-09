import { useCallback } from "react";
import { startRun as apiStartRun } from "@/lib/api";
import { useWebSocket } from "@/lib/ws";
import { useAppStore } from "@/store/app-store";
import type { ActionStep, WsEvent } from "@/lib/types";

export function useRun() {
  const project = useAppStore((s) => s.currentProject);
  const runId = useAppStore((s) => s.runId);
  const runStatus = useAppStore((s) => s.runStatus);
  const runEvents = useAppStore((s) => s.runEvents);
  const llmBuffer = useAppStore((s) => s.llmBuffer);
  const pushEvent = useAppStore((s) => s.pushEvent);
  const startRunStore = useAppStore((s) => s.startRun);
  const resetRun = useAppStore((s) => s.resetRun);

  const onEvent = useCallback((ev: WsEvent) => pushEvent(ev), [pushEvent]);

  useWebSocket({ project: project || "", runId, onEvent });

  const run = useCallback(
    async (steps: ActionStep[]) => {
      if (!project) return;
      const { run_id } = await apiStartRun(project, steps);
      startRunStore(run_id);
    },
    [project, startRunStore]
  );

  return { runId, runStatus, runEvents, llmBuffer, run, resetRun };
}
