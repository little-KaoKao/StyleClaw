import { create } from "zustand";
import type { Gallery, ProjectDetail, ViewSlice, WsEvent } from "@/lib/types";

interface AppState {
  currentProject: string | null;
  setCurrentProject: (name: string | null) => void;

  detail: ProjectDetail | null;
  setDetail: (d: ProjectDetail | null) => void;

  gallery: Gallery | null;
  setGallery: (g: Gallery | null) => void;

  viewing: ViewSlice | null;
  setViewing: (v: ViewSlice | null) => void;

  runId: string | null;
  runStatus: "idle" | "running" | "done" | "error";
  runEvents: WsEvent[];
  llmBuffer: string;
  startRun: (runId: string) => void;
  pushEvent: (ev: WsEvent) => void;
  resetRun: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentProject: null,
  setCurrentProject: (name) =>
    set({ currentProject: name, detail: null, gallery: null, viewing: null }),

  detail: null,
  setDetail: (d) => set({ detail: d }),

  gallery: null,
  setGallery: (g) => set({ gallery: g }),

  viewing: null,
  setViewing: (v) => set({ viewing: v }),

  runId: null,
  runStatus: "idle",
  runEvents: [],
  llmBuffer: "",
  startRun: (runId) => set({ runId, runStatus: "running", runEvents: [], llmBuffer: "" }),
  pushEvent: (ev) =>
    set((s) => {
      const events = [...s.runEvents, ev];
      let status = s.runStatus;
      let llmBuffer = s.llmBuffer;
      if (ev.type === "done") status = "done";
      if (ev.type === "error") status = "error";
      if (ev.type === "llm_delta") llmBuffer += ev.text;
      if (ev.type === "step_start") llmBuffer = "";
      return { runEvents: events, runStatus: status, llmBuffer };
    }),
  resetRun: () => set({ runId: null, runStatus: "idle", runEvents: [], llmBuffer: "" }),
}));
