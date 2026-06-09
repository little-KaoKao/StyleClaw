// Mirrors backend Pydantic models

export type Phase =
  | "INIT"
  | "MODEL_SELECT"
  | "STYLE_REFINE"
  | "BATCH_T2I"
  | "BATCH_I2I"
  | "COMPLETED";

export interface ProjectSummary {
  name: string;
  phase: Phase;
  current_round: number;
  current_batch: number;
  last_updated: string;
}

export interface ProjectState {
  phase: Phase;
  selected_models: string[];
  selected_variant: string;
  current_round: number;
  current_batch: number;
  current_model_select_pass: number;
  last_updated: string;
}

export interface ProjectConfig {
  name: string;
  ip_info: string;
  ref_images: string[];
  sref_index: number;
}

export interface ProjectDetail {
  state: ProjectState;
  config: ProjectConfig;
  suggestions: string[];
}

export interface GalleryGroup {
  label: string;
  images: string[];
  scores: Record<string, number> | null;
}

export interface Gallery {
  phase: Phase;
  ref_images: string[];
  groups: GalleryGroup[];
}

export interface ActionStep {
  name: string;
  description?: string;
  args?: Record<string, unknown>;
}

export interface ActionPlan {
  summary: string;
  steps: ActionStep[];
  loop: { start_step: number; end_step: number; max_iterations: number } | null;
  stop_summary: string;
}

export interface RunResponse {
  run_id: string;
}

export interface RunSnapshot {
  run_id: string;
  project: string;
  status: "running" | "done" | "error";
  events: WsEvent[];
}

export interface StepResult {
  ok: boolean;
  message: string;
  data: Record<string, unknown> | null;
}

// WebSocket event types
export type WsEvent =
  | { type: "run_started"; run_id: string; project: string; kind: string; steps: string[] }
  | { type: "step_start"; index: number; name: string; description: string }
  | { type: "llm_delta"; step_index: number; role: string; text: string }
  | { type: "step_done"; index: number; name: string; status: string; summary: string }
  | { type: "needs_human"; round: number; weakest_dim: string; score: number; suggestion: string }
  | { type: "phase_paused"; phase: string; next_phase: string }
  | { type: "done"; run_id: string }
  | { type: "error"; message: string; detail: string };
