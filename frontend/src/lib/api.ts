import type {
  ActionPlan,
  ActionStep,
  Gallery,
  History,
  ProjectDetail,
  ProjectSummary,
  RunResponse,
  RunSnapshot,
  StepResult,
} from "./types";

const BASE = "";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${url}`, init);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// --- Projects ---

export async function listProjects(): Promise<{ projects: ProjectSummary[] }> {
  return json("/api/projects");
}

export async function getProject(name: string): Promise<ProjectDetail> {
  return json(`/api/projects/${name}`);
}

export async function getGallery(
  name: string,
  params?: { phase?: string; pass?: number; round?: number; batch?: number }
): Promise<Gallery> {
  const qs = new URLSearchParams();
  if (params?.phase) qs.set("phase", params.phase);
  if (params?.pass != null) qs.set("pass", String(params.pass));
  if (params?.round != null) qs.set("round", String(params.round));
  if (params?.batch != null) qs.set("batch", String(params.batch));
  const q = qs.toString();
  return json(`/api/projects/${name}/gallery${q ? "?" + q : ""}`);
}

export async function getHistory(name: string): Promise<History> {
  return json(`/api/projects/${name}/history`);
}

export async function rollback(
  name: string,
  to: string,
  round?: number
): Promise<{ ok: boolean; message: string; data: unknown }> {
  return json(`/api/projects/${name}/rollback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to, round }),
  });
}

// --- Runs ---

export async function previewPlan(
  name: string,
  intent: string
): Promise<ActionPlan> {
  return json(`/api/projects/${name}/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ intent }),
  });
}

export async function startRun(
  name: string,
  steps: ActionStep[],
  opts?: { loop?: object; summary?: string }
): Promise<RunResponse> {
  return json(`/api/projects/${name}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, ...opts }),
  });
}

export async function getRun(
  name: string,
  runId: string
): Promise<RunSnapshot> {
  return json(`/api/projects/${name}/runs/${runId}`);
}

// --- Confirm actions ---

export async function createProject(form: FormData): Promise<StepResult> {
  const resp = await fetch(`${BASE}/api/projects`, { method: "POST", body: form });
  return resp.json();
}

export async function selectModel(
  name: string,
  models: string,
  variant: string
): Promise<StepResult> {
  return json(`/api/projects/${name}/select-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models, variant }),
  });
}

export async function addRefs(name: string, form: FormData): Promise<StepResult> {
  const resp = await fetch(`${BASE}/api/projects/${name}/refs`, {
    method: "POST",
    body: form,
  });
  return resp.json();
}
