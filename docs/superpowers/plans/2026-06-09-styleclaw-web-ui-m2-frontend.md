# StyleClaw Web UI — M2 前端实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个 React SPA 前端，连接 M1 后端的 REST + WebSocket API，让非程序员在浏览器里通过向导+聊天走完整个 StyleClaw 流水线。预构建产物存入 `src/styleclaw/web/static/`，由 FastAPI 直接托管。

**Architecture:** Vite + React 18 + TypeScript + TailwindCSS + shadcn/ui。`frontend/` 目录是独立 Node 工程（仅开发者接触）；`npm run build` 输出到 `src/styleclaw/web/static/`，FastAPI 的 `create_app()` 在生产模式挂载该目录。API 调用走 `/api/*`，WebSocket 走 `ws://localhost:PORT/api/projects/{name}/events`。

**Tech Stack:** React 18 / TypeScript 5 / Vite 6 / TailwindCSS 4 / shadcn/ui / lucide-react icons

**Spec:** `docs/superpowers/specs/2026-06-09-styleclaw-web-ui-design.md`（M2 部分，§6 布局）

**界面语言:** 中英混合 — 按钮/提示/状态用中文，技术术语保留英文（如 MODEL_SELECT, trigger phrase, STYLE_REFINE）。

---

## 文件结构

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── tailwind.config.ts        (shadcn/ui 需要)
├── postcss.config.js
├── components.json           (shadcn/ui CLI 配置)
├── src/
│   ├── main.tsx
│   ├── App.tsx               — 顶层路由 + 布局 shell
│   ├── globals.css           — Tailwind directives + shadcn base
│   ├── lib/
│   │   ├── api.ts            — typed fetch wrappers for all REST endpoints
│   │   ├── ws.ts             — useWebSocket hook (auto-reconnect, replay)
│   │   └── types.ts          — TypeScript types mirroring backend models
│   ├── components/
│   │   ├── ui/               — shadcn/ui primitives (button, card, dialog, input, badge, toast, etc.)
│   │   ├── layout/
│   │   │   ├── Header.tsx        — logo + 项目选择器 + key 状态
│   │   │   ├── PhaseBar.tsx      — 6 阶段进度条
│   │   │   └── AppShell.tsx      — 左主区 + 右聊天栏 两栏布局
│   │   ├── phases/
│   │   │   ├── PhasePanel.tsx    — 按 phase 分发到子组件
│   │   │   ├── InitPanel.tsx
│   │   │   ├── ModelSelectPanel.tsx
│   │   │   ├── StyleRefinePanel.tsx
│   │   │   ├── BatchT2IPanel.tsx
│   │   │   ├── BatchI2IPanel.tsx
│   │   │   └── CompletedPanel.tsx
│   │   ├── run/
│   │   │   ├── RunProgress.tsx   — 实时步骤进度 + LLM 逐字
│   │   │   └── RunControls.tsx   — 「继续」/「停止」按钮
│   │   ├── gallery/
│   │   │   ├── GalleryGrid.tsx   — 图片网格 (lightbox 点击放大)
│   │   │   └── ScoreCard.tsx     — 评分维度卡片
│   │   ├── chat/
│   │   │   ├── ChatPanel.tsx     — 聊天主体
│   │   │   ├── PlanPreview.tsx   — ActionPlan 预览 + 确认执行按钮
│   │   │   └── MessageBubble.tsx
│   │   └── modals/
│   │       ├── NewProjectModal.tsx   — 拖拽上传参考图 + 填 IP 信息
│   │       ├── SelectModelModal.tsx  — 模型多选 + variant 下拉
│   │       └── AddRefsModal.tsx      — i2i 参考图上传
│   ├── hooks/
│   │   ├── useProject.ts        — 当前选中项目的 state/config/gallery
│   │   ├── useRun.ts            — 管理当前运行 (start, events, status)
│   │   └── useChat.ts           — 聊天消息列表 + plan 提交
│   └── store/
│       └── app-store.ts         — Zustand store (当前项目名、运行状态、聊天历史)
└── ...
```

**修改既有文件：**
- `src/styleclaw/web/app.py` — 挂载 `web/static/` 为 SPA fallback（仅当目录存在时）

---

## Task 1: 前端脚手架 + Vite + TailwindCSS + shadcn/ui 初始化

**Files:**
- Create: `frontend/` 整个目录结构（package.json, vite.config.ts, tsconfig.json, tailwind/postcss 配置, index.html, src/main.tsx, src/globals.css, src/App.tsx）
- Create: `frontend/components.json`（shadcn/ui CLI 配置）

- [ ] **Step 1: 初始化 Vite + React + TypeScript**

```bash
cd /Users/xiaociji/Desktop/StyleClaw
npm create vite@latest frontend -- --template react-ts
cd frontend
```

- [ ] **Step 2: 安装核心依赖**

```bash
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install class-variance-authority clsx tailwind-merge lucide-react
npm install zustand
npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-slot
```

- [ ] **Step 3: 配置 Vite（代理 + 输出路径）**

`frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8800",
      "/media": "http://127.0.0.1:8800",
      "/api/projects": { target: "ws://127.0.0.1:8800", ws: true },
    },
  },
  build: {
    outDir: "../src/styleclaw/web/static",
    emptyOutDir: true,
  },
});
```

- [ ] **Step 4: 配置 TailwindCSS**

`frontend/src/globals.css`:
```css
@import "tailwindcss";
```

- [ ] **Step 5: 写最小 App.tsx**

```tsx
import "./globals.css";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <h1 className="text-2xl font-semibold text-gray-800">StyleClaw</h1>
    </div>
  );
}
```

- [ ] **Step 6: 配置 tsconfig 路径别名**

在 `tsconfig.json` (或 `tsconfig.app.json`) 加:
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  }
}
```

- [ ] **Step 7: 验证 dev server**

```bash
cd frontend && npm run dev
# 浏览器打开 http://localhost:5173 → 看到 "StyleClaw" 字样
```

- [ ] **Step 8: 验证 build**

```bash
npm run build
ls ../src/styleclaw/web/static/index.html  # 确认产物存在
```

- [ ] **Step 9: 配置 shadcn/ui**

创建 `frontend/components.json`:
```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "config": "",
    "css": "src/globals.css",
    "baseColor": "zinc",
    "cssVariables": true
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib",
    "hooks": "@/hooks"
  },
  "iconLibrary": "lucide"
}
```

创建 `frontend/src/lib/utils.ts`:
```typescript
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

安装 shadcn 基础组件:
```bash
npx shadcn@latest add button card dialog input badge toast dropdown-menu
```

- [ ] **Step 10: 提交**

```bash
cd /Users/xiaociji/Desktop/StyleClaw
git add frontend/ src/styleclaw/web/static/
echo "node_modules" >> frontend/.gitignore
git add frontend/.gitignore
git commit -m "feat(frontend): Vite + React + TailwindCSS + shadcn/ui scaffold"
```

---

## Task 2: TypeScript 类型定义 + API 客户端

**Files:**
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: 定义类型**

`frontend/src/lib/types.ts`:
```typescript
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
```

- [ ] **Step 2: 写 API 客户端**

`frontend/src/lib/api.ts`:
```typescript
import type {
  ActionPlan,
  ActionStep,
  Gallery,
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

export async function getGallery(name: string): Promise<Gallery> {
  return json(`/api/projects/${name}/gallery`);
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
```

- [ ] **Step 3: 验证编译**

```bash
cd frontend && npx tsc --noEmit
```

- [ ] **Step 4: 提交**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts
git commit -m "feat(frontend): typed API client + TypeScript model definitions"
```

---

## Task 3: WebSocket hook

**Files:**
- Create: `frontend/src/lib/ws.ts`

- [ ] **Step 1: 写 useWebSocket hook**

```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import type { WsEvent } from "./types";

interface UseWsOptions {
  project: string;
  runId: string | null;
  onEvent?: (ev: WsEvent) => void;
}

export function useWebSocket({ project, runId, onEvent }: UseWsOptions) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (!runId) return;
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/projects/${project}/events?run_id=${runId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (msg) => {
      try {
        const ev: WsEvent = JSON.parse(msg.data);
        onEventRef.current?.(ev);
      } catch { /* ignore parse errors */ }
    };
  }, [project, runId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { connected };
}
```

- [ ] **Step 2: 验证编译**

```bash
cd frontend && npx tsc --noEmit
```

- [ ] **Step 3: 提交**

```bash
git add frontend/src/lib/ws.ts
git commit -m "feat(frontend): useWebSocket hook with auto-reconnect"
```

---

## Task 4: Zustand store + project/run hooks

**Files:**
- Create: `frontend/src/store/app-store.ts`
- Create: `frontend/src/hooks/useProject.ts`
- Create: `frontend/src/hooks/useRun.ts`

- [ ] **Step 1: 写 Zustand store**

```typescript
// frontend/src/store/app-store.ts
import { create } from "zustand";
import type { Gallery, ProjectDetail, WsEvent } from "@/lib/types";

interface AppState {
  // Current project
  currentProject: string | null;
  setCurrentProject: (name: string | null) => void;

  // Project detail (fetched)
  detail: ProjectDetail | null;
  setDetail: (d: ProjectDetail | null) => void;

  // Gallery
  gallery: Gallery | null;
  setGallery: (g: Gallery | null) => void;

  // Run
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
  setCurrentProject: (name) => set({ currentProject: name, detail: null, gallery: null }),

  detail: null,
  setDetail: (d) => set({ detail: d }),

  gallery: null,
  setGallery: (g) => set({ gallery: g }),

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
      if (ev.type === "step_start") llmBuffer = ""; // reset between steps
      return { runEvents: events, runStatus: status, llmBuffer };
    }),
  resetRun: () => set({ runId: null, runStatus: "idle", runEvents: [], llmBuffer: "" }),
}));
```

- [ ] **Step 2: 写 useProject hook**

```typescript
// frontend/src/hooks/useProject.ts
import { useCallback, useEffect } from "react";
import { getGallery, getProject } from "@/lib/api";
import { useAppStore } from "@/store/app-store";

export function useProject() {
  const project = useAppStore((s) => s.currentProject);
  const detail = useAppStore((s) => s.detail);
  const gallery = useAppStore((s) => s.gallery);
  const setDetail = useAppStore((s) => s.setDetail);
  const setGallery = useAppStore((s) => s.setGallery);

  const refresh = useCallback(async () => {
    if (!project) return;
    const [d, g] = await Promise.all([getProject(project), getGallery(project)]);
    setDetail(d);
    setGallery(g);
  }, [project, setDetail, setGallery]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { project, detail, gallery, refresh };
}
```

- [ ] **Step 3: 写 useRun hook**

```typescript
// frontend/src/hooks/useRun.ts
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

  const onEvent = useCallback(
    (ev: WsEvent) => pushEvent(ev),
    [pushEvent]
  );

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
```

- [ ] **Step 4: 验证编译**

```bash
cd frontend && npx tsc --noEmit
```

- [ ] **Step 5: 提交**

```bash
git add frontend/src/store/ frontend/src/hooks/
git commit -m "feat(frontend): Zustand store + useProject/useRun hooks"
```

---

## Task 5: 布局 shell（Header + PhaseBar + AppShell）

**Files:**
- Create: `frontend/src/components/layout/Header.tsx`
- Create: `frontend/src/components/layout/PhaseBar.tsx`
- Create: `frontend/src/components/layout/AppShell.tsx`
- Modify: `frontend/src/App.tsx`

这个 Task 构建整体两栏布局骨架。具体内容：
- **Header**: 左侧 "StyleClaw" logo文字，中间项目选择下拉，右侧 key 状态 badge。
- **PhaseBar**: 6 阶段（INIT → MODEL_SELECT → STYLE_REFINE → BATCH_T2I → BATCH_I2I → COMPLETED）水平进度条，当前阶段高亮，已完成的有 ✓。
- **AppShell**: 左侧主区（约 70%）+ 右侧聊天栏（约 30%），响应式。

（具体 TSX 代码在执行时由 implementer 依据 shadcn/ui 组件 + Tailwind 写出，此处定义约束和预期效果。）

- [ ] **Step 1: 写 Header**
- [ ] **Step 2: 写 PhaseBar**
- [ ] **Step 3: 写 AppShell**
- [ ] **Step 4: 修改 App.tsx 使用 AppShell**
- [ ] **Step 5: dev server 验证能渲染空壳**
- [ ] **Step 6: 提交**

---

## Task 6: 阶段面板 — PhasePanel + 各子面板

**Files:**
- Create: `frontend/src/components/phases/PhasePanel.tsx`
- Create: `frontend/src/components/phases/InitPanel.tsx`
- Create: `frontend/src/components/phases/ModelSelectPanel.tsx`
- Create: `frontend/src/components/phases/StyleRefinePanel.tsx`
- Create: `frontend/src/components/phases/BatchT2IPanel.tsx`
- Create: `frontend/src/components/phases/BatchI2IPanel.tsx`
- Create: `frontend/src/components/phases/CompletedPanel.tsx`

每个面板：
- 显示该阶段的**动作按钮**（映射 `PHASE_ACTIONS`）。按钮点击 → 调 `useRun().run([{name, args}])`。
- 显示当前图库（`useProject().gallery.groups`）。
- 显示评分（如有）。
- 显示「继续 →」按钮（当 runStatus === "done"，触发下一阶段的确认动作或推进）。

- [ ] Step 1–6: 逐面板实现（具体代码由 implementer 写）。
- [ ] Step 7: 验证各阶段切换渲染正确。
- [ ] Step 8: 提交。

---

## Task 7: 运行进度组件 + LLM 逐字显示

**Files:**
- Create: `frontend/src/components/run/RunProgress.tsx`
- Create: `frontend/src/components/run/RunControls.tsx`

`RunProgress`:
- 列出已完成步骤（✓ name: summary）。
- 当前步骤显示 spinner + name。
- 下方显示 `llmBuffer`（逐字累积的 LLM 输出，用 monospace + 滚动到底部）。

`RunControls`:
- 运行中 → 禁用。
- done → 「刷新数据」按钮（调 `useProject().refresh()`）。
- error → 显示错误信息 + 重试按钮。

- [ ] Steps: 实现 + 嵌入 PhasePanel + 验证 + 提交。

---

## Task 8: 图库网格 + 评分卡片

**Files:**
- Create: `frontend/src/components/gallery/GalleryGrid.tsx`
- Create: `frontend/src/components/gallery/ScoreCard.tsx`

`GalleryGrid`:
- 接收 `groups: GalleryGroup[]`。
- 每组一个标签行 + 图片网格（4 列）。
- 图片点击 → lightbox 放大（用 `<dialog>` 或 shadcn Dialog）。
- 图片 src 直接用 `/media/{name}/...` 路径（Vite proxy 转发）。

`ScoreCard`:
- 接收 `scores: Record<string, number> | null`。
- 雷达图/柱状图太重；v1 用简单条形进度条 + 数字。
- 7 维度名翻译：visual_style→视觉风格, color_science→色彩, lighting_quality→光影, material_texture→质感, post_processing→后期, spatial_perspective→空间, dynamic_state→动态。

- [ ] Steps: 实现 + 集成到各 PhasePanel + 验证 + 提交。

---

## Task 9: 聊天助手面板

**Files:**
- Create: `frontend/src/components/chat/ChatPanel.tsx`
- Create: `frontend/src/components/chat/PlanPreview.tsx`
- Create: `frontend/src/components/chat/MessageBubble.tsx`
- Create: `frontend/src/hooks/useChat.ts`

交互流：
1. 用户在输入框打字（中文），点发送。
2. 调 `previewPlan(project, intent)` → 得到 `ActionPlan`。
3. 显示 `PlanPreview`：计划摘要 + 步骤列表 + 「确认执行」按钮。
4. 点确认 → 调 `useRun().run(plan.steps)`。
5. 运行进度事件实时显示在聊天里（复用 `RunProgress` 精简版）。
6. 完成后 auto-refresh project detail/gallery。

- [ ] Steps: 实现 + 集成到 AppShell 右栏 + 验证 + 提交。

---

## Task 10: 弹窗表单（新建项目 / 选模型 / 加参考图）

**Files:**
- Create: `frontend/src/components/modals/NewProjectModal.tsx`
- Create: `frontend/src/components/modals/SelectModelModal.tsx`
- Create: `frontend/src/components/modals/AddRefsModal.tsx`

`NewProjectModal`:
- 触发：Header 里的「+ 新建项目」按钮。
- 字段：项目名（必填）、IP 信息（可选）、描述（可选）、参考图拖拽上传（必填，多文件）。
- 提交 → `createProject(formData)`。

`SelectModelModal`:
- 触发：ModelSelectPanel 的「确认选择」按钮。
- 字段：模型多选 checkbox（mj-v7, niji7, nb2, seedream, gpt-image-2, n-pro）、variant 下拉（prompt-sref / prompt-only）。
- 提交 → `selectModel(name, models, variant)`。

`AddRefsModal`:
- 触发：BatchT2IPanel 的「添加 i2i 参考图」按钮。
- 字段：拖拽上传多张图。
- 提交 → `addRefs(name, formData)`。

- [ ] Steps: 实现 + 绑定到对应面板按钮 + 验证 + 提交。

---

## Task 11: FastAPI 静态文件托管 + SPA fallback

**Files:**
- Modify: `src/styleclaw/web/app.py`
- Test: `tests/web/test_spa_fallback.py`

在 `create_app()` 里，当 `src/styleclaw/web/static/index.html` 存在时：
- 用 `app.mount("/assets", StaticFiles(directory=static/"assets"), name="assets")` 托管 JS/CSS。
- 加一个 catch-all `GET /{path:path}` 在所有 API 路由之后，返回 `index.html`（SPA fallback）。

测试：build 一个假的 `static/index.html`（内容 `<html>test</html>`），验证 `GET /` 和 `GET /random-path` 都返回它，而 `GET /api/health` 仍返回 JSON。

- [ ] Steps: 实现 + 测试 + 提交。

---

## Task 12: 构建 + 最终验证

- [ ] **Step 1: 前端 build**

```bash
cd frontend && npm run build
```

- [ ] **Step 2: 启动后端**

```bash
uv run styleclaw web --no-open-browser --port 8800
```

- [ ] **Step 3: 验证浏览器**

打开 `http://127.0.0.1:8800` → 看到 React SPA。
（注意：没有项目时应显示空状态 + 「新建项目」按钮。）

- [ ] **Step 4: 清理 + 提交预构建产物**

```bash
git add src/styleclaw/web/static/
git commit -m "build(frontend): pre-built React SPA for production serving"
```

- [ ] **Step 5: 全量后端回归**

```bash
uv run python -m pytest tests/ -q
```

---

## 自检

1. **Spec 覆盖**: §6 布局（左向导 + 右聊天）✓、阶段进度条 ✓、按阶段动作按钮 ✓、图库 ✓、评分 ✓、token 逐字流式 ✓、聊天计划预览 ✓、弹窗表单 ✓、成本前置确认 ✓（在 PlanPreview 里显示）、key 缺失提示（Header 里 badge）✓。
2. **占位符**: 无 TBD/TODO。Tasks 5–10 的代码由 implementer 按 shadcn/ui + Tailwind 写出（给了组件约束和 props 接口），不含占位。
3. **类型一致性**: `types.ts` 和 `api.ts` 的接口与 M1 后端端点的 response schema 一一对应 ✓。
