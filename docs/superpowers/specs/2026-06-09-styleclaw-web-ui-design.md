# StyleClaw 本地交互版（Web UI）设计

**日期**: 2026-06-09
**状态**: 已确认设计，待写实现计划
**目标读者**: 让不懂代码的人也能用 StyleClaw 走完整条风格触发词探索流程。

---

## 1. 目标与范围

### 1.1 问题

StyleClaw 现在只有 Typer CLI。非程序员无法使用：要记命令、面对黑窗口、看不到图。
但项目已经有两块现成资产，让「交互版」主要是一层表现层而非重写：

- **orchestrator**：把自由中文/英文意图变成 `ActionPlan` 并逐步执行（`styleclaw run "..."`）。
- **HTML 报告**：每个阶段的图片可视化（`reports/templates/*.html`）。

### 1.2 形态（已确认）

**本地单用户 Web 应用**。在跑 StyleClaw 的这台机器上，用一条命令启动一个本地服务，浏览器自动打开 `http://127.0.0.1:8800`，非程序员在网页里点击操作。

- 绑定 `127.0.0.1`，**单用户、无鉴权**。
- **同一时刻每个项目只有一个运行**（复用 `project_store.project_lock`）。
- 这把并发/安全/多租户复杂度全部砍掉。

### 1.3 交互范式（已确认）

**向导 + 聊天 混合**：

- 左主区 = 按阶段变化的向导（大按钮 + 实时进度 + 图库 + 评分 + 阶段边界「继续」）。
- 右栏 = 聊天助手，复用 `planner.plan` 把中文意图变 `ActionPlan`，**先预览再执行**。

### 1.4 自动化程度（已确认）

**默认阶段级暂停**：一个阶段内部的链（如 refine→generate→poll→evaluate）自动跑完，到阶段边界停下让人看结果、点「继续」才跨阶段。也保留「每步手动」的操作方式（单动作按钮）。

### 1.5 技术栈（已确认）

**FastAPI 后端 + React SPA 前端**。前端**预构建**为静态文件，由 FastAPI 直接托管；终端用户**永不接触 npm**。

### 1.6 v1 范围内

- 浏览器内**完整生命周期**：新建项目（拖拽参考图 + 填 IP 信息/描述）→ 走完 5 个阶段 → 看图库 → 看评分 → 阶段级暂停 →「继续」。
- 聊天助手：自然语言 → planner → 计划预览 → 执行。
- **token 级流式打字效果**（LLM 增量逐字显示，单模型路径）。
- 批量生成前的**成本前置确认**（复用 `orchestrator.cost_estimate`）。
- 断线/刷新后**重连重放**进度。
- API key 缺失时的友好提示页。

### 1.7 明确不做（v2+）

- 多用户 / 账号 / 鉴权。
- 云端部署。
- 在网页里编辑 `.env`（v1 只读显示 key 状态）。
- panel/并发 LLM 调用的逐字流式（回退到步骤级进度，避免多模型 token 交错）。

---

## 2. 架构总览

```
浏览器 (React SPA, 预构建静态文件)
   │  REST  : 项目 CRUD / 单动作 / 计划预览 / 确认动作表单
   │  WS    : 实时事件（step / llm_delta / needs_human / done / error）
   ▼
FastAPI  (src/styleclaw/web/)
   │  复用 ExecutionContext (RunningHubClient + LLMProvider + RoleRouter)
   ▼
现有层：orchestrator.actions / planner / executor / cost_estimate
        / storage.project_store / scripts.report
   ▼
DATA_ROOT 下的 JSON + 图片（存储格式完全不变）
```

**复用而非重写**：所有业务逻辑仍在 `ACTION_REGISTRY` / `planner` / `executor` 里。Web 层只做三件事：HTTP/WS 接口、运行生命周期管理、把 orchestrator 钩子转成事件。

---

## 3. 后端模块拆分

新增目录 `src/styleclaw/web/`，每个文件单一职责：

| 文件 | 职责 | 依赖 |
|------|------|------|
| `app.py` | FastAPI 工厂：装配路由、挂载静态前端、用 lifespan 管理 `ExecutionContext`（复用 `cli._build_context` / `_close_resource` 逻辑，抽成可共享函数） | 下面所有 routes + run_manager |
| `context.py` | 构建/释放 `ExecutionContext`（client + llm + router）。从 `cli.py` 抽出共享逻辑，CLI 与 Web 共用 | `cli` 现有构建逻辑 |
| `routes_projects.py` | 项目读/建/查询/图库 端点 | project_store, report 数据函数 |
| `routes_actions.py` | 单动作执行、计划预览、运行启动、确认动作端点 | actions, planner, run_manager |
| `run_manager.py` | **核心**：单项目运行生命周期 + 事件总线。把 `execute()` 钩子转 WS 事件；持有当前运行句柄；加锁防并发；保存事件供重连重放 | executor, events |
| `events.py` | Pydantic 事件 schema（前后端契约） | — |
| `delta_sink.py` | `contextvars.ContextVar` 持有可选的 LLM 增量回调；供 token 流式使用 | — |
| `launch.py` | 起 uvicorn + 自动开浏览器，供 CLI `styleclaw web` 调用 | uvicorn |

### 3.1 确认动作的处理（绕开同步钩子阻塞）

`executor.execute()` 的 `on_confirm` 是**同步**回调（`ConfirmCallback = Callable[[str, dict, ExecutionContext], dict | None]`），无法在 async 循环里阻塞等浏览器。

**方案**：三个需要确认的动作（`init` / `select-model` / `add-refs`）各有**专属 REST 端点**，参数由 React 表单一次给齐，**不进自动循环**。`execute()` 的自动链里只含无需确认的动作（`refine` / `generate` / `poll` / `evaluate` / `approve` / `design-cases` / `batch-submit` / `report`）。

这同时匹配「阶段边界由人显式操作」的 UX：新建项目、选模型、加 i2i 参考图本就是人主动做的阶段切换动作。

---

## 4. HTTP / WS 接口契约

### 4.1 REST 端点

| 方法 & 路径 | 作用 | 备注 |
|------|------|------|
| `GET /api/projects` | 列出所有项目（name, phase, round/batch, last_updated） | `project_store.list_projects` + `load_state` |
| `POST /api/projects` | 新建项目（multipart: `files[]` 参考图 + `name` + `ip_info` + `description` + `force`） | 走 `init` action（确认动作端点） |
| `GET /api/projects/{name}` | 项目详情：`{state, config, phase_hints, suggestions, env_ok}` | suggestions 复用 `suggest_next_steps` |
| `GET /api/projects/{name}/gallery` | 当前（或指定）阶段图库 JSON：参考图、各结果组、评分 | query: `phase/pass/round/batch`，缺省取 state 当前值 |
| `POST /api/projects/{name}/actions/{action}` | 执行**单个**无需确认动作，JSON 传 args | 长动作经 run_manager 后台跑，返回 `run_id` |
| `POST /api/projects/{name}/plan` | `{intent}` → `ActionPlan` 预览（不执行） | `planner.plan` |
| `POST /api/projects/{name}/run` | 启动后台运行：`{plan}` 或 `{phase}`（阶段自动链） | 返回 `run_id`，立即返回 |
| `GET /api/projects/{name}/runs/{run_id}` | 运行状态 + 最近事件（**重连重放**用） | run_manager 保存的事件缓冲 |
| `POST /api/projects/{name}/select-model` | `{models, variant}` 推进 MODEL_SELECT→STYLE_REFINE | 确认动作端点 |
| `POST /api/projects/{name}/refs` | multipart 上传 i2i 参考图 → `add-refs` | 确认动作端点 |
| `WS /api/projects/{name}/events` | 实时事件流（见 4.2） | 连上后先收一次当前运行快照 |
| `GET /media/{name}/...` | 托管 `DATA_ROOT/{name}` 下的图片 | 路径校验，禁止越界（复用项目名校验） |
| `GET /` + `/assets/*` | 托管预构建 React 静态文件 | — |

### 4.2 WebSocket 事件 schema（`events.py`）

所有事件是带 `type` 字段的 Pydantic 模型，JSON 序列化：

- `run_started` `{run_id, project, kind: "plan"|"phase", steps: [...]}`
- `step_start` `{index, name, description}`
- `llm_delta` `{step_index, role, text}` — **token 流式**；仅单模型路径发，panel/并发不发
- `step_done` `{index, name, status, summary}`
- `needs_human` `{round, weakest_dim, score, suggestion}` — 复用 `_should_continue_loop` 的判定
- `phase_paused` `{phase, next_phase}` — 阶段链跑完，等人点「继续」
- `done` `{run_id}`
- `error` `{message, detail}`

确认流程 v1 **不走 WS**（用 4.1 的表单端点在运行前给齐参数）。

---

## 5. 数据流与实时进度

### 5.1 启动 → 进度 → 暂停

1. React `POST .../run`（带 `phase` 或 `plan`）→ `run_manager` 在后台 asyncio task 里跑 `execute()`，立即返回 `run_id`。
2. React 连 `WS .../events`，按 4.2 收事件渲染进度。
3. **阶段级暂停**：阶段自动链跑完发 `phase_paused`，UI 显示图库+评分+「继续 →」，**不自动跨阶段**。
4. **needs_human**：循环里复用 `_should_continue_loop`（读盘上最新 `RoundEvaluation`，看 `needs_human()` / `should_approve()`），UI 高亮最弱维度并给一句建议方向。

### 5.2 token 流式打字效果

- `delta_sink.py` 暴露一个 `ContextVar[Callable[[str], None] | None]`。
- `run_manager` 在运行期间把 sink 指向「向当前 WS 推 `llm_delta`」的回调，并随 `on_step_start` 更新当前 step_index。
- `providers/llm/openai_compat.py` 的流式循环：原 `print()` 处改为——**若 sink 已设则调用 sink(delta)，否则保持原 `print()`（受 `STREAM_DISPLAY` 控制）**。CLI 行为完全不变。
- **并发/panel 限制**：当一次运行触发并发 LLM 调用（`LLM_CONCURRENCY` 或 panel 三模型）时，逐字会交错。v1 在这些路径**不设 sink**（回退步骤级进度），只在常见单模型路径开逐字。需在 spec 实现时确认 `invoke` 与 `invoke_with_thinking` 两条流式路径都接 sink。

### 5.3 断线重连（poll 可能跑 ~30 分钟）

- 运行在**服务端后台 task 里持续**，与 WS 连接无关。关 tab / 刷新**不中断**运行。
- `run_manager` 为每个运行保存一个**有上限的事件缓冲**。
- 浏览器重连：先 `GET .../runs/{run_id}` 取状态 + 重放最近事件补齐 UI，再连 WS 接续后续事件。

### 5.4 成本前置

跑 `batch-submit` 等高成本动作前，用 `orchestrator.cost_estimate` 估任务数，UI 弹「将提交 N 个任务，约 M 张图」确认框。

### 5.5 API key 缺失

启动时若 `config.validate_env()` 不过，首页显示**友好提示页**（指引去填 `.env`），而非崩溃。只读显示当前 key 状态（不在网页里编辑 `.env`）。

---

## 6. 前端布局（React SPA）

```
┌─────────────────────────────────────────────────────────┐
│  StyleClaw   [项目: spiderverse ▼]          key: ✓ 已配置 │
├─────────────────────────────────────────────────────────┤
│  ① INIT ─ ②模型筛选 ─ ③风格精炼 ─ ④批量T2I ─ ⑤批量I2I ─✓ │  ← 阶段进度条(当前高亮)
├──────────────────────────────────────┬──────────────────┤
│  当前阶段面板                          │  💬 助手          │
│  ┌────────────────────────────────┐  │  ┌────────────┐  │
│  │ [分析风格] [生成图片] [评分]    │  │  │ 你: 帮我... │  │
│  └────────────────────────────────┘  │  │ 计划预览:   │  │
│  ▶ 运行进度: generate ●●●○ 轮询中…    │  │  1.refine   │  │
│    ↓ LLM 正在分析…(逐字流式)          │  │  2.generate │  │
│                                       │  │ [确认执行]  │  │
│  图库:                                │  └────────────┘  │
│  [img][img][img][img]  评分: 色彩8.0  │  自由指令也从这里 │
│  [img][img][img][img]       线条7.5   │                  │
│                          [继续 → ]    │                  │
└──────────────────────────────────────┴──────────────────┘
```

- 阶段进度条映射状态机 `INIT→MODEL_SELECT→STYLE_REFINE→BATCH_T2I→BATCH_I2I→COMPLETED`，当前阶段高亮。
- 主区按阶段渲染不同的动作按钮 + 进度 + 图库 + 评分。
- 右栏聊天：意图→`planner.plan`→计划预览→`POST .../run` 执行。
- 三个确认动作 = 弹窗表单（新建项目 / 选模型 / 加 i2i 参考图）。
- 前端只做关键组件冒烟测试（vitest），逻辑重心在后端。

---

## 7. 复用与小幅改造

- **图库数据**：把 `scripts/report.py` 里四个 `generate_*_report` 中**收集图片/评分的循环**抽成可复用的纯数据函数（返回结构化 dict），供 React 图库端点与现有 HTML 报告共用。已确认底层 `image_store.list_output_images()` 与 `project_store.*_results_dir()` 等存在。
- **ExecutionContext 构建**：从 `cli._build_context` / `_close_resource` 抽出共享构建/释放逻辑到 `web/context.py`（或公共模块），CLI 与 Web 共用，避免重复。
- **delta sink**：在 `openai_compat.py` 流式循环加 sink 出口（5.2），CLI 默认行为不变。
- **新 CLI 命令** `styleclaw web [--port 8800] [--no-open]`：启动服务并开浏览器。
- 不做无关重构，仅限服务本目标的改动。

---

## 8. 错误处理

- REST 错误返回结构化 JSON `{error, detail}`，前端 toast/区域展示。
- 运行内异常 → WS `error` 事件，UI 显示并给重试按钮（`poll` 本身已自动重试 FAILED 一次）。
- 长轮询沿用现有超时（`STYLECLAW_TASK_TIMEOUT` / `MAX_POLL_CYCLES`）。
- key 缺失 → 友好提示页（5.5）。
- 并发保护：第二个运行请求命中 `project_lock` → 返回「该项目正在运行中」。

---

## 9. 测试策略

- **后端（重心）**：starlette `TestClient` / httpx AsyncClient + monkeypatch `DATA_ROOT` + stub `RunningHubClient` / `LLMProvider`（沿用现有测试夹具）。覆盖：
  - 项目列表/新建/详情/图库端点
  - 单动作路由与参数校验
  - `plan` 端点（NL→ActionPlan）
  - `run` 启动 + WS 事件序列（用假 orchestrator 步骤断言 `step_start`/`step_done`/`phase_paused`/`done`）
  - 确认动作端点（init/select-model/add-refs）
  - delta sink：断言 sink 设置时 delta 进事件、未设时回退 print
  - 重连：`runs/{run_id}` 事件重放
- **前端**：vitest 关键组件冒烟（阶段进度条、图库、聊天计划预览）。
- 维持仓库 `fail_under=80%` 覆盖率门槛（后端新代码需带测试）。

---

## 10. 里程碑（实现顺序）

1. **M1 — 纯后端 + 完整事件流（含 token 流式）**：无 React 也能用 `curl` / `websocat` 跑通项目 CRUD、单动作、plan、run、WS 事件、delta 流。**所有逻辑与风险集中在此层，pytest 火力集中于此。** 可作为可演示检查点。
2. **M2 — React 前端**：事件流通了之后，UI 是常规活。向导 + 聊天 + 图库 + 弹窗表单。
3. **M3 — 打包 + 双击启动**：预构建前端进 `web/static/`；`styleclaw web` 命令；可选 `.command`/`.bat` 双击脚本给非程序员。

---

## 11. 关键决策小结

| 决策 | 取舍理由 |
|------|---------|
| 本地单用户 + 127.0.0.1 + 无鉴权 | 砍掉并发/安全复杂度 |
| 复用 ACTION_REGISTRY/planner/executor | 业务逻辑零重写，Web 只做表现层 |
| 确认动作走专属表单端点、不进自动循环 | 绕开同步 `on_confirm` 在 async 里阻塞的问题 |
| 运行在服务端后台 task + 事件重放 | 关 tab / 30 分钟长轮询不中断 |
| token 流式用 ContextVar sink | 不改 LLM 调用签名，CLI 行为不变 |
| panel/并发不做逐字流式 | 避免多模型 token 交错糊成一团 |
| M1 先交付可验证纯后端 | 风险与逻辑在后端，先拿到可演示检查点 |
