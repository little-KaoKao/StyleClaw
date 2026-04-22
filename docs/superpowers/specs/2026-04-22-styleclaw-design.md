# StyleClaw — 风格触发词探索 Agent 系统设计文档

## 概述

StyleClaw 是一个半自动化的 AI 图像风格触发词探索系统。给定参考图和画风信息，系统通过 AI 分析画风、生成触发词、调用图像生成 API 出图、对比评分、迭代优化的闭环，帮助用户找到最佳的风格触发词，并批量验证其泛化能力。

## 核心决策

| 决策项 | 选择 | 理由 |
|---|---|---|
| 技术栈 | Python + uv | boto3 原生支持 Bedrock，图片处理生态丰富 |
| LLM | Claude via AWS Bedrock（可插拔，未来支持 Gemini） | 用户现有环境 |
| 图像生成 | RunningHub 标准模型 API | 简洁直接，4 个目标模型均有端点 |
| 交互模式 | CLI 脚本 + Claude Code 混合 | 机械工作用脚本省 token，创造性工作用 Agent |
| 持久化 | 文件系统 + JSON | 可追溯、可回滚、可手动编辑 |
| 架构模式 | 状态机管道 | 流程线性 + 局部循环，最自然的建模 |
| 相似度判定 | AI 自动迭代 + 人工拍板 | 兼顾效率和质量把控 |

## 支持的模型

| 模型 ID | 名称 | 文生图端点 | 图生图端点 | prompt 上限 | 特有参数 |
|---|---|---|---|---|---|
| `mj-v7` | 悠船文生图 v7 (Midjourney) | `/openapi/v2/youchuan/text-to-image-v7` | 同端点 `imageUrl` + `iw` | 8192 | sref, sw, stylize, chaos, weird, raw |
| `niji7` | 悠船文生图 niji7 (Midjourney) | `/openapi/v2/youchuan/text-to-image-niji7` | 同端点 `imageUrl` + `iw` | 8192 | sref, sw, stylize, chaos, weird, raw |
| `nb2` | 全能图片V2 (NanoBanana2) | `/openapi/v2/rhart-image-n-g31-flash-official/text-to-image` | `/openapi/v2/rhart-image-n-g31-flash-official/image-to-image` | 20000 | resolution(1k/2k/4k) |
| `seedream` | Seedream v5-lite (即梦) | `/openapi/v2/seedream-v5-lite/text-to-image` | `/openapi/v2/seedream-v5-lite/image-to-image` | 2000 | resolution(2k/3k), maxImages |

通用 API：
- 文件上传: `POST /openapi/v2/media/upload/binary` (multipart/form-data)
- 任务状态: `POST /task/openapi/status` (已弃用但可用)
- 任务结果: `POST /task/openapi/outputs`
- 认证: `Authorization: Bearer <API_KEY>`

## 状态机与流程

### 5 个阶段（Phase）

```
INIT ──> MODEL_SELECT ──> STYLE_REFINE ──> BATCH_T2I ──> BATCH_I2I ──> COMPLETED
                               ▲    │         │              │
                               └────┘         │              │
                               ▲              │              │
                               └──────────────┘              │
                               ▲                             │
                               └─────────────────────────────┘
```

### 状态转换规则

```python
TRANSITIONS = {
    "INIT":          ["MODEL_SELECT"],
    "MODEL_SELECT":  ["STYLE_REFINE"],
    "STYLE_REFINE":  ["BATCH_T2I", "STYLE_REFINE"],
    "BATCH_T2I":     ["BATCH_I2I", "STYLE_REFINE"],
    "BATCH_I2I":     ["STYLE_REFINE", "COMPLETED"],
}
```

任何阶段可通过 `rollback` 命令回退到之前的任意阶段，历史数据保留不删除。

### Phase 1: INIT

- 输入：参考图（1 张或多张）、IP/画风信息描述、项目名称
- 动作：上传参考图到 RunningHub 获取 URL，创建项目目录结构，写入 config.json
- 执行者：CLI 脚本

### Phase 2: MODEL_SELECT

- 动作：
  1. **Agent** 分析参考图画风特征，输出初始触发词
  2. **脚本** 用同一触发词并行调用 4 个模型，每模型生成 2-3 张
  3. **脚本** 轮询任务状态、下载结果图
  4. **Agent** 对比 4 个模型的结果与参考图，评分 + 推荐
  5. **人工** 确认选定 1-2 个模型
- 执行者：Agent + 脚本 + 人工

### Phase 3: STYLE_REFINE（核心循环）

每轮（Round）：
1. **Agent** 基于上一轮结果（或初始分析），调整触发词
2. **脚本** 用新触发词调用选定模型生成图片（每轮每模型 3 张）
3. **脚本** 轮询 + 下载
4. **Agent** 对比结果图与参考图，5 维度打分（色彩、线条、光影、质感、氛围，各 1-10）
5. 自动判定：
   - 所有维度 ≥ 7 且总分 ≥ 7.5 → 暂停等人工审核
   - 有维度 < 5 → 暂停等人工介入
   - 其余 → 继续下一轮
6. 最大自动迭代 5 轮，超过强制暂停

人工可选操作：确认通过 / 给调整方向继续 / 回退到某轮

### Phase 4: BATCH_T2I

1. **Agent** 根据 IP 设定设计 100 条角色/场景描述（10 类 × 10 条）
2. **人工** 审核修改 cases.json
3. **脚本** 拼接触发词 + 角色描述，批量提交
4. **脚本** 轮询 + 下载 + 生成 HTML 报告
5. **人工** 审核，确认通过或给修改方向（回退 Phase 3 或在 Phase 4 内修改重跑）

画幅规则：
- 室外场景、室内场景、多人合照 → 16:9
- 其余（人物类） → 9:16

### Phase 5: BATCH_I2I

1. **人工** 提供一批参考图
2. **脚本** 上传参考图 + 批量调用图生图端点
3. **脚本** 轮询 + 下载 + 生成对比报告（原图 vs 生成图并排）
4. **人工** 审核，确认通过或回退 Phase 3

## 数据存储结构

```
data/projects/<project-name>/
├── config.json                          # 项目配置（创建时写入，不可变）
├── state.json                           # 运行时状态
├── refs/                                # 参考图
│   ├── ref-001.png
│   └── uploads.json                     # 本地文件名 → RunningHub URL
├── model-select/                        # Phase 2
│   ├── initial-analysis.json            # 画风分析 + 初始触发词
│   ├── results/
│   │   └── <model-id>/
│   │       ├── task.json                # API 请求/响应完整记录
│   │       └── output-*.png
│   └── evaluation.json                  # 模型评分 + 推荐
├── style-refine/                        # Phase 3
│   └── round-NNN/
│       ├── prompt.json                  # 触发词 + 模型参数 + 来源 + 修改说明
│       ├── results/
│       │   └── <model-id>/
│       │       ├── task.json
│       │       └── output-*.png
│       └── evaluation.json              # 5维评分 + 建议 + 达标判定
├── batch-t2i/                           # Phase 4
│   └── batch-NNN/
│       ├── cases.json                   # 100 条测试用例
│       ├── results/
│       │   └── case-NNN/
│       │       ├── task.json
│       │       └── output-*.png
│       └── report.html
└── batch-i2i/                           # Phase 5
    └── batch-NNN/
        ├── source-images/
        ├── uploads.json
        ├── cases.json
        ├── results/
        └── report.html
```

### 核心 JSON Schema

**config.json**
```json
{
  "name": "string",
  "created_at": "ISO8601",
  "description": "string",
  "ip_info": "string (IP/画风详细描述)",
  "ref_images": ["refs/ref-001.png", "..."]
}
```

**state.json**
```json
{
  "phase": "INIT | MODEL_SELECT | STYLE_REFINE | BATCH_T2I | BATCH_I2I | COMPLETED",
  "selected_models": ["model-id", "..."],
  "current_round": 0,
  "current_batch": 0,
  "last_updated": "ISO8601",
  "history": [
    {"phase": "...", "completed_at": "...", "metadata": {}}
  ]
}
```

**prompt.json**（每轮触发词）
```json
{
  "round": 1,
  "trigger_phrase": "string (通用触发词)",
  "model_params": {
    "<model-id>": {
      "prompt_template": "{trigger_phrase}, {character_desc}",
      "stylize": 200,
      "aspectRatio": "9:16"
    }
  },
  "derived_from": "round-NNN | initial-analysis",
  "adjustment_note": "string (修改说明)"
}
```

**evaluation.json**（评分记录）
```json
{
  "round": 1,
  "evaluations": [
    {
      "model": "model-id",
      "image": "output-001.png",
      "scores": {
        "color_palette": 8,
        "line_style": 7,
        "lighting": 8,
        "texture": 6,
        "overall_mood": 8
      },
      "total": 7.4,
      "analysis": "string",
      "suggestions": "string"
    }
  ],
  "recommendation": "approve | continue_refine | needs_human",
  "next_direction": "string"
}
```

**cases.json**（批测用例）
```json
{
  "batch": 1,
  "trigger_phrase": "string",
  "cases": [
    {
      "id": "case-001",
      "category": "adult_male | adult_female | shota | loli | elderly_male | elderly_female | creature | outdoor_scene | indoor_scene | group",
      "description": "string (角色/场景描述)",
      "aspect_ratio": "9:16 | 16:9",
      "status": "pending | submitted | completed | failed"
    }
  ]
}
```

## 模块架构

```
styleclaw/
├── src/styleclaw/
│   ├── core/                    # 核心领域层（纯逻辑，无 IO）
│   │   ├── models.py                # Pydantic 数据模型
│   │   ├── state_machine.py         # 状态机转换 + 回滚逻辑
│   │   ├── prompt_builder.py        # 触发词拼接 + 模型参数适配
│   │   └── case_generator.py        # 用例骨架生成（类别 × 画幅）
│   │
│   ├── storage/                 # 持久化层
│   │   ├── project_store.py         # 项目目录 CRUD
│   │   └── image_store.py           # 图片下载 + 本地存储
│   │
│   ├── providers/               # 外部服务适配器（可插拔）
│   │   ├── base.py                  # 抽象接口
│   │   ├── runninghub/
│   │   │   ├── client.py            # HTTP 客户端（认证、重试、限流）
│   │   │   ├── models.py            # 4 模型端点配置 + 参数映射
│   │   │   ├── tasks.py             # 提交/轮询/取结果
│   │   │   └── upload.py            # 文件上传
│   │   └── llm/
│   │       ├── base.py              # LLM Provider 抽象接口
│   │       ├── bedrock.py           # AWS Bedrock Claude 实现
│   │       └── prompts/             # Agent prompt 模板
│   │           ├── analyze.md
│   │           ├── select_model.md
│   │           ├── refine.md
│   │           ├── evaluate.md
│   │           └── design_cases.md
│   │
│   ├── scripts/                 # CLI 脚本（不调 LLM，不烧 token）
│   │   ├── init_project.py
│   │   ├── generate.py              # 批量提交生成任务
│   │   ├── poll.py                  # 轮询 + 下载
│   │   ├── batch_submit.py          # 批测提交
│   │   └── report.py               # HTML 报告生成
│   │
│   ├── agents/                  # Agent 入口（调 LLM，消耗 token）
│   │   ├── analyze_style.py         # 分析参考图 → 初始触发词
│   │   ├── select_model.py          # 对比 4 模型 → 推荐
│   │   ├── refine_prompt.py         # 优化触发词
│   │   ├── evaluate_result.py       # 对比评分
│   │   └── design_cases.py          # 设计 100 条用例
│   │
│   ├── reports/                 # 报告生成
│   │   ├── templates/
│   │   │   ├── model_select.html
│   │   │   ├── style_refine.html
│   │   │   ├── batch_t2i.html
│   │   │   └── batch_i2i.html
│   │   └── renderer.py
│   │
│   └── cli.py                   # 统一 CLI 入口
│
├── data/                        # 运行时数据（.gitignore）
├── tests/
├── pyproject.toml
├── .env.example
└── CLAUDE.md
```

### 模块职责

**core/**（纯逻辑，无副作用）

| 模块 | 职责 |
|---|---|
| `models.py` | Pydantic 模型：Project, Round, BatchCase, Evaluation, ModelConfig。所有 JSON 读写走这里 |
| `state_machine.py` | 阶段转换验证、回滚逻辑 |
| `prompt_builder.py` | trigger_phrase + character_desc + model params → 最终 prompt。处理 Seedream 2000 字符截断 |
| `case_generator.py` | 100 条用例的骨架（10 类 × 10 条 + 画幅映射） |

**providers/runninghub/**

| 模块 | 职责 |
|---|---|
| `client.py` | httpx.AsyncClient，统一认证、指数退避重试（3 次）、并发信号量（5）|
| `models.py` | MODEL_REGISTRY 配置：端点、prompt 长度限制、比例参数格式、特有参数 |
| `tasks.py` | submit_task → poll_status → get_results 三步流程 |
| `upload.py` | multipart/form-data 文件上传，返回 download_url |

**providers/llm/**

| 模块 | 职责 |
|---|---|
| `base.py` | `LLMProvider` Protocol：`invoke(system, messages, max_tokens, temperature) -> str` |
| `bedrock.py` | boto3 Bedrock 实现，env: AWS_REGION, AWS_BEARER_TOKEN_BEDROCK |

**scripts/**（机械工作，独立运行）

| 脚本 | 输入 | 输出 |
|---|---|---|
| `init_project.py` | 项目名、参考图路径、IP 描述 | config.json + refs/ |
| `generate.py` | 项目名 + prompt.json | task.json（含 taskId）|
| `poll.py` | 项目名 | 更新 task.json + 下载图片 |
| `batch_submit.py` | 项目名 + cases.json | 每 case 的 task.json |
| `report.py` | 项目名 + 阶段 | report.html |

**agents/**（消耗 token，创造性工作）

| Agent | 输入 | 输出 |
|---|---|---|
| `analyze_style.py` | 参考图 + IP 信息 | initial-analysis.json |
| `select_model.py` | 参考图 + 4 模型生成图 | evaluation.json（含推荐）|
| `refine_prompt.py` | 参考图 + 上轮结果 + 历史评分 + 人工方向 | prompt.json |
| `evaluate_result.py` | 参考图 + 本轮生成图 | evaluation.json |
| `design_cases.py` | IP 信息 + 触发词 + 用例骨架 | cases.json |

## RunningHub API 适配

### 模型参数差异处理

| 差异项 | MJ-v7 / Niji7 | NB2 | Seedream |
|---|---|---|---|
| prompt 长度 | 8192 | 20000 | 2000（自动压缩） |
| 比例参数 | `aspectRatio: "9:16"` | `aspectRatio: "9:16"` | `width` × `height` 或 `resolution` |
| 图生图方式 | 同端点 `imageUrl` + `iw` | 独立端点 `imageUrls[]` | 独立端点 `imageUrls[]` |
| 风格参考 | `sref` + `sw`（独有） | 无 | 无 |

Seedream 2000 字符截断策略：保留核心风格词 + 角色描述，去掉冗余修饰。超限则记录警告并截断。

MJ sref 利用：在 MODEL_SELECT 和 STYLE_REFINE 阶段，MJ 系列可通过 `sref` 传入参考图作为风格锚点。

### 并发与轮询

- 并发上限：5 个同时请求（信号量控制）
- 轮询间隔：3 秒
- 单任务超时：300 秒
- 失败重试：3 次，指数退避
- `poll` 命令幂等：已完成的跳过，只处理 pending

### 成本估算

| 阶段 | 图片数量 |
|---|---|
| MODEL_SELECT | ~12 张（4 模型 × 3 张）|
| STYLE_REFINE（5 轮）| ~30 张（每轮 3 张 × 2 模型）|
| BATCH_T2I | 100 张 |
| BATCH_I2I | N 张（取决于参考图数量）|
| **单次完整探索** | **~150 张** |

## Agent 提示词策略

### 通用原则

- 每个 Agent 是一次性调用（非多轮对话），输入输出明确
- 不使用 Agent 框架，直接 boto3 → Bedrock API
- 图片通过多模态 API base64 传入
- 输出约束为 JSON schema，减少解析错误

### 5 个 Agent 的 Prompt 要点

**analyze_style**: 6 维度分析（色彩、线条、光影、质感、构图、氛围），输出触发词 + 变体 + 模型建议

**select_model**: 5 维度评分（1-10），输出每模型评分 + 推荐排序

**refine_prompt**: 输入历史评分趋势，约束每轮修改幅度 ≤ 30%，保留高分维度词汇，聚焦低分维度

**evaluate_result**: 5 维度评分 + 达标判定（approve / continue_refine / needs_human）

**design_cases**: 10 类 × 10 条，每条 50-150 字英文描述，确保类内差异性

### Token 节约

- 图片压缩到 1024px 长边再传 LLM
- refine agent 只传最近 3 轮完整评分，更早的只传分数
- 模板填充、拼接、格式化全在 Python 做

## CLI 命令

```bash
# 项目管理
styleclaw init <name> --ref <image>... --info "画风描述"
styleclaw status [<name>]
styleclaw rollback <name> --to <phase> [--round <n>]

# Phase 2: 模型选择
styleclaw analyze <name>
styleclaw generate <name> --phase model-select
styleclaw poll <name>
styleclaw evaluate <name> --phase model-select
styleclaw select-model <name> --models mj-v7,niji7

# Phase 3: 风格优化
styleclaw refine <name>
styleclaw generate <name>
styleclaw poll <name>
styleclaw evaluate <name>
styleclaw approve <name>
styleclaw adjust <name> --direction "线条再柔和"

# Phase 4: 文生图批测
styleclaw design-cases <name>
styleclaw batch-submit <name>
styleclaw poll <name>
styleclaw report <name>

# Phase 5: 图生图批测
styleclaw add-refs <name> --images <image>...
styleclaw batch-submit <name> --i2i
styleclaw poll <name>
styleclaw report <name> --i2i
```

## 环境配置

```bash
# .env.example
RUNNINGHUB_API_KEY=<your-runninghub-api-key>
AWS_REGION=us-east-1
AWS_BEARER_TOKEN_BEDROCK=<your-bedrock-token>
CLAUDE_MODEL=anthropic.claude-sonnet-4-20250514  # 可配置模型
```

注意：标准模型 API 的响应可能直接包含 results（同步完成），也可能返回 QUEUED/RUNNING 状态需要轮询。客户端统一按"提交 → 检查响应 → 如未完成则轮询"处理。

## 依赖

```
httpx          # HTTP 客户端
boto3          # AWS Bedrock
pydantic       # 数据模型 + JSON schema
typer          # CLI 框架
jinja2         # HTML 报告模板
pillow         # 图片压缩
python-dotenv  # 环境变量
```
