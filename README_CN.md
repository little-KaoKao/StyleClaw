# StyleClaw

AI 驱动的风格触发词探索系统，用于图像生成。

给定一组代表某 IP 视觉风格的参考图片，StyleClaw 通过 LLM 分析 + 批量图像生成，迭代发现并验证一个简洁的**触发短语（trigger phrase）**，使其能在多样化的主题下可靠地复现该风格。

## 工作原理

```
参考图片 ──▶ LLM 风格分析 ──▶ 模型选择 ──▶ 迭代精炼 ──▶ 100 用例验证
```

StyleClaw 采用状态机驱动的流水线：

```
INIT → MODEL_SELECT → STYLE_REFINE → BATCH_T2I → BATCH_I2I → COMPLETED
```

1. **INIT** — 提供参考图片，LLM 提取风格维度和初始触发短语
2. **MODEL_SELECT** — 用多个模型生成测试图，LLM 评选最优模型
3. **STYLE_REFINE** — 迭代精炼触发短语（最多 5 轮，按 5 个维度评分）
4. **BATCH_T2I** — 100 个多样化用例的泛化验证（10 个类别 × 10 个）
5. **BATCH_I2I** — 图生图测试，进一步验证风格迁移能力

---

## 环境要求

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** 包管理器
- **RunningHub** API 密钥（用于图像生成）
- **LLM 提供方** — 三选一（同时配置时优先级见下）：
  - **OpenAI 兼容** 服务，如 [gptproto.com](https://gptproto.com)（推荐），或
  - **RunningHub LLM**（`https://llm.runninghub.cn/v1`，与图像 API 共用 `RUNNINGHUB_API_KEY`），或
  - **AWS Bedrock** 访问权限 + Bearer Token（旧方案）

---

## 安装

```bash
git clone https://github.com/little-KaoKao/StyleClaw.git
cd StyleClaw

# 安装所有依赖
uv sync

# 配置环境变量
cp .env.example .env
```

编辑 `.env` 填入你的密钥，**LLM 三选一**（不要混用多套 LLM 凭据）：

```env
RUNNINGHUB_API_KEY=<你的 RunningHub API 密钥>

# 方案 A：OpenAI 兼容（推荐；若设置则优先于 RunningHub LLM 与 Bedrock）
OPENAI_COMPAT_API_KEY=<你的 API 密钥>
OPENAI_COMPAT_BASE_URL=https://api.gptproto.com/v1
LLM_MODEL=gemini-2.5-pro-preview-05-06

# 方案 B：AWS Bedrock（旧方案）
# AWS_REGION=us-east-1
# AWS_BEARER_TOKEN_BEDROCK=<你的 Bedrock Token>
# LLM_MODEL=anthropic.claude-sonnet-4-20250514

# 方案 C：RunningHub LLM（与图像共用 RUNNINGHUB_API_KEY；勿与方案 A 同时启用）
# RUNNINGHUB_LLM=1
# RUNNINGHUB_LLM_BASE_URL=https://llm.runninghub.cn/v1
# LLM_MODEL=rh-llm-g/rh-g-pro-preview-31
# RUNNINGHUB_LLM_REASONING_EFFORT=high
```

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `RUNNINGHUB_API_KEY` | 是 | RunningHub 图像生成 API 密钥 |
| `OPENAI_COMPAT_API_KEY` | 方案 A | OpenAI 兼容服务的 API 密钥 |
| `OPENAI_COMPAT_BASE_URL` | 方案 A | 服务端点 URL（如 `https://api.gptproto.com/v1`） |
| `LLM_MODEL` | 是 | 所选提供方的模型 ID |
| `RUNNINGHUB_LLM` | 方案 C | 设为 `1` / `true` / `yes` / `on` 时启用 RunningHub LLM |
| `RUNNINGHUB_LLM_BASE_URL` | 否 | LLM 网关，默认 `https://llm.runninghub.cn/v1` |
| `RUNNINGHUB_LLM_REASONING_EFFORT` | 否 | 推理强度，默认 `high`；`off` 则省略该字段 |
| `AWS_REGION` | 方案 B | AWS 区域 |
| `AWS_BEARER_TOKEN_BEDROCK` | 方案 B | Bedrock 代理网关的 Bearer Token |

**优先级**：设置了 `OPENAI_COMPAT_API_KEY` → 走 OpenAI 兼容；否则 `RUNNINGHUB_LLM` 为真 → 走 RunningHub LLM；否则走 Bedrock。

#### 可选的运行时调优变量

下面这些都有合理的默认值，按需调整即可。

| 变量 | 默认 | 用途 |
|------|----:|------|
| `STYLECLAW_DATA_ROOT` | `data/projects` | 项目数据根目录 |
| `STYLECLAW_LOG_LEVEL` | `INFO` | 默认日志等级（`DEBUG` / `WARNING` 等）；`-v` 是单次调用快捷方式 |
| `STYLECLAW_SKIP_ENV_CHECK` | — | 任何 truthy 值会跳过 CLI 启动时的环境检查 |
| `STYLECLAW_MAX_ROUNDS` | `5` | 自动精炼最多多少轮 |
| `STYLECLAW_CONCURRENCY` | `5` | 图像生成并发上限 |
| `STYLECLAW_LLM_CONCURRENCY` | `4` | 并发 LLM 调用上限 |
| `STYLECLAW_TASK_TIMEOUT` | `300` | 单任务轮询超时（秒） |
| `STYLECLAW_POLL_INTERVAL` | `3` | 内层轮询间隔（秒） |
| `STYLECLAW_ORCH_POLL_INTERVAL` | `30` | orchestrator 外层轮询间隔（秒） |
| `STYLECLAW_MAX_POLL_CYCLES` | `60` | orchestrator 轮询循环上限，超过即报超时 |

验证安装：

```bash
uv run styleclaw --help
```

---

## 快速上手

### 自然语言模式（推荐）

使用 `styleclaw run` + 自然语言描述你想做什么，系统自动规划并执行：

```bash
# 第一步：创建项目
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "蜘蛛侠：平行宇宙动画风格"

# 第二步：用自然语言驱动各阶段（-p 指定项目名，多项目时必填）
uv run styleclaw run "分析风格并选出最佳模型" -p spider-verse
uv run styleclaw run "迭代优化触发短语直到评分通过" -p spider-verse
uv run styleclaw run "设计测试用例并跑批量生成" -p spider-verse
```

`run` 命令通过 LLM 将你的意图转换为执行计划（ActionPlan），展示给你确认后逐步执行。计划包含：

- **summary**：本次计划在做什么（中文一句话）
- **steps**：有序的操作列表（动作名 + 中文描述 + 参数）
- **loop**（可选）：迭代体（如"精炼 → 生成 → 等待 → 评估"反复跑直到通过）
- **stop_summary**（"停在哪"）：告诉你计划停在哪一步、下一步可以做什么

循环退出逻辑：当本轮评估通过（5 个维度都 ≥ 7.0 且总分 ≥ 7.5）时停止；当某维度跌破 5.0 时也会停下，并打印 `!! needs_human` 诊断，指出最弱维度并给出"可以这样说"的方向建议。

执行结束后，`run` 会列出 1-5 个针对当前阶段的"下一步可以这样说"示例；任何时候跑 `status <项目名>` 也会显示同样的建议。

需要参数确认的操作（`init` / `select-model` / `add-refs`）在执行前会弹出交互式提示，让你调整 LLM 给的默认值——例如改用其他模型、选 prompt-sref / prompt-only 出图方案、补全图生图源图目录等。`--yes` 会跳过顶层 "Execute?" 询问以及上述内联确认。

```bash
# 参数总览
uv run styleclaw run "<意图>" -p <项目名>           # 多项目时必填；仅一个项目时可省略
uv run styleclaw run "<意图>" --yes                 # 跳过所有确认提示
uv run styleclaw run "<意图>" --dry-run             # 只打印计划，不实际执行
uv run styleclaw run "<意图>" --no-show-thinking    # 不保存 LLM 思考过程
uv run styleclaw run "<意图>" --thinking-budget 8000 # 调高思考预算 token 上限（默认 5000）
```

### 逐步手动模式

也可以手动执行每个命令，实现更精细的控制：

```bash
# 1. 创建项目
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "蜘蛛侠：平行宇宙动画风格"
# 或从目录自动发现图片：
uv run styleclaw init spider-verse --ref-dir /path/to/refs --info "蜘蛛侠：平行宇宙动画风格"

# 2. 分析参考图片（LLM 提取风格特征 + 初始触发短语）
uv run styleclaw analyze spider-verse

# 3. 生成测试图片，对比所有模型（2 种变体 × 2 种性别）
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
# 可选：只跑指定模型，例如 styleclaw generate spider-verse --models mj-v7,niji7

# 4. 评估并选择最佳模型 + 出图方案
uv run styleclaw evaluate spider-verse
uv run styleclaw select-model spider-verse --models mj-v7 --variant prompt-sref

# 5. 精炼触发短语（重复直到满意）
uv run styleclaw refine spider-verse
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw evaluate spider-verse

# 6. 确认进入批量测试
uv run styleclaw approve spider-verse --yes        # --yes 跳过交互式 "Proceed?" 确认
uv run styleclaw design-cases spider-verse
uv run styleclaw batch-submit spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw report spider-verse
```

---

## 典型使用场景与示例

### 场景一：从零开始探索一个新 IP 的风格

适合第一次使用，或对某个 IP 完全没有触发词积累的情况。

```bash
# 准备好 3 张以上参考图，放到本地目录
uv run styleclaw init my-ip \
  --ref-dir ./refs \
  --info "某动画电影的赛璐珞手绘风格，强调粗线条和高饱和色块"

# 用自然语言一键跑完整流程
uv run styleclaw run "完整跑一遍：分析风格、选模型、精炼触发词、批量验证" -p my-ip
```

### 场景二：只想快速选出最适合的生成模型

已有参考图，想先看看哪个模型最能还原风格，再决定是否深入精炼。

```bash
uv run styleclaw init style-test \
  --ref ref-001.png --ref ref-002.png \
  --info "日系水彩插画风格"

uv run styleclaw run "分析参考图，对比所有模型，给出推荐" -p style-test
# 系统会生成 HTML 报告，可在浏览器中直观对比各模型效果
```

### 场景三：已有触发词，想验证泛化能力

已经有了一个触发短语，想用 100 个多样化用例测试它在不同主题下的稳定性。

```bash
# 假设已完成 STYLE_REFINE，当前处于 BATCH_T2I 阶段
uv run styleclaw run "设计 100 个测试用例并提交批量生成" -p my-ip

# 等待生成完成后查看报告
uv run styleclaw poll my-ip
uv run styleclaw report my-ip
```

### 场景四：精炼效果不理想，想手动给出调整方向

LLM 自动精炼几轮后，你觉得方向不对，想亲自介入。

```bash
# 查看当前状态和触发词
uv run styleclaw status my-ip

# 手动指定精炼方向
uv run styleclaw refine my-ip --direction "增加半调网点效果，降低色彩饱和度，强调黑色轮廓线"

uv run styleclaw generate my-ip
uv run styleclaw poll my-ip
uv run styleclaw evaluate my-ip
```

### 场景五：更换风格参考图后重新对比模型

发现原来的参考图不够典型，换了一张更有代表性的图，需要重新跑模型对比。

```bash
# 查看当前参考图列表（显示 0-based 索引）
uv run styleclaw status my-ip

# 切换风格参考图（例如切换到第 2 张，索引为 1）
# 当前 pass 已有 SUCCESS 结果时，会自动 bump 到下一 pass（旧 pass 保留），
# 不需要再加 --force。空 pass 时则原地改 sref_index。
uv run styleclaw set-sref my-ip 1

# 在新 pass（或空 pass）里跑生成 → 评估
uv run styleclaw generate my-ip
uv run styleclaw poll my-ip
uv run styleclaw evaluate my-ip
uv run styleclaw select-model my-ip --models mj-v7
```

### 场景六：从 STYLE_REFINE 阶段重新对比模型

精炼过程中发现选错了模型，想回到模型选择阶段重新测试。

```bash
# 在 STYLE_REFINE 或 BATCH_T2I 阶段均可执行
uv run styleclaw retest-models my-ip
# → 自动创建新的 pass 目录（pass-002），不破坏已有数据

uv run styleclaw generate my-ip
uv run styleclaw poll my-ip
uv run styleclaw evaluate my-ip
uv run styleclaw select-model my-ip --models niji7
```

### 场景七：回退到某一轮重新精炼

某一轮精炼后效果变差，想回到之前的某轮重新出发。

```bash
# 软回退：只改变状态指针，不删除任何数据
uv run styleclaw rollback my-ip --to STYLE_REFINE --round 2

# 下一次 refine 会自动创建新的轮次（跳过已有轮次编号）
uv run styleclaw refine my-ip
uv run styleclaw generate my-ip
uv run styleclaw poll my-ip
uv run styleclaw evaluate my-ip
```

### 场景八：完成文生图验证后，追加图生图测试

100 用例文生图通过后，想进一步用图生图验证风格迁移能力。

```bash
# 添加用于图生图的参考图（同时自动推进到 BATCH_I2I 阶段）
uv run styleclaw add-refs my-ip --images source1.png --images source2.png

# 提交图生图批量任务
uv run styleclaw batch-submit my-ip --i2i

uv run styleclaw poll my-ip
uv run styleclaw report my-ip --i2i

# 完成后标记项目为已完成
uv run styleclaw approve my-ip --phase completed --yes
```

### 场景九：管理多个并行项目

同时在探索多个 IP 风格，需要在项目间切换。

```bash
# 查看所有项目及其当前阶段
uv run styleclaw status

# 对特定项目执行操作（多项目时 -p 必填）
uv run styleclaw run "继续精炼触发词" -p project-a
uv run styleclaw run "提交批量测试" -p project-b

# 查看某个项目的详细状态
uv run styleclaw status project-a
```

### 场景十：先看 LLM 怎么规划，再决定要不要执行

不确定 `run` 会跑哪些步骤？用 `--dry-run` 只生成并预览计划，确认无误再去掉它正式跑。

```bash
uv run styleclaw run "完整跑一遍：分析风格、选模型、精炼触发词、批量验证" -p my-ip --dry-run
# 看到计划满意后：
uv run styleclaw run "完整跑一遍：分析风格、选模型、精炼触发词、批量验证" -p my-ip
```

### 场景十一：上一批 100 用例的某些类别不理想，重新设计

`design-cases --feedback` 总是新建一个 batch（不会覆盖已有数据），并把反馈合并进设计提示。

```bash
uv run styleclaw design-cases my-ip --feedback "上一批群像太少，场景多偏室外；这次室内场景多来几个，群像至少 15 个"
uv run styleclaw batch-submit my-ip
uv run styleclaw poll my-ip
uv run styleclaw report my-ip
```

### 场景十二：归档不再活跃的实验项目

项目越攒越多想清理时，用 `archive`（单个）或 `clean --stalled`（批量）做非破坏性归档——只是移动到 `.archive/` 下，数据不删。

```bash
# 单个项目立即归档
uv run styleclaw archive old-test

# 找出所有 7 天没动且未完成的项目（默认 dry-run，只列出不归档）
uv run styleclaw clean --stalled
uv run styleclaw clean --stalled --days 14         # 改阈值为 14 天
uv run styleclaw clean --stalled --yes             # 加 --yes 真正执行归档
```

---

## CLI 命令参考

### 编排器

| 命令 | 说明 |
|------|------|
| `run "<意图>"` | 自然语言执行 — LLM 规划，用户确认，系统自动执行 |
| `run "<意图>" -p <name>` | 指定项目名称（多项目时必填） |
| `run "<意图>" --yes` | 跳过确认，直接执行 |

### 核心流水线命令

| 命令 | 所属阶段 | 说明 |
|------|---------|------|
| `init <name> --ref <img>...` | — | 创建项目，指定参考图片 |
| `analyze <name>` | INIT | LLM 分析参考图，提取初始触发短语 |
| `generate <name>` | MODEL_SELECT / STYLE_REFINE | 提交图像生成任务 |
| `poll <name>` | 任意活跃阶段 | 轮询任务状态，下载已完成的图片 |
| `evaluate <name>` | MODEL_SELECT / STYLE_REFINE | LLM 对生成图片评分 |
| `select-model <name> --models <ids>` | MODEL_SELECT | 选择使用的模型 |
| `refine <name>` | STYLE_REFINE | LLM 精炼触发短语 |
| `approve <name>` | STYLE_REFINE / BATCH_I2I | 确认进入下一阶段 |
| `design-cases <name>` | BATCH_T2I | LLM 设计 100 个测试用例描述 |
| `batch-submit <name>` | BATCH_T2I / BATCH_I2I | 提交批量生成任务 |
| `report <name>` | BATCH_T2I / BATCH_I2I | 生成 HTML 可视化报告 |

### 辅助命令

| 命令 | 说明 |
|------|------|
| `status` | 列出所有项目（项目名 + 当前阶段） |
| `status <name>` | 查看项目详细状态，并根据当前阶段给出"建议下一步"自然语言示例 |
| `adjust <name> --direction <text>` | 手动提供精炼方向（等价于 `refine --direction`） |
| `rollback <name> --to <phase> --round <n>` | 软回退到之前的阶段/轮次（不删除数据） |
| `retest-models <name>` | 从 STYLE_REFINE / BATCH_T2I 重新进入模型选择（创建新 pass） |
| `back-to-t2i <name>` | 从 BATCH_I2I 返回 BATCH_T2I |
| `set-sref <name> <index>` | 设置用作风格参考的图片（0-based 索引） |
| `set-pass <name> <pass>` | 切换当前活跃的模型选择 pass 编号 |
| `add-refs <name> --images <img>...` | 为图生图测试添加参考图片（同时推进到 BATCH_I2I） |
| `archive <name>` | 把项目移动到 `data/projects/.archive/<时间戳>-<项目名>/`（不删除） |
| `clean --stalled [--days N] [--yes]` | 列出（默认）或归档"超过 N 天未更新且未完成"的项目 |
| `migrate <name>` | 将旧布局项目迁移到当前 pass 分层布局（可重复执行） |

### 全局参数

| 参数 | 适用命令 | 说明 |
|------|---------|------|
| `--verbose / -v` | 任意命令 | 把根日志器调到 DEBUG（仅本次调用） |
| `--show-thinking / --no-show-thinking` | `analyze` / `evaluate` / `refine` / `run` | 是否保存 LLM 思考过程到 `*.thinking.md`（默认开启） |
| `--thinking-budget <int>` | 同上 | 传给 `invoke_with_thinking` 的 token 预算（默认 5000） |
| `--dry-run` | `run` / `generate` / `batch-submit` | 只打印计划/预估任务数，不调用 API |

### 常用参数

```bash
uv run styleclaw init <name> \
  --ref <图片路径> \        # 可重复多次
  --ref-dir <目录> \        # 自动发现目录下所有图片
  --info <文本> \           # IP 描述（影响 LLM 分析方向）
  --desc <文本> \           # 项目备注
  --force                   # 覆盖已有项目

uv run styleclaw generate <name> \
  --force \                 # 仅当当前 pass 没有任何 SUCCESS 任务时（例如全是 FAILED/QUEUED）才允许；
                            # 若当前 pass 已有成功任务会被拒绝，防止静默覆盖。
                            # 想重做一个已有成功结果的 pass：用 `retest-models`（开新 pass、相同 sref）
                            # 或 `set-sref`（开新 pass、换 sref），不要用 --force。
  --models mj-v7,niji7 \    # MODEL_SELECT 阶段限定只跑这几个模型
  --dry-run                 # 只打印将要提交的任务，不实际提交

uv run styleclaw refine <name> \
  --direction <文本>        # 手动指定精炼方向

uv run styleclaw select-model <name> \
  --models <模型ID> \       # 逗号分隔，如 "mj-v7"
  --variant prompt-sref     # 或 prompt-only —— 锁定 STYLE_REFINE 阶段的出图方案。
                            # 省略时自动采用 evaluation.json 里的 recommended_variant。

uv run styleclaw design-cases <name> \
  --feedback "<文本>"       # 对上一批的反馈；本命令始终新建一个 batch，不覆盖已有

uv run styleclaw batch-submit <name> \
  --i2i \                   # 提交图生图任务（默认文生图）
  --model <模型ID> \        # 指定模型（默认使用已选模型）
  --dry-run                 # 预览将要提交的任务，不实际提交

uv run styleclaw approve <name> \
  --phase completed \       # 直接标记为已完成（BATCH_I2I → COMPLETED）
  --yes                     # 跳过确认

uv run styleclaw report <name> \
  --i2i                     # 生成图生图报告（默认文生图）
```

---

## 可用模型

| 模型 ID | 名称 | 风格引用方式 | 备注 |
|---------|------|:----------:|------|
| `mj-v7` | Midjourney v7 | `param` | 默认模型；`--sref` + `sw=100`；stylize=200，每次生成 4 张图 |
| `niji7` | Midjourney niji7 | `param` | `--sref` + `sw=100`；动漫向，stylize=200 |
| `nb2` | NanoBanana2 | `prompt` | 提示词前缀 `参考图1的风格：` + `imageUrls`；2K 分辨率，最长 20000 字符 |
| `seedream` | Seedream v5-lite | `prompt` | 提示词前缀 `参考图1的风格：` + `imageUrls`；width×height，最长 2000 字符 |
| `gpt-image-2` | GPT-Image-2 | `prompt` | 提示词前缀 `参考图1的风格：` + `imageUrls`；2K 分辨率，quality=medium，最长 20000 字符 |

**风格引用方式**：`param` 表示通过 API 参数（`--sref`）传入；`prompt` 表示通过提示词前缀 + `imageUrls` 传入。所有模型均支持风格参考。

MODEL_SELECT 阶段每个模型会测试两种变体：
- **prompt-only**：仅触发短语，不附加风格参考图
- **prompt-sref**：触发短语 + 风格参考图

若 prompt-only 效果已足够（总分 ≥ 7.0），优先选用，灵活性更高。`evaluate` 阶段会把推荐 variant 写入 `evaluation.json` 的 `recommended_variant` 字段；`select-model` 在省略 `--variant` 时会自动采用它。

---

## 风格精炼评分

在 STYLE_REFINE 阶段，LLM 会从 5 个维度对生成图片评分（满分 10 分）：

| 维度 | 说明 |
|------|------|
| 色彩调性（Color Palette） | 色彩是否匹配参考风格 |
| 线条风格（Line Style） | 笔触粗细、边缘处理、线条质感 |
| 光影效果（Lighting） | 光线方向、对比度、阴影风格 |
| 纹理质感（Texture） | 表面细节、颗粒感、材质表现 |
| 整体氛围（Overall Mood） | 情感基调和氛围一致性 |

**通过标准**：所有维度 ≥ 7.0 且总分 ≥ 7.5。

### 迭代循环的处理

每次 `evaluate` 后，LLM 会在 `evaluation.json` 给出三种 `recommendation` 之一：

| 推荐值 | 触发条件 | 应对 |
|------|------|------|
| `continue_refine` | 分数稳步上升但未达标 | `refine`（自动读取 `next_direction`） |
| `needs_human` | 某维度跌破 5，或整体倒退 | `refine --direction "<具体修正方向>"`；或 `rollback --to STYLE_REFINE --round <较好的那轮>` 后重新精炼 |
| `approve` | 5 维都 ≥ 7 且总分 ≥ 7.5 | `approve --yes` |

自动循环上限为 `STYLECLAW_MAX_ROUNDS`（默认 5 轮）。`rollback` 是软回退——旧轮次数据全部保留，下一次 `refine` 会跳过已被占用的 round 编号、新开一轮。

---

## 批量测试类别

100 用例泛化测试覆盖 10 个类别（每类 10 个）：

| 类别 | 说明 |
|------|------|
| `adult_male` | 成年男性 |
| `adult_female` | 成年女性 |
| `shota` | 少年 |
| `loli` | 少女 |
| `elderly_male` | 老年男性 |
| `elderly_female` | 老年女性 |
| `creature` | 生物/怪物 |
| `outdoor_scene` | 室外场景 |
| `indoor_scene` | 室内场景 |
| `group` | 群像 |

**泛化规则**：100 个用例中最多只有 1-2 个可引用原始 IP 元素，其余 98+ 个必须是全新主题，以验证触发短语的泛化能力。

---

## 项目数据结构

所有项目数据存储在 `data/projects/<项目名>/` 下：

```
data/projects/<项目名>/
├── config.json                          # 项目配置（名称、IP 描述、参考图列表）
├── state.json                           # 当前状态（阶段、轮次、批次、已选模型）
├── refs/                                # 参考图片 + 上传记录
├── model-select/
│   └── pass-001/                        # 每次模型对比独立 pass
│       ├── initial-analysis.json        # LLM 风格分析结果
│       ├── evaluation.json              # LLM 模型评估结果
│       ├── report.html                  # 可视化对比报告
│       └── results/<model-id>/<variant>/
├── style-refine/
│   └── pass-001/
│       └── round-001/                   # 每轮精炼独立目录
│           ├── prompt.json              # 本轮触发短语
│           ├── evaluation.json          # 本轮评分
│           └── results/<model-id>/
├── batch-t2i/
│   └── batch-001/
│       ├── cases.json                   # 100 个测试用例
│       ├── report.html
│       └── results/<case-id>/
└── batch-i2i/
    └── batch-001/
        ├── source-images/               # 图生图源图
        ├── uploads.json
        ├── cases.json
        └── results/<case-id>/
```

旧项目可能仍是未按 pass 分层的旧布局，可用以下命令迁移：

```bash
uv run styleclaw migrate <项目名>
```

---

## 开发指南

```bash
# 运行测试
uv run python -m pytest tests/ -v

# 运行测试并检查覆盖率（最低要求 80%）
uv run python -m pytest tests/ --cov=src

# 跳过慢速集成测试
uv run python -m pytest tests/ -m "not integration"
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 包管理 | uv |
| HTTP 客户端 | httpx（异步） |
| LLM | OpenAI 兼容 API、RunningHub LLM 或 AWS Bedrock（旧方案） |
| 数据模型 | Pydantic v2 |
| 命令行 | Typer |
| 报告 | Jinja2 HTML 模板 |
| 图像处理 | Pillow |
| 配置 | python-dotenv |

---

## 许可证

详见 [LICENSE](LICENSE)。
