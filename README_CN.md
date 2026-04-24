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

## 环境要求

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** 包管理器
- **RunningHub** API 密钥（用于图像生成）
- **AWS Bedrock** 访问权限 + Bearer Token（用于 Claude LLM 调用）

## 安装

```bash
git clone https://github.com/little-KaoKao/StyleClaw.git
cd StyleClaw

# 安装所有依赖
uv sync

# 配置环境变量
cp .env.example .env
```

编辑 `.env` 填入你的密钥：

```env
RUNNINGHUB_API_KEY=<你的 RunningHub API 密钥>
AWS_REGION=us-east-1
AWS_BEARER_TOKEN_BEDROCK=<你的 Bedrock Token>
CLAUDE_MODEL=anthropic.claude-sonnet-4-20250514    # 可选
```

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `RUNNINGHUB_API_KEY` | 是 | RunningHub 图像生成 API 密钥 |
| `AWS_REGION` | 是 | AWS Bedrock 区域（如 `us-east-1`） |
| `AWS_BEARER_TOKEN_BEDROCK` | 是 | AWS Bedrock Bearer Token 认证 |
| `CLAUDE_MODEL` | 否 | Bedrock 模型 ID（默认：`anthropic.claude-sonnet-4-20250514`） |

验证安装：

```bash
uv run styleclaw --help
```

## 快速上手

### 自然语言模式（推荐）

使用 `styleclaw run` + 自然语言描述你想做什么，系统自动规划并执行：

```bash
# 先创建项目
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "蜘蛛侠：平行宇宙动画风格"

# 然后用自然语言描述意图
uv run styleclaw run "分析风格并选出最佳模型"
uv run styleclaw run "迭代优化触发短语直到评分通过"
uv run styleclaw run "设计测试用例并跑批量生成"
```

`run` 命令通过 LLM 将你的意图转换为执行计划，展示给你确认后逐步执行。支持循环执行（精炼 → 生成 → 等待 → 评估），根据评分自动决定是否继续迭代。

```bash
# 选项
uv run styleclaw run "<意图>" -p <项目名>   # 指定项目（仅有一个项目时自动选择）
uv run styleclaw run "<意图>" --yes          # 跳过确认直接执行
```

### 逐步手动模式

也可以手动执行每个命令，实现更精细的控制：

```bash
# 1. 创建项目，指定参考图片和 IP 描述
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "蜘蛛侠：平行宇宙动画风格"

# 2. 分析参考图片（LLM 提取风格特征 + 初始触发短语）
uv run styleclaw analyze spider-verse

# 3. 生成测试图片，对比所有模型
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse

# 4. 评估并选择最佳模型
uv run styleclaw evaluate spider-verse
uv run styleclaw select-model spider-verse --models mj-v7

# 5. 精炼触发短语（重复直到满意）
uv run styleclaw refine spider-verse
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw evaluate spider-verse

# 6. 确认进入批量测试
uv run styleclaw approve spider-verse
uv run styleclaw design-cases spider-verse
uv run styleclaw batch-submit spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw report spider-verse
```

## CLI 命令参考

### 编排器

| 命令 | 说明 |
|------|------|
| `run "<意图>"` | 自然语言执行 — LLM 规划，用户确认，系统自动执行 |
| `run "<意图>" -p <name>` | 指定项目名称 |
| `run "<意图>" --yes` | 跳过确认，直接执行 |

### 核心流水线命令

| 命令 | 所属阶段 | 说明 |
|------|---------|------|
| `init <name> --ref <img>...` | — | 创建项目，指定参考图片 |
| `analyze <name>` | INIT | LLM 分析参考图，提取初始触发短语 |
| `generate <name>` | MODEL_SELECT / STYLE_REFINE | 提交图像生成任务 |
| `poll <name>` | 任意活跃阶段 | 轮询任务状态，下载已完成的图片 |
| `evaluate <name>` | MODEL_SELECT / STYLE_REFINE | LLM 对生成图片评分 |
| `select-model <name> --models <ids>` | MODEL_SELECT / STYLE_REFINE | 选择使用的模型 |
| `refine <name>` | STYLE_REFINE | LLM 精炼触发短语 |
| `approve <name>` | STYLE_REFINE / BATCH_I2I | 确认进入下一阶段 |
| `design-cases <name>` | BATCH_T2I | LLM 设计 100 个测试用例描述 |
| `batch-submit <name>` | BATCH_T2I / BATCH_I2I | 提交批量生成任务 |
| `report <name>` | BATCH_T2I / BATCH_I2I | 生成 HTML 可视化报告 |

### 辅助命令

| 命令 | 说明 |
|------|------|
| `status` | 列出所有项目 |
| `status <name>` | 查看项目详细状态 |
| `adjust <name> --direction <text>` | 手动提供精炼方向 |
| `rollback <name> --to <phase> --round <n>` | 回退到之前的阶段 |
| `add-refs <name> --images <img>...` | 为图生图测试添加参考图片 |

### 常用参数

```bash
uv run styleclaw init <name> \
  --ref <图片路径>            # 参考图片（可重复指定多张）
  --info <文本>               # IP 描述信息
  --desc <文本>               # 项目描述

uv run styleclaw refine <name> \
  --direction <文本>          # 可选：人工指定精炼方向

uv run styleclaw batch-submit <name> \
  --i2i                       # 提交图生图任务（默认为文生图）
  --model <模型ID>            # 指定模型（默认使用项目已选模型）

uv run styleclaw approve <name> \
  --phase completed           # 用于 BATCH_I2I → COMPLETED 转换
  --yes                       # 跳过确认提示

uv run styleclaw report <name> \
  --i2i                       # 生成图生图报告（默认为文生图）
```

## 可用模型

| 模型 ID | 名称 | 风格引用 | 备注 |
|---------|------|:--------:|------|
| `mj-v7` | Midjourney v7 | 支持 | 默认模型；stylize=200，每次生成 4 张图 |
| `niji7` | Midjourney niji7 | 支持 | 动漫向，stylize=200 |
| `nb2` | NanoBanana2 | 不支持 | 2K 分辨率，提示词最长 20000 字符 |
| `seedream` | Seedream v5-lite | 不支持 | 使用 width×height 而非 aspectRatio，提示词最长 2000 字符 |

## 风格精炼评分

在 STYLE_REFINE 阶段，LLM 会从 5 个维度对生成图片评分：

| 维度 | 说明 |
|------|------|
| 色彩调性（Color Palette） | 色彩是否匹配参考风格 |
| 线条风格（Line Style） | 笔触粗细、边缘处理、线条质感 |
| 光影效果（Lighting） | 光线方向、对比度、阴影风格 |
| 纹理质感（Texture） | 表面细节、颗粒感、材质表现 |
| 整体氛围（Overall Mood） | 情感基调和氛围一致性 |

**通过标准**：所有维度 ≥ 7.0 且总分 ≥ 7.5（满分 10 分）。

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

## 项目数据结构

所有项目数据存储在 `data/projects/<项目名>/` 下：

```
data/projects/<项目名>/
├── config.json              # 项目配置
├── state.json               # 当前状态（阶段、轮次、批次、已选模型）
├── refs/                    # 参考图片 + 上传记录
├── model-select/            # 模型对比结果 + 报告
├── style-refine/round-NNN/  # 各轮精炼结果 + 评估
├── batch-t2i/batch-NNN/     # 100 用例文生图结果 + 报告
└── batch-i2i/batch-NNN/     # 图生图结果 + 报告
```

## 开发指南

```bash
# 运行测试
uv run python -m pytest tests/ -v

# 运行测试并检查覆盖率（最低要求 80%）
uv run python -m pytest tests/ --cov=src

# 跳过慢速集成测试
uv run python -m pytest tests/ -m "not integration"
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 包管理 | uv |
| HTTP 客户端 | httpx（异步） |
| LLM | Claude via AWS Bedrock |
| 数据模型 | Pydantic v2 |
| 命令行 | Typer |
| 报告 | Jinja2 HTML 模板 |
| 图像处理 | Pillow |
| 配置 | python-dotenv |

## 许可证

详见 [LICENSE](LICENSE)。
