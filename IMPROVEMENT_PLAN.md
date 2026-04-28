# StyleClaw 优化改进计划

> 共 6 条并行工作流，按文件归属划分以避免合并冲突。
> 每条工作流可在独立 worktree + 独立会话中执行。

## 工作流总览

```
时间线 ──────────────────────────────────────────────────►

WS-A [providers]   ██████████ (~1.5h)   ← 无依赖，可最先启动
WS-B [storage]     ████████ (~1h)       ← 无依赖
WS-C [scripts]     ██████████ (~1.5h)   ← 无依赖
WS-D [state+config]████ (~0.5h)         ← 无依赖
WS-E [cli+orch]    ████████████████ (~3h) ← 依赖 WS-A, WS-B, WS-C 先合入
WS-F [tests]       ████████████ (~2h)   ← 分两批：F1 立即启动，F2 等 WS-E 合入
```

---

## WS-A: Provider 可靠性加固

**分支**: `improve/providers`
**涉及文件** (仅这些文件，其他工作流不碰):
- `src/styleclaw/providers/llm/bedrock.py`
- `src/styleclaw/providers/runninghub/client.py`

**优先级**: P0-P1 | **预估**: 1.5 小时

### A1. Client 实现 async context manager (P1)

RunningHubClient 和 BedrockProvider 都需要手动 close()，容易泄漏。

**改动**:
- `RunningHubClient` 加 `__aenter__` / `__aexit__`
- `BedrockProvider` 加 `__aenter__` / `__aexit__`

```python
# 两个 client 都加上:
async def __aenter__(self) -> Self:
    return self

async def __aexit__(self, *exc: Any) -> None:
    await self.close()
```

### A2. BedrockProvider 加重试 (P1)

bedrock.py:57-58 直接 post 无重试，LLM 调用容易超时。

**改动**:
- `invoke()` 方法加指数退避重试，复用 RunningHubClient 相同模式
- 重试次数 3 次，捕获 `httpx.HTTPStatusError`（仅 5xx）和 `httpx.TransportError`

### A3. Semaphore 内存泄漏修复 (P2)

client.py:15 的 `_semaphore_map` 以 event loop ID 为 key，从不清理。

**改动**:
- 将 semaphore 移到 `RunningHubClient` 实例属性上（每个 client 一个）
- 删除模块级 `_semaphore_map`

```python
class RunningHubClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        # ...
```

### 验收标准
- [ ] `async with RunningHubClient(...) as client:` 可用
- [ ] `async with BedrockProvider() as llm:` 可用
- [ ] Bedrock invoke 失败时自动重试（3 次）
- [ ] 无模块级 semaphore map
- [ ] 现有测试全部通过

---

## WS-B: 存储层改进

**分支**: `improve/storage`
**涉及文件**:
- `src/styleclaw/storage/project_store.py`

**优先级**: P0-P2 | **预估**: 1 小时

### B1. 项目名路径穿越防御 (P0)

`project_dir(name)` 直接用用户输入拼路径，存在安全风险。

**改动**:
- 新增 `_validate_project_name(name: str)` 函数
- 在 `create_project()` 和 `project_dir()` 入口调用
- 拒绝包含 `/`、`\`、`..` 或以 `.` 开头的名字

```python
import re

_PROJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

def _validate_project_name(name: str) -> None:
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name '{name}'. "
            "Use only letters, digits, hyphens, and underscores."
        )
```

### B2. 泛型化 load/save 减少重复 (P2)

当前 t2i、i2i、round 各有一套几乎相同的 load/save/load_all 函数。

**改动**:
- 抽出 `_load_model(cls, path)` 和 `_save_model(model, path)` 泛型函数
- 抽出 `_load_all_records(results_dir: Path) -> dict[str, TaskRecord]` 统一遍历逻辑
- 保持所有现有公开函数签名不变（内部委托到泛型实现）

```python
from typing import TypeVar
T = TypeVar("T", bound=BaseModel)

def _load_model(model_cls: type[T], path: Path) -> T:
    return model_cls.model_validate(_read_json(path))

def _save_model(model: BaseModel, path: Path) -> None:
    _write_json(path, model.model_dump())

def _load_all_records(results_dir: Path) -> dict[str, TaskRecord]:
    records: dict[str, TaskRecord] = {}
    if not results_dir.exists():
        return records
    for d in results_dir.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = TaskRecord.model_validate(_read_json(task_file))
    return records
```

### 验收标准
- [ ] `project_dir("../../etc")` 抛出 ValueError
- [ ] `project_dir("my-project")` 正常工作
- [ ] 所有现有 store 测试通过
- [ ] load_all_*_task_records 三个函数都委托到统一实现

---

## WS-C: Scripts 性能与健壮性

**分支**: `improve/scripts`
**涉及文件**:
- `src/styleclaw/scripts/poll.py`
- `src/styleclaw/scripts/generate.py`
- `src/styleclaw/scripts/batch_submit.py`

**优先级**: P1-P3 | **预估**: 1.5 小时

### C1. poll 并发化 (P2)

poll.py 中 100 个 batch case 逐个串行轮询，效率低。

**改动**:
- 抽出 `_download_results(results, dest_dir)` 消除三处下载重复
- `poll_batch` 内部用 `asyncio.TaskGroup` 并发轮询（已有 semaphore(5) 限流）
- `poll_model_select` 和 `poll_style_refine` 同理改为并发

```python
async def _download_results(
    results: list[dict[str, Any]], dest_dir: Path, client: RunningHubClient | None = None,
) -> None:
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        if url:
            dest = dest_dir / f"output-{i:03d}.png"
            await download_image(url, dest, client=client)
```

### C2. generate 幂等性保护 (P3)

重复运行 `generate` 会重复提交任务，浪费 API 调用。

**改动**:
- `generate_model_select()` 提交前检查是否已有非 FAILED 的 TaskRecord
- `generate_style_refine()` 同理
- 跳过已提交的模型，只提交缺失的

### C3. poll 进度实时持久化 (P3)

当前 poll 完成所有才保存，崩溃丢进度。

**改动**:
- 每个 task poll 完成后立即 save record + 下载图片
- 这在并发化后自然实现（每个 task 独立保存）

### 验收标准
- [ ] poll_batch 并发执行（可通过日志时间戳验证）
- [ ] 重复 generate 不会重复提交已成功的 task
- [ ] 下载重复逻辑只有一处定义
- [ ] 现有 poll/generate 测试通过

---

## WS-D: 状态机 + 配置集中化

**分支**: `improve/core`
**涉及文件**:
- `src/styleclaw/core/state_machine.py`
- `src/styleclaw/core/config.py` (**新建**)

**优先级**: P2-P3 | **预估**: 0.5 小时

### D1. 集中管理魔法数字 (P2)

MAX_AUTO_ROUNDS、CONCURRENCY_LIMIT、TASK_TIMEOUT 等散布各处。

**改动**:
- 新建 `src/styleclaw/core/config.py`
- 从环境变量读取，提供合理默认值
- 各模块改为 `from styleclaw.core.config import ...`

```python
# src/styleclaw/core/config.py
import os

MAX_AUTO_ROUNDS: int = int(os.getenv("STYLECLAW_MAX_ROUNDS", "5"))
CONCURRENCY_LIMIT: int = int(os.getenv("STYLECLAW_CONCURRENCY", "5"))
TASK_TIMEOUT: float = float(os.getenv("STYLECLAW_TASK_TIMEOUT", "300"))
POLL_INTERVAL: float = float(os.getenv("STYLECLAW_POLL_INTERVAL", "3"))
ORCHESTRATOR_POLL_INTERVAL: float = float(os.getenv("STYLECLAW_ORCH_POLL_INTERVAL", "30"))
MAX_POLL_CYCLES: int = int(os.getenv("STYLECLAW_MAX_POLL_CYCLES", "60"))
```

### D2. 状态机支持跳过 BATCH_I2I (P3)

用户可能不需要 i2i 测试，但目前强制走 BATCH_T2I -> BATCH_I2I -> COMPLETED。

**改动**:
- TRANSITIONS 中增加 `Phase.BATCH_T2I: [..., Phase.COMPLETED]`

```python
TRANSITIONS: dict[Phase, list[Phase]] = {
    # ...
    Phase.BATCH_T2I: [Phase.BATCH_I2I, Phase.STYLE_REFINE, Phase.COMPLETED],
    # ...
}
```

### 验收标准
- [ ] 所有常量可通过环境变量覆盖
- [ ] BATCH_T2I 可直接跳转 COMPLETED
- [ ] 现有 state_machine 测试通过 + 新增转换测试

---

## WS-E: CLI + Orchestrator 重构 (最大改动)

**分支**: `improve/cli-orchestrator`
**涉及文件**:
- `src/styleclaw/cli.py`
- `src/styleclaw/orchestrator/actions.py`

**优先级**: P1 | **预估**: 3 小时

> ⚠️ **依赖**: 建议在 WS-A、WS-B、WS-C 合入 main 后再开始，
> 因为本工作流会大幅改写 cli.py，与其他改动冲突概率高。
> 如需并行，至少确保 WS-A（context manager）先合入。

### E1. CLI 委托 actions 层 (P1)

当前 cli.py 和 actions.py 各自实现一遍同样的逻辑。

**改动思路**:
- CLI 命令只负责：参数解析 -> 构造 ExecutionContext -> 调用 action 函数 -> 格式化输出
- actions.py 成为唯一的业务逻辑层
- 删除 cli.py 中的 `_generate_model_select`、`_generate_style_refine`、`_poll_*`、`_evaluate_*` 等私有函数

```python
# 改造后的 cli.py 命令示例:
@app.command()
def generate(name: str = typer.Argument(...)):
    """Submit generation tasks (auto-detects phase)."""
    result = _run_action(name, do_generate, {}, needs_client=True)
    typer.echo(result.message)

def _run_action(
    name: str,
    action_fn: Callable,
    args: dict,
    needs_client: bool = False,
    needs_llm: bool = False,
) -> StepResult:
    async def _exec():
        async with _build_context(name, needs_client, needs_llm) as ctx:
            return await action_fn(ctx, args)
    return asyncio.run(_exec())
```

### E2. do_poll 加超时保护 (P1)

actions.py:90 的 `while True` 无上限。

**改动**:
- 加 `max_cycles` 参数（从 config.py 读取，默认 60）
- 超时后返回 `StepResult(ok=False, message="Poll timed out")`

### E3. run 命令资源泄漏修复 (P1)

cli.py:731 创建 BedrockProvider 但取消执行时不 close。

**改动**:
- 统一用 context manager 管理所有 client/llm 生命周期
- 这在 E1 重构中自然解决

### 验收标准
- [ ] cli.py 缩减到 ~400 行
- [ ] actions.py 是唯一的业务逻辑层
- [ ] 所有 CLI 命令功能不变
- [ ] `run` 命令取消后无资源泄漏
- [ ] do_poll 有超时退出机制
- [ ] 手动测试: `styleclaw --help` 和 `styleclaw status` 正常

---

## WS-F: 测试补全

**分支**: `improve/tests` (F1) 和 `improve/tests-coverage` (F2)
**涉及文件**:
- `tests/` 目录下所有文件（不碰 src/）

**优先级**: P0 (F1) + P1 (F2) | **预估**: 2 小时

### F1. 修复失败测试 (P0) — 可立即启动

`test_submits_for_all_models` 因新增 gpt-image-2 导致断言失败。

**改动**:
- 更新测试中预期的模型数量（4 -> 5）或改为动态读取 `len(MODEL_REGISTRY)`
- 推荐用动态方式，避免每次加模型都要改测试

### F2. 补充 orchestrator 测试以达到 80% 覆盖 (P1) — 等 WS-E 合入后

> ⚠️ 如果 WS-E 重构了 actions.py，测试应基于重构后的版本写。
> 如果 WS-E 尚未开始，可先基于现有代码补测试，但预期会有改动。

**需要覆盖的函数**:
- `do_analyze` — mock LLM + store
- `do_generate` — mock client, 分 MODEL_SELECT 和 STYLE_REFINE 两个分支
- `do_poll` — mock client, 测试正常完成 + 超时退出
- `do_evaluate` — mock LLM, 分两个 phase
- `do_select_model` — 测试 MODEL_SELECT 和 STYLE_REFINE 两个分支
- `do_refine` — mock LLM
- `do_approve` — 测试 batch-t2i 和 completed 两个 target
- `do_batch_submit` — mock client, 分 t2i 和 i2i

**额外**: 新增一个端到端集成测试，走完 init -> analyze -> generate -> poll -> evaluate -> select-model 的完整状态机流转（全部用 respx mock HTTP）。

### 验收标准
- [ ] 0 failing tests
- [ ] 总覆盖率 >= 80%
- [ ] actions.py 覆盖率 >= 70%

---

## 合并顺序

```
阶段 1 (并行):  WS-A + WS-B + WS-C + WS-D + WS-F1
                 ↓ 全部合入 main
阶段 2:         WS-E (基于最新 main)
                 ↓ 合入 main
阶段 3:         WS-F2 (基于重构后的 actions.py 补测试)
                 ↓ 合入 main -> 最终验证覆盖率 >= 80%
```

## 各工作流 Claude Code 启动命令参考

```bash
# 每个工作流在独立终端 + worktree 中执行:

# WS-A
claude "按 IMPROVEMENT_PLAN.md 中 WS-A 的内容执行，涉及 providers 层的 context manager、重试、semaphore 修复"

# WS-B
claude "按 IMPROVEMENT_PLAN.md 中 WS-B 的内容执行，涉及 project_store 的路径校验和泛型化"

# WS-C
claude "按 IMPROVEMENT_PLAN.md 中 WS-C 的内容执行，涉及 poll 并发化、generate 幂等性、下载去重"

# WS-D
claude "按 IMPROVEMENT_PLAN.md 中 WS-D 的内容执行，涉及 config.py 集中配置和状态机跳转"

# WS-F1
claude "按 IMPROVEMENT_PLAN.md 中 WS-F1 的内容执行，修复 test_submits_for_all_models 失败测试"

# --- 以上合入后 ---

# WS-E
claude "按 IMPROVEMENT_PLAN.md 中 WS-E 的内容执行，CLI 委托 actions 层重构 + do_poll 超时 + 资源泄漏修复"

# WS-F2
claude "按 IMPROVEMENT_PLAN.md 中 WS-F2 的内容执行，补充 orchestrator 测试达到 80% 覆盖率"
```
