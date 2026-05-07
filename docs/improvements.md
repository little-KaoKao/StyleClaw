# StyleClaw 改进实施计划

来源：对话中的代码审查分析（23 个问题）。按优先级分 4 周实施。

---

## Week 1 — 安全与稳定性

### 1. 敏感信息泄露（bedrock.py）
**目标**：防止 bearer token 出现在日志/traceback 中。

- `bedrock.py`：将 `self._token` 改为 `pydantic.SecretStr`，构建 headers 时调用 `.get_secret_value()`，不在 `__repr__` / `__str__` 中暴露
- 同样处理 `openai_compat.py`（如存在）

### 2. 路径遍历加固（project_store.py）
**目标**：确保 `project_dir()` 解析后仍在 `DATA_ROOT` 内。

- `project_store.py:project_dir()`：在返回前调用 `.resolve()`，并断言结果 `is_relative_to(DATA_ROOT.resolve())`

### 3. 资源清理保障（cli.py）
**目标**：`_build_context()` 中任一 provider 初始化失败时，已创建的资源仍能被清理。

- `cli.py:_build_context()`：分两步初始化（先 client，再 llm），`finally` 块对每个 close 加 `asyncio.wait_for(..., timeout=5.0)` 并捕获异常单独记录

### 4. 启动时配置验证（cli.py）
**目标**：在执行任何命令前检测缺失的环境变量。

- 新增 `src/styleclaw/core/config.py`：`validate_env() -> list[str]`，检查 `RUNNINGHUB_API_KEY`、至少一个 LLM token
- `cli.py` 的 `@app.callback()` 中调用，有错误则打印并 `raise typer.Exit(1)`

---

## Week 2 — 性能优化

### 5. 确保异步图片编码一致（image_utils.py + agents）
**目标**：消除 async 上下文中的同步 CPU 阻塞。

- 审计所有 `encode_image_for_llm()` 和 `build_image_block()` 的调用点
- 在 async 函数中统一改用已有的 `build_image_block_async()`
- 文件：`agents/analyze_style.py`、`agents/select_model.py`、`agents/evaluate_result.py`

### 6. 轮询指数退避（tasks.py）
**目标**：减少无效轮询请求，降低 API 成本。

- `tasks.py:poll_task()`：前 3 次保持 `interval`，之后 `min(interval * 1.5^(n-3), 60)`
- 不改变函数签名，退避逻辑内置

### 7. 下载重试机制（poll.py）
**目标**：单张图片下载失败时自动重试，并记录失败列表。

- `poll.py:_download_results()`：每个 URL 最多重试 3 次，指数退避
- `DownloadStats` 增加 `failed_urls: list[str]` 字段
- 失败列表写入 `dest_dir/failed_downloads.json`

### 8. 连接池配置（bedrock.py）
**目标**：为高并发场景配置合理的连接池。

- `bedrock.py`：`httpx.AsyncClient` 增加 `limits=httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=30.0)`

---

## Week 3 — 用户体验

### 9. 进度条（generate.py、batch_submit.py）
**目标**：批量提交时显示实时进度。

- 依赖：`uv add tqdm`
- `generate.py` 和 `batch_submit.py` 的 `TaskGroup` 循环中，用 `tqdm` 包装，`task.add_done_callback` 更新进度

### 10. 改进错误信息（text_utils.py）
**目标**：JSON 解析失败时显示清理后的内容预览和修复提示。

- `text_utils.py:parse_llm_response()`：捕获 `JSONDecodeError` 时附加 `cleaned[:200]` 预览和 hint 文本

### 11. 状态机友好提示（state_machine.py）
**目标**：转换失败时告知用户下一步操作。

- `state_machine.py:advance()`：根据 `state.phase` 附加具体的操作提示（如"Run 'styleclaw analyze'"）

### 12. 幂等性支持（project_store.py、cli.py）
**目标**：允许安全重试失败的操作。

- `project_store.py:create_project()`：增加 `force: bool = False` 参数；`force=True` 时先备份再重建
- `cli.py:init` 命令：透传 `--force` 选项

---

## Week 4 — 高级功能

### 13. 检查点机制（batch_submit.py）
**目标**：批量任务中断后可从断点续传。

- 新增 `src/styleclaw/core/checkpoint.py`：`Checkpoint` 类，`save(key, value)` / `get(key)` / `clear()`，持久化到 `data/projects/<name>/.checkpoint_<phase>.json`
- `batch_submit.py`：提交前检查 checkpoint，完成后更新，全部完成后清理

### 14. Dry-run 模式（cli.py）
**目标**：预览操作而不实际执行。

- `generate`、`batch-submit` 命令增加 `--dry-run` 选项
- dry-run 时打印计划操作（模型列表、任务数量）后返回

### 15. 日志级别规范（tasks.py、generate.py、poll.py）
**目标**：减少 INFO 噪音，让重要信息更突出。

- 每个任务的提交/轮询细节：`INFO` → `DEBUG`
- 阶段性进展（"开始生成 N 个任务"、"完成 50/100"）保留 `INFO`
- 可恢复错误（重试、跳过）：`WARNING`

---

## 不在计划内的项目

以下分析中提到的建议**暂不实施**，原因如下：

| 建议 | 原因 |
|------|------|
| MIME 类型验证（python-magic） | 引入系统级依赖（libmagic），Pillow 解码验证已足够 |
| 请求去重（内容哈希） | 现有本地记录检查已覆盖主要场景，哈希方案复杂度高 |
| 性能监控装饰器 | 过度设计，日志已提供足够可观测性 |
| 交互式配置向导 | 低频操作，.env.example 已足够 |
| 数据导出功能 | 超出当前需求范围 |
| WebP 格式优化 | 影响极小，不值得增加复杂度 |
| 测试覆盖率补充 | 需要单独专项，不在本计划范围 |
| 架构演进（数据库、Web UI、插件） | 长期方向，不在本计划范围 |

---

## 文件变更清单

| 文件 | 涉及问题 |
|------|---------|
| `src/styleclaw/providers/llm/bedrock.py` | #1 #8 |
| `src/styleclaw/storage/project_store.py` | #2 #12 |
| `src/styleclaw/cli.py` | #3 #4 #9 #12 #14 |
| `src/styleclaw/core/config.py` (新建) | #4 |
| `src/styleclaw/core/image_utils.py` | #5 |
| `src/styleclaw/agents/*.py` | #5 |
| `src/styleclaw/providers/runninghub/tasks.py` | #6 |
| `src/styleclaw/scripts/poll.py` | #7 |
| `src/styleclaw/scripts/generate.py` | #9 #15 |
| `src/styleclaw/scripts/batch_submit.py` | #9 #13 #15 |
| `src/styleclaw/core/text_utils.py` | #10 |
| `src/styleclaw/core/state_machine.py` | #11 |
| `src/styleclaw/core/checkpoint.py` (新建) | #13 |
| `pyproject.toml` | #9（tqdm 依赖） |
