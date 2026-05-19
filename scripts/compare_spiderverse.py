"""Standalone Spider-Verse style 9-case × 5-model comparison runner.

Does NOT touch the styleclaw project state machine — output lives under
``data/batch/<run-name>/`` only. Run with::

    uv run python scripts/compare_spiderverse.py
    uv run python scripts/compare_spiderverse.py --run-name my-run
    uv run python scripts/compare_spiderverse.py --models mj-v7,niji7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from styleclaw.core.config import DOWNLOAD_CONCURRENCY
from styleclaw.core.models import TaskRecord, TaskStatus
from styleclaw.core.prompt_builder import build_params
from styleclaw.core.time_utils import utcnow_iso
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import MODEL_REGISTRY, get_model
from styleclaw.providers.runninghub.tasks import poll_and_update, submit_task
from styleclaw.storage.image_store import download_image

logger = logging.getLogger("compare_spiderverse")


TRIGGER_PHRASE = (
    "Spider-Verse glitched comic animation style, halftone dots, "
    "chromatic aberration, dynamic lines, neon color palette, "
    "multi-dimensional distortions."
)

PORTRAIT_RATIO = "9:16"  # character cases
LANDSCAPE_RATIO = "16:9"  # scene-only cases

DEFAULT_MODELS: tuple[str, ...] = ("mj-v7", "niji7", "nb2", "seedream", "gpt-image-2")


@dataclass(frozen=True)
class Case:
    id: str
    label_zh: str
    label_en: str
    description: str
    aspect_ratio: str = PORTRAIT_RATIO


CASES: tuple[Case, ...] = (
    Case(
        id="01-adult-male",
        label_zh="成男",
        label_en="Adult Male — Dimension-K Runner",
        description=(
            'A young adult male "Dimension-K Runner." He wears a futuristic '
            "jacket that flickers with data streams, a neon blue arm, and "
            "holds a secure, encrypted data core. His skin has a subtle "
            "chromatic aberration effect. Background is a glitched cyber-punk "
            "city. Halftone dots, dynamic lines, and multi-dimensional "
            "distortions are visible."
        ),
    ),
    Case(
        id="02-adult-female",
        label_zh="成女",
        label_en="Adult Female — Dimension-K Architect",
        description=(
            'An adult female "Dimension-K Architect." Her sleek bodysuit is '
            "composed of fragmented data shards, and her long neon-pink hair "
            "flows with dynamic, pixelated lines. She constructs a geometric "
            "hologram with her hand. The background is an abstract, glitched "
            "reality. Halftone dots and chromatic aberration create depth."
        ),
    ),
    Case(
        id="03-young-boy",
        label_zh="正太",
        label_en="Young Boy — Glitch Hacker",
        description=(
            'A young boy "Glitch Hacker." He has a retro-futuristic cap and a '
            "backpack that releases neon data particles. He rides a pixelated "
            "skateboard. His eyes show multi-dimensional glimmers. Background "
            "is a vibrant, glitched alley. Chromatic aberration and halftone "
            "dots are central to the texture."
        ),
    ),
    Case(
        id="04-young-girl",
        label_zh="萝莉",
        label_en="Young Girl — Dimensional Singer",
        description=(
            'A young girl "Dimensional Singer." She wears a flowing dress made '
            "of neon light and sound waves. Her microphone is a glitched "
            "energy source. A cascade of geometric sound data surrounds her. "
            "Background is a concert stage in a fractured dimension. Halftone "
            "dots and chromatic aberration define the art."
        ),
    ),
    Case(
        id="05-old-man",
        label_zh="老头",
        label_en="Old Man — Information Librarian",
        description=(
            'An elderly male "Information Librarian." His entire suit is made '
            "of old circuits and torn book pages that flicker with a dynamic "
            "code. He holds a neon magnifying glass. His beard is a mix of "
            "pixels and hair. Background is an endless library of glitched "
            "data. Chromatic aberration is severe. Halftone dots for shadows."
        ),
    ),
    Case(
        id="06-old-woman",
        label_zh="老太",
        label_en="Old Woman — Reality Weaver",
        description=(
            'An elderly female "Reality Weaver." She is at a large, glitched '
            "loom, weaving threads of pure neon energy into a stable reality "
            "pattern. She has a subtle, colorful haze around her head. Her "
            "expression is focused. Background is a cosmic, fragmented "
            "dimension. Halftone dots and dynamic lines texture the scene."
        ),
    ),
    Case(
        id="07-pet",
        label_zh="宠物",
        label_en="Pet — Dimensional Hound",
        description=(
            "A multi-dimensional hound-like pet. Its entire body is composed "
            "of flowing data streams and geometric code blocks. Its fur is a "
            "patchwork of pixelated textures. Its eyes glow with a powerful, "
            "shifting neon light. Background is an unstable, glitched "
            "dimension. Chromatic aberration and halftone dots create a "
            "vibrant feel."
        ),
    ),
    Case(
        id="08-outdoor",
        label_zh="室外（无人）",
        label_en="Outdoor (Unmanned) — Dimension-K Data Plaza",
        description=(
            'An empty, vast "Dimension-K Data Plaza." It features futuristic, '
            "flickering neon architecture and upside-down data waterfalls. "
            "Geometric patterns dominate the ground. Large circles and "
            "dynamic lines of energy crisscross the air. No people are "
            "present. Halftone dots texture the shadows. Chromatic aberration "
            "is heavy."
        ),
        aspect_ratio=LANDSCAPE_RATIO,
    ),
    Case(
        id="09-indoor",
        label_zh="室内（无人）",
        label_en="Indoor (Unmanned) — Glitch Archive Inner Sanctum",
        description=(
            'The empty, inner sanctuary of the "Glitch Archives." Shelves are '
            "filled with glowing data cubes that flicker and distort. A "
            "single, large hologram of an abstract, fragmented dimension "
            "pulses. The floor is an unstable data grid. No people are "
            "present. Halftone dots and chromatic aberration create a dense, "
            "chaotic atmosphere."
        ),
        aspect_ratio=LANDSCAPE_RATIO,
    ),
)


@dataclass(frozen=True)
class JobKey:
    model_id: str
    case_id: str

    def __str__(self) -> str:
        return f"{self.model_id}::{self.case_id}"


# ---------------------------------------------------------------- filesystem


def _run_dir(root: Path, run_name: str) -> Path:
    return root / "data" / "batch" / run_name


def _case_dir(run_root: Path, model_id: str, case_id: str) -> Path:
    return run_root / "results" / model_id / case_id


def _save_tasks_index(run_root: Path, records: dict[str, TaskRecord]) -> None:
    """Persist the full job → TaskRecord map so a re-run can pick up where the
    previous one left off (we don't implement resume here, but the index makes
    debugging trivial)."""
    payload = {key: rec.model_dump() for key, rec in records.items()}
    (run_root / "tasks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_config(run_root: Path, models: list[str]) -> None:
    cfg = {
        "trigger_phrase": TRIGGER_PHRASE,
        "models": models,
        "cases": [
            {
                "id": c.id,
                "label_zh": c.label_zh,
                "label_en": c.label_en,
                "description": c.description,
                "aspect_ratio": c.aspect_ratio,
            }
            for c in CASES
        ],
        "created_at": utcnow_iso(),
    }
    (run_root / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ----------------------------------------------------------- submit / poll


async def _submit_one(
    client: RunningHubClient, model_id: str, case: Case
) -> TaskRecord:
    mc = get_model(model_id)
    params = build_params(
        model_id=model_id,
        trigger_phrase=TRIGGER_PHRASE,
        character_desc=case.description,
        aspect_ratio=case.aspect_ratio,
    )
    return await submit_task(client, mc.t2i_endpoint, params, model_id)


async def submit_all(
    client: RunningHubClient, models: list[str]
) -> dict[str, TaskRecord]:
    jobs: list[JobKey] = [
        JobKey(model_id=m, case_id=c.id) for m in models for c in CASES
    ]
    cases_by_id = {c.id: c for c in CASES}

    async def _wrapped(job: JobKey) -> tuple[JobKey, TaskRecord | BaseException]:
        try:
            rec = await _submit_one(client, job.model_id, cases_by_id[job.case_id])
            logger.info("Submitted %s -> %s", job, rec.task_id)
            return job, rec
        except (RuntimeError, httpx.HTTPError) as exc:
            logger.error("Submit failed for %s: %s", job, exc)
            # Build a placeholder FAILED record so the pipeline keeps going
            return job, TaskRecord(
                task_id="",
                model_id=job.model_id,
                status=TaskStatus.FAILED,
                prompt="",
                params={},
                error_message=f"submit_error: {exc}",
            )

    outcomes = await asyncio.gather(*[_wrapped(j) for j in jobs])
    records: dict[str, TaskRecord] = {}
    for job, outcome in outcomes:
        if isinstance(outcome, BaseException):
            records[str(job)] = TaskRecord(
                task_id="",
                model_id=job.model_id,
                status=TaskStatus.FAILED,
                error_message=f"submit_exception: {outcome}",
            )
        else:
            records[str(job)] = outcome
    return records


async def poll_all(
    client: RunningHubClient, records: dict[str, TaskRecord]
) -> dict[str, TaskRecord]:
    async def _one(key: str, rec: TaskRecord) -> tuple[str, TaskRecord]:
        if not rec.task_id:
            return key, rec
        if rec.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
            return key, rec
        new = await poll_and_update(client, rec)
        logger.info("Polled %s -> %s", key, new.status)
        return key, new

    outcomes = await asyncio.gather(*[_one(k, r) for k, r in records.items()])
    return {k: r for k, r in outcomes}


async def retry_failed_once(
    client: RunningHubClient,
    records: dict[str, TaskRecord],
    cases_by_id: dict[str, Case],
) -> dict[str, TaskRecord]:
    failed_keys = [k for k, r in records.items() if r.status == TaskStatus.FAILED]
    if not failed_keys:
        return records

    logger.info("Retrying %d failed task(s) once...", len(failed_keys))

    async def _retry(key: str) -> tuple[str, TaskRecord]:
        old = records[key]
        model_id, case_id = key.split("::", 1)
        try:
            new = await _submit_one(client, model_id, cases_by_id[case_id])
            logger.info("Resubmitted %s -> %s", key, new.task_id)
            return key, new
        except (RuntimeError, httpx.HTTPError) as exc:
            logger.error("Resubmit failed for %s: %s", key, exc)
            return key, old

    outcomes = await asyncio.gather(*[_retry(k) for k in failed_keys])
    updated = dict(records)
    for k, r in outcomes:
        updated[k] = r

    return await poll_all(client, updated)


# ---------------------------------------------------------------- download


async def download_all(
    records: dict[str, TaskRecord], run_root: Path
) -> dict[str, list[Path]]:
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    saved: dict[str, list[Path]] = {k: [] for k in records}

    async def _one_result(
        key: str, idx: int, url: str, dest_dir: Path, client: httpx.AsyncClient
    ) -> tuple[str, Path | None]:
        async with sem:
            try:
                dest = dest_dir / f"output-{idx:03d}.png"
                actual = await download_image(url, dest, client=client)
                return key, actual
            except RuntimeError as exc:
                logger.error("Download failed for %s #%d: %s", key, idx, exc)
                return key, None

    async with httpx.AsyncClient(timeout=60) as dl_client:
        coros = []
        for key, rec in records.items():
            if rec.status != TaskStatus.SUCCESS or not rec.results:
                continue
            model_id, case_id = key.split("::", 1)
            dest_dir = _case_dir(run_root, model_id, case_id)
            for idx, result in enumerate(rec.results, 1):
                url = result.get("url", "")
                if not url:
                    continue
                coros.append(_one_result(key, idx, url, dest_dir, dl_client))

        outcomes = await asyncio.gather(*coros) if coros else []

    for key, path in outcomes:
        if path is not None:
            saved[key].append(path)
    return saved


# ----------------------------------------------------------------- report


def _img_tag(rel_path: str) -> str:
    return f'<img src="{rel_path}" loading="lazy" />'


def render_html(
    run_root: Path,
    records: dict[str, TaskRecord],
    saved: dict[str, list[Path]],
    models: list[str],
) -> Path:
    """Build a self-contained HTML grid (rows=cases, cols=models)."""
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0e0e10; color: #eee; margin: 0; padding: 24px; }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    .meta { color: #888; font-size: 13px; margin-bottom: 24px; }
    .meta code { background: #1c1c1f; padding: 2px 6px; border-radius: 4px; color: #ddd; }
    table { border-collapse: separate; border-spacing: 8px; width: 100%; }
    th, td { vertical-align: top; padding: 0; }
    th.case-header { text-align: left; min-width: 220px; max-width: 280px;
                     background: #1a1a1d; color: #ddd; padding: 12px;
                     border-radius: 6px; font-weight: 500; font-size: 13px; }
    th.case-header .zh { font-size: 15px; font-weight: 600; color: #fff; }
    th.case-header .en { color: #aaa; margin-top: 2px; }
    th.case-header .desc { color: #777; font-size: 11px; margin-top: 8px;
                           line-height: 1.4; max-height: 90px; overflow-y: auto; }
    th.model-header { background: #1a1a1d; color: #fff; padding: 10px;
                      border-radius: 6px; font-size: 14px; text-align: center; }
    td.cell { background: #16161a; border-radius: 6px; padding: 8px;
              min-width: 240px; }
    td.cell .imgs { display: grid; grid-template-columns: repeat(2, 1fr);
                    gap: 4px; }
    td.cell .imgs.single { grid-template-columns: 1fr; }
    td.cell img { width: 100%; height: auto; border-radius: 4px;
                  display: block; background: #000; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 11px; font-weight: 600; }
    .badge.failed { background: #3a1414; color: #ff7070; }
    .badge.empty { background: #2a2a2e; color: #888; }
    .status-line { font-size: 11px; color: #777; margin-top: 6px; }
    .task-id { font-family: ui-monospace, monospace; font-size: 10px;
               color: #555; word-break: break-all; }
    """

    head = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>Spider-Verse 模型对比</title>
<style>{css}</style></head>
<body>
<h1>Spider-Verse 触发词 × 多模型对比</h1>
<div class="meta">
  <div>Trigger: <code>{_html_escape(TRIGGER_PHRASE)}</code></div>
  <div>Run: <code>{run_root.name}</code> · Generated at {utcnow_iso()}</div>
</div>
<table>
<thead><tr><th class="case-header" style="background:transparent">Case ↓ &nbsp; Model →</th>
"""
    parts = [head]
    for m in models:
        mc = get_model(m)
        parts.append(f'<th class="model-header">{_html_escape(mc.name)}<br/><span style="color:#888;font-size:11px">{m}</span></th>')
    parts.append("</tr></thead><tbody>")

    for case in CASES:
        parts.append("<tr>")
        parts.append(
            f'<th class="case-header"><div class="zh">{_html_escape(case.label_zh)} '
            f'<span style="color:#888;font-size:11px;font-weight:400">({case.aspect_ratio})</span></div>'
            f'<div class="en">{_html_escape(case.label_en)}</div>'
            f'<div class="desc">{_html_escape(case.description)}</div></th>'
        )
        for m in models:
            key = f"{m}::{case.id}"
            rec = records.get(key)
            paths = saved.get(key, [])
            parts.append('<td class="cell">')
            if rec is None:
                parts.append('<span class="badge empty">no record</span>')
            elif rec.status == TaskStatus.FAILED:
                parts.append('<span class="badge failed">FAILED</span>')
                if rec.error_message:
                    parts.append(
                        f'<div class="status-line">{_html_escape(rec.error_message[:200])}</div>'
                    )
            elif rec.status == TaskStatus.SUCCESS and paths:
                cls = "imgs single" if len(paths) == 1 else "imgs"
                parts.append(f'<div class="{cls}">')
                for p in paths:
                    rel = p.relative_to(run_root).as_posix()
                    parts.append(_img_tag(rel))
                parts.append("</div>")
            elif rec.status == TaskStatus.SUCCESS:
                parts.append('<span class="badge empty">SUCCESS (no images)</span>')
            else:
                parts.append(
                    f'<span class="badge empty">{_html_escape(str(rec.status))}</span>'
                )
            if rec is not None and rec.task_id:
                parts.append(f'<div class="task-id">{rec.task_id}</div>')
            parts.append("</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></body></html>")
    out = run_root / "report.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# -------------------------------------------------------------------- main


async def run(run_name: str, models: list[str]) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    run_root = _run_dir(repo_root, run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", run_root)

    _save_config(run_root, models)

    api_key = os.environ.get("RUNNINGHUB_API_KEY", "")
    if not api_key:
        raise SystemExit("RUNNINGHUB_API_KEY not set in environment.")

    cases_by_id = {c.id: c for c in CASES}

    async with RunningHubClient(api_key=api_key) as client:
        logger.info("=== Submitting %d tasks (%d models × %d cases) ===",
                    len(models) * len(CASES), len(models), len(CASES))
        records = await submit_all(client, models)
        _save_tasks_index(run_root, records)

        logger.info("=== Polling ===")
        records = await poll_all(client, records)
        _save_tasks_index(run_root, records)

        logger.info("=== Retrying failed (once) ===")
        records = await retry_failed_once(client, records, cases_by_id)
        _save_tasks_index(run_root, records)

        logger.info("=== Downloading images ===")
        saved = await download_all(records, run_root)

    n_success = sum(1 for r in records.values() if r.status == TaskStatus.SUCCESS)
    n_failed = sum(1 for r in records.values() if r.status == TaskStatus.FAILED)
    n_imgs = sum(len(v) for v in saved.values())
    logger.info(
        "Done. success=%d failed=%d images_downloaded=%d",
        n_success, n_failed, n_imgs,
    )

    report_path = render_html(run_root, records, saved, models)
    logger.info("Report: %s", report_path)
    return report_path


def _default_run_name() -> str:
    return "spiderverse-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("STYLECLAW_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default=_default_run_name(),
        help="Folder name under data/batch/ (default: spiderverse-<UTC-ts>)",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model ids. Default: {','.join(DEFAULT_MODELS)}",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        sys.exit(
            f"Unknown model(s): {unknown}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    asyncio.run(run(args.run_name, models))


if __name__ == "__main__":
    main()
