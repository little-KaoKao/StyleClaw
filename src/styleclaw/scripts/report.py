from __future__ import annotations

import logging
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from styleclaw.core.models import TaskStatus
from styleclaw.storage import project_store
from styleclaw.storage.image_store import list_output_images

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "reports" / "templates"
# autoescape=True is load-bearing: report HTML embeds LLM-generated text
# (trigger phrases, analysis, case descriptions) that must not be able to
# inject <script> or break out of attribute values. Do not add `| safe` to
# any LLM-sourced field without a dedicated sanitizer.
_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


def _relative_img_src(img_path: Path, report_dir: Path) -> str:
    """Return an HTML-usable `src` for an image file, expressed relative to the
    directory containing the report. Returns empty string when the file does
    not exist, so templates can skip missing images gracefully.

    Using relative paths (not `data:` URIs) keeps report HTML small — a batch
    of 400+ images would otherwise produce a 100+ MB file.
    """
    if not img_path.exists():
        logger.warning("Image not found for report: %s", img_path)
        return ""
    rel = os.path.relpath(img_path, report_dir)
    return rel.replace(os.sep, "/")


def generate_model_select_report(name: str, pass_num: int = 1) -> Path:

    root = project_store.project_dir(name)
    config = project_store.load_config(name)
    analysis = project_store.load_analysis(name, pass_num=pass_num)
    evaluation = project_store.load_evaluation(name, pass_num=pass_num)

    dest_dir = project_store.model_select_dir(name, pass_num)

    ref_images = [
        _relative_img_src(root / r, dest_dir) for r in config.ref_images
    ]

    model_data: list[dict] = []
    for ev in evaluation.evaluations:
        if ev.variant:
            results_dir = project_store.model_results_dir(
                name, ev.model, variant=ev.variant, pass_num=pass_num,
            )
        else:
            results_dir = project_store.model_results_dir(
                name, ev.model, pass_num=pass_num,
            )
        images = list_output_images(results_dir)
        model_data.append({
            "model": ev.model,
            "variant": ev.variant,
            "scores": ev.scores.model_dump(),
            "total": ev.total,
            "analysis": ev.analysis,
            "suggestions": ev.suggestions,
            "images": [_relative_img_src(p, dest_dir) for p in images],
        })

    template = _env.get_template("model_select.html")
    html = template.render(
        project_name=name,
        pass_num=pass_num,
        trigger_phrase=analysis.trigger_phrase,
        recommendation=evaluation.recommendation,
        recommended_variant=evaluation.recommended_variant,
        ref_images=ref_images,
        models=model_data,
    )

    dest = dest_dir / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Model-select report saved: %s", dest)
    return dest


def generate_style_refine_report(
    name: str, round_num: int, pass_num: int = 1,
) -> Path:

    root = project_store.project_dir(name)
    config = project_store.load_config(name)
    prompt_config = project_store.load_prompt_config(name, round_num, pass_num=pass_num)
    evaluation = project_store.load_round_evaluation(name, round_num, pass_num=pass_num)

    dest_dir = project_store.round_dir(name, round_num, pass_num=pass_num)

    ref_images = [
        _relative_img_src(root / r, dest_dir) for r in config.ref_images
    ]

    model_data: list[dict] = []
    for ev in evaluation.evaluations:
        results_dir = project_store.round_results_dir(
            name, round_num, ev.model, pass_num=pass_num,
        )
        images = list_output_images(results_dir)
        model_data.append({
            "model": ev.model,
            "scores": ev.scores.model_dump(),
            "total": ev.total,
            "analysis": ev.analysis,
            "images": [_relative_img_src(p, dest_dir) for p in images],
        })

    template = _env.get_template("style_refine.html")
    html = template.render(
        project_name=name,
        round_num=round_num,
        trigger_phrase=prompt_config.trigger_phrase,
        recommendation=evaluation.recommendation,
        ref_images=ref_images,
        models=model_data,
    )

    dest = dest_dir / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Style-refine report saved: %s", dest)
    return dest


def generate_batch_t2i_report(name: str, batch_num: int) -> Path:

    config = project_store.load_batch_config(name, batch_num)
    records = project_store.load_all_batch_task_records(name, batch_num)

    dest_dir = project_store.batch_t2i_dir(name, batch_num)

    cases_data: list[dict] = []
    for case in config.cases:
        case_dir = project_store.batch_t2i_case_dir(name, batch_num, case.id)
        images = list_output_images(case_dir)
        record = records.get(case.id)
        cases_data.append({
            "id": case.id,
            "category": case.category,
            "description": case.description,
            "aspect_ratio": case.aspect_ratio,
            "status": record.status if record else case.status,
            "images": [_relative_img_src(p, dest_dir) for p in images],
        })

    template = _env.get_template("batch_t2i.html")
    html = template.render(
        project_name=name,
        batch_num=batch_num,
        trigger_phrase=config.trigger_phrase,
        cases=cases_data,
        total=len(config.cases),
        completed=sum(1 for c in cases_data if c["status"] == TaskStatus.SUCCESS),
    )

    dest = dest_dir / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Batch-t2i report saved: %s", dest)
    return dest


def generate_batch_i2i_report(name: str, batch_num: int) -> Path:

    uploads = project_store.load_i2i_uploads(name, batch_num)
    records = project_store.load_all_i2i_task_records(name, batch_num)

    dest_dir = project_store.batch_i2i_dir(name, batch_num)

    pairs: list[dict] = []
    for i, upload in enumerate(uploads, 1):
        case_id = f"i2i-{i:03d}"
        case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
        source_path = dest_dir / "source-images" / Path(upload.local_path).name
        gen_images = list_output_images(case_dir)
        record = records.get(case_id)
        pairs.append({
            "case_id": case_id,
            "source": _relative_img_src(source_path, dest_dir),
            "generated": [_relative_img_src(p, dest_dir) for p in gen_images],
            "status": record.status if record else "pending",
        })

    template = _env.get_template("batch_i2i.html")
    html = template.render(
        project_name=name,
        batch_num=batch_num,
        pairs=pairs,
        total=len(uploads),
        completed=sum(1 for p in pairs if p["status"] == TaskStatus.SUCCESS),
    )

    dest = dest_dir / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Batch-i2i report saved: %s", dest)
    return dest
