from __future__ import annotations

import base64
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from styleclaw.core.models import TaskStatus
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "reports" / "templates"
_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


def _img_to_data_uri(path: Path) -> str:
    if not path.exists():
        logger.warning("Image not found for report: %s", path)
        return ""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    mt = mime.get(suffix.lstrip("."), "image/png")
    return f"data:{mt};base64,{data}"


def generate_model_select_report(name: str, pass_num: int = 1) -> Path:

    root = project_store.project_dir(name)
    config = project_store.load_config(name)
    analysis = project_store.load_analysis(name, pass_num=pass_num)
    evaluation = project_store.load_evaluation(name, pass_num=pass_num)

    ref_images = [
        _img_to_data_uri(root / r) for r in config.ref_images
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
        images = sorted(results_dir.glob("output-*.png"))
        model_data.append({
            "model": ev.model,
            "variant": ev.variant,
            "scores": ev.scores.model_dump(),
            "total": ev.total,
            "analysis": ev.analysis,
            "suggestions": ev.suggestions,
            "images": [_img_to_data_uri(p) for p in images],
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

    dest = project_store.model_select_dir(name, pass_num) / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Model-select report saved: %s", dest)
    return dest


def generate_style_refine_report(name: str, round_num: int) -> Path:

    root = project_store.project_dir(name)
    config = project_store.load_config(name)
    state = project_store.load_state(name)
    prompt_config = project_store.load_prompt_config(name, round_num)
    evaluation = project_store.load_round_evaluation(name, round_num)

    ref_images = [
        _img_to_data_uri(root / r) for r in config.ref_images
    ]

    model_data: list[dict] = []
    for ev in evaluation.evaluations:
        results_dir = project_store.round_results_dir(name, round_num, ev.model)
        images = sorted(results_dir.glob("output-*.png"))
        model_data.append({
            "model": ev.model,
            "scores": ev.scores.model_dump(),
            "total": ev.total,
            "analysis": ev.analysis,
            "images": [_img_to_data_uri(p) for p in images],
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

    dest = project_store.round_dir(name, round_num) / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Style-refine report saved: %s", dest)
    return dest


def generate_batch_t2i_report(name: str, batch_num: int) -> Path:

    config = project_store.load_batch_config(name, batch_num)
    records = project_store.load_all_batch_task_records(name, batch_num)

    cases_data: list[dict] = []
    for case in config.cases:
        case_dir = project_store.batch_t2i_case_dir(name, batch_num, case.id)
        images = sorted(case_dir.glob("output-*.png"))
        record = records.get(case.id)
        cases_data.append({
            "id": case.id,
            "category": case.category,
            "description": case.description,
            "aspect_ratio": case.aspect_ratio,
            "status": record.status if record else case.status,
            "images": [_img_to_data_uri(p) for p in images],
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

    dest = project_store.batch_t2i_dir(name, batch_num) / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Batch-t2i report saved: %s", dest)
    return dest


def generate_batch_i2i_report(name: str, batch_num: int) -> Path:

    uploads = project_store.load_i2i_uploads(name, batch_num)
    records = project_store.load_all_i2i_task_records(name, batch_num)

    pairs: list[dict] = []
    for i, upload in enumerate(uploads, 1):
        case_id = f"i2i-{i:03d}"
        case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
        source_path = project_store.batch_i2i_dir(name, batch_num) / "source-images" / Path(upload.local_path).name
        gen_images = sorted(case_dir.glob("output-*.png"))
        record = records.get(case_id)
        pairs.append({
            "case_id": case_id,
            "source": _img_to_data_uri(source_path),
            "generated": [_img_to_data_uri(p) for p in gen_images],
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

    dest = project_store.batch_i2i_dir(name, batch_num) / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Batch-i2i report saved: %s", dest)
    return dest
