from __future__ import annotations

from pathlib import Path

from styleclaw.core.models import Phase
from styleclaw.storage import project_store
from styleclaw.storage.image_store import list_output_images


def _media_url(name: str, path: Path) -> str:
    rel = path.relative_to(project_store.project_dir(name))
    return f"/media/{name}/{rel.as_posix()}"


def _ref_urls(name: str, config) -> list[str]:
    root = project_store.project_dir(name)
    urls = []
    for r in config.ref_images:
        p = root / r
        if p.exists():
            urls.append(_media_url(name, p))
    return urls


def build_gallery(
    name: str,
    *,
    phase: str | None = None,
    pass_num: int | None = None,
    round_num: int | None = None,
    batch_num: int | None = None,
) -> dict:
    """Return a JSON-serializable gallery for the requested (or current) slice.

    The shared ``trigger`` phrase is surfaced once at the top (like report.html);
    each group's ``caption`` carries only the part that VARIES across images
    (test subject / case description), not the full repeated prompt.
    """
    state = project_store.load_state(name)
    config = project_store.load_config(name)
    refs = _ref_urls(name, config)
    groups: list[dict] = []
    trigger: str | None = None

    eff_phase = Phase(phase) if phase else state.phase
    eff_pass = pass_num if pass_num is not None else (state.current_model_select_pass or 1)
    eff_round = round_num if round_num is not None else state.current_round
    eff_batch = batch_num if batch_num is not None else state.current_batch

    if eff_phase == Phase.MODEL_SELECT:
        pass_num = eff_pass
        try:
            evaluation = project_store.load_evaluation(name, pass_num=pass_num)
        except FileNotFoundError:
            evaluation = None
        # Trigger + per-gender test subjects come from the pass's analysis.
        from styleclaw.scripts.generate import resolve_test_subjects
        try:
            analysis = project_store.load_analysis(name, pass_num=pass_num)
            trigger = analysis.trigger_phrase or None
            subjects = resolve_test_subjects(analysis.test_subjects)
        except FileNotFoundError:
            subjects = {}
        # Evaluation scores keyed by (model, BASE variant) — e.g. "mj-v7/prompt-sref".
        # Task-record keys carry gender suffix ("mj-v7/prompt-sref-male").
        # Strip suffix when joining.
        score_by_key: dict[str, dict] = {}
        if evaluation is not None:
            for e in evaluation.evaluations:
                k = f"{e.model}/{e.variant}" if e.variant else e.model
                score_by_key[k] = {"total": e.total, **e.scores.model_dump()}
        records = project_store.load_all_task_records(name, pass_num=pass_num)
        for rec_key in sorted(records):
            caption = None
            if "/" in rec_key:
                model_id, variant = rec_key.split("/", 1)
                results_dir = project_store.model_results_dir(
                    name, model_id, variant=variant, pass_num=pass_num,
                )
                base_variant = variant
                for gender in ("male", "female"):
                    if base_variant.endswith(f"-{gender}"):
                        base_variant = base_variant[: -len(gender) - 1]
                        caption = subjects.get(gender)
                        break
                score_key = f"{model_id}/{base_variant}"
            else:
                results_dir = project_store.model_results_dir(name, rec_key, pass_num=pass_num)
                score_key = rec_key
            imgs = list_output_images(results_dir) if results_dir.exists() else []
            groups.append({
                "label": rec_key,
                "images": [_media_url(name, p) for p in imgs],
                "scores": score_by_key.get(score_key),
                "caption": caption,
            })

    elif eff_phase == Phase.STYLE_REFINE:
        pass_num = eff_pass
        round_num = eff_round
        try:
            prompt_config = project_store.load_prompt_config(name, round_num, pass_num=pass_num)
            trigger = prompt_config.trigger_phrase or None
        except FileNotFoundError:
            pass
        try:
            evaluation = project_store.load_round_evaluation(name, round_num, pass_num=pass_num)
        except FileNotFoundError:
            evaluation = None
        records = project_store.load_all_round_task_records(name, round_num, pass_num=pass_num)
        score_by_model = {}
        if evaluation is not None:
            score_by_model = {
                e.model: {"total": e.total, **e.scores.model_dump()}
                for e in evaluation.evaluations
            }
        for mid in sorted(records):
            results_dir = project_store.round_results_dir(name, round_num, mid, pass_num=pass_num)
            imgs = list_output_images(results_dir) if results_dir.exists() else []
            groups.append({
                "label": mid,
                "images": [_media_url(name, p) for p in imgs],
                "scores": score_by_model.get(mid),
                "caption": None,
            })

    elif eff_phase == Phase.BATCH_T2I:
        batch_num = eff_batch
        try:
            batch_config = project_store.load_batch_config(name, batch_num)
        except FileNotFoundError:
            batch_config = None
        if batch_config is not None:
            trigger = batch_config.trigger_phrase or None
            for case in batch_config.cases:
                case_dir = project_store.batch_t2i_case_dir(name, batch_num, case.id)
                imgs = list_output_images(case_dir) if case_dir.exists() else []
                groups.append({
                    "label": f"{case.id} · {case.category}",
                    "images": [_media_url(name, p) for p in imgs],
                    "scores": None,
                    "caption": case.description or None,
                })

    elif eff_phase == Phase.BATCH_I2I:
        batch_num = eff_batch
        try:
            prompt_config = project_store.load_prompt_config(name, eff_round, pass_num=eff_pass)
            trigger = prompt_config.trigger_phrase or None
        except FileNotFoundError:
            pass
        uploads = project_store.load_i2i_uploads(name, batch_num)
        for i, _upload in enumerate(uploads, 1):
            case_id = f"i2i-{i:03d}"
            case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
            imgs = list_output_images(case_dir) if case_dir.exists() else []
            groups.append({
                "label": case_id,
                "images": [_media_url(name, p) for p in imgs],
                "scores": None,
                "caption": None,
            })

    return {
        "phase": eff_phase.value,
        "pass": eff_pass,
        "round": eff_round,
        "batch": eff_batch,
        "trigger": trigger,
        "ref_images": refs,
        "groups": groups,
    }
