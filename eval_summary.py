import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eval_utils import round_floats


def avg(case_reports: List[Dict[str, Any]], path: List[str]) -> Optional[float]:
    values = []
    for report in case_reports:
        current: Any = report
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)):
            values.append(float(current))
    return float(np.mean(values)) if values else None


def summarize_ddx(case_reports: List[Dict[str, Any]], output_key: str) -> Dict[str, Any]:
    return {
        "avg_required_precision": avg(case_reports, [output_key, "required_precision"]),
        "avg_required_recall": avg(case_reports, [output_key, "required_recall"]),
        "avg_required_f1": avg(case_reports, [output_key, "required_f1"]),
        "avg_all_precision": avg(case_reports, [output_key, "all_precision"]),
        "avg_all_recall": avg(case_reports, [output_key, "all_recall"]),
        "avg_all_f1": avg(case_reports, [output_key, "all_f1"]),
        "golden_first_accuracy": avg(case_reports, [output_key, "golden_first_accuracy"]),
        "golden_first_recall_at_1": avg(case_reports, [output_key, "golden_first_recall_at_1"]),
        "golden_first_recall_at_3": avg(case_reports, [output_key, "golden_first_recall_at_3"]),
        "golden_first_recall_at_5": avg(case_reports, [output_key, "golden_first_recall_at_5"]),
        "median_golden_first_rank": float(np.median([
            report[output_key]["golden_first_rank"] for report in case_reports
        ])) if case_reports else None,
    }


def summarize_exam(case_reports: List[Dict[str, Any]], output_key: str) -> Dict[str, Any]:
    return {
        "avg_required_precision": avg(case_reports, [output_key, "required_precision"]),
        "avg_required_recall": avg(case_reports, [output_key, "required_recall"]),
        "avg_required_f1": avg(case_reports, [output_key, "required_f1"]),
        "avg_all_precision": avg(case_reports, [output_key, "all_precision"]),
        "avg_all_recall": avg(case_reports, [output_key, "all_recall"]),
        "avg_all_f1": avg(case_reports, [output_key, "all_f1"]),
        "avg_extra_count": avg(case_reports, [output_key, "extra_count"]),
        "avg_extra_rate": avg(case_reports, [output_key, "extra_rate"]),
    }


def summarize_reasoning_reports(case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ["output_2", "output_3", "output_5", "output_6", "output_8", "output_9", "stage1", "stage2", "stage3", "overall"]:
        summary[key] = {
            "efficiency": avg(case_reports, ["reasoning_score_summary", key, "efficiency"]),
            "factuality": avg(case_reports, ["reasoning_score_summary", key, "factuality"]),
            "completeness": avg(case_reports, ["reasoning_score_summary", key, "completeness"]),
            "reasoning_score": avg(case_reports, ["reasoning_score_summary", key, "reasoning_score"]),
        }
    return summary


def summarize_reports(case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranks = [report["output_8"]["final_dx_rank"] for report in case_reports]
    return {
        "question_count": len(case_reports),
        "output_2_metrics_summary": summarize_ddx(case_reports, "output_2"),
        "output_3_metrics_summary": summarize_exam(case_reports, "output_3"),
        "output_5_metrics_summary": summarize_ddx(case_reports, "output_5"),
        "output_6_metrics_summary": summarize_exam(case_reports, "output_6"),
        "output_8_metrics_summary": {
            "final_dx_accuracy": avg(case_reports, ["output_8", "final_dx_accuracy"]),
            "final_dx_clinically_useful_rate": avg(case_reports, ["output_8", "final_dx_clinically_useful"]),
            "avg_final_dx_bond_score": avg(case_reports, ["output_8", "final_dx_bond_score"]),
            "final_dx_recall_at_1": avg(case_reports, ["output_8", "final_dx_recall_at_1"]),
            "final_dx_recall_at_3": avg(case_reports, ["output_8", "final_dx_recall_at_3"]),
            "final_dx_recall_at_5": avg(case_reports, ["output_8", "final_dx_recall_at_5"]),
            "median_rank": float(np.median(ranks)) if ranks else None,
            "avg_structured_score": avg(case_reports, ["output_8", "structured_score"]),
        },
        "output_9_metrics_summary": {
            "avg_treatment_precision": avg(case_reports, ["output_9", "treatment_precision"]),
            "avg_treatment_recall": avg(case_reports, ["output_9", "treatment_recall"]),
            "avg_treatment_f1": avg(case_reports, ["output_9", "treatment_f1"]),
            "avg_treatment_quality_score": avg(case_reports, ["output_9", "treatment_quality_score"]),
            "avg_safety_score": avg(case_reports, ["output_9", "safety_score"]),
            "avg_unsafe_or_hallucinated_treatment_count": avg(case_reports, ["output_9", "unsafe_or_hallucinated_treatment_count"]),
        },
    }


def build_report_payload(generation_payload: Dict[str, Any], generation_file: Path, eval_log_path: Path, args, case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_reports(case_reports)
    return {
        "provider": generation_payload.get("provider"),
        "model_name": generation_payload.get("model_name"),
        "image_input_mode": generation_payload.get("image_input_mode", "text"),
        "generation_file": str(generation_file),
        "generation_log_file": generation_payload.get("generation_log_file"),
        "eval_log_file": str(eval_log_path),
        "judge_mode": args.judge_mode,
        **summary,
        "metric_notes": {
            "output_2": "Initial differential diagnosis: required/all precision-recall-F1 plus top-k recall@1/3/5 for the first gold diagnosis in this output.",
            "output_3": "Initial recommended exams: required/all precision-recall-F1 and extra exam rate; no top-k.",
            "output_5": "Refined differential diagnosis: same metrics as output_2.",
            "output_6": "Further exams: same metrics as output_3.",
            "output_8": "Final diagnosis correctness evaluates only the final_diagnosis field with Bond 1-5 grading; final_dx_accuracy is the Bond=5 exact-correct rate, and final_dx_clinically_useful_rate is the Bond>=4 clinically useful rate. Recall@1/3/5 and rank are retained for this single final diagnosis; structured_score follows entity/basis rubric.",
            "output_9": "Treatment plan uses all-item precision-recall-F1 plus quality and safety scores.",
        },
        "case_reports": case_reports,
    }


def build_score_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "provider",
        "model_name",
        "image_input_mode",
        "generation_file",
        "generation_log_file",
        "eval_log_file",
        "judge_mode",
        "question_count",
        "output_2_metrics_summary",
        "output_3_metrics_summary",
        "output_5_metrics_summary",
        "output_6_metrics_summary",
        "output_8_metrics_summary",
        "output_9_metrics_summary",
        "metric_notes",
    ]
    return {key: report.get(key) for key in keys}



def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(round_floats(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
