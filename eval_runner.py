import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from eval_case import evaluate_case
from eval_summary import build_report_payload, build_score_summary, save_json
from eval_utils import resolve_script_path, safe_path_name
from llm_clients import LLMHandler


THREAD_LOCAL = threading.local()


def get_thread_handler(args):
    llm = getattr(THREAD_LOCAL, "llm", None)
    if llm is None:
        llm = LLMHandler(args, enabled=args.judge_mode == "llm", purpose="evaluation")
        THREAD_LOCAL.llm = llm
    return llm.handler


def write_eval_log(log_handle, entry: Dict[str, Any]) -> None:
    log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_handle.flush()


def load_existing_case_reports(report_path: Path) -> List[Dict[str, Any]]:
    if not report_path.exists():
        return []
    try:
        with report_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    reports = payload.get("case_reports", [])
    return reports if isinstance(reports, list) else []


def sort_case_reports(case_reports: List[Dict[str, Any]], case_order: Dict[str, int]) -> List[Dict[str, Any]]:
    return sorted(
        case_reports,
        key=lambda report: case_order.get(str(report.get("case_id")), 10**9),
    )


def evaluate_case_worker(case: Dict[str, Any], args) -> Dict[str, Any]:
    return evaluate_case(case, get_thread_handler(args))


def evaluate_generation_file(args) -> Path:
    generation_file = resolve_script_path(args.generation_file)
    llm = LLMHandler(args, enabled=args.judge_mode == "llm", purpose="evaluation")
    with generation_file.open("r", encoding="utf-8") as f:
        generation_payload = json.load(f)

    model_name = generation_payload.get("model_name", llm.model_name)
    input_mode = generation_payload.get("image_input_mode", "text")
    output_dir = resolve_script_path(args.report_dir) / safe_path_name(model_name) / safe_path_name(input_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / args.report_template.format(
        model_name=safe_path_name(model_name),
        provider=generation_payload.get("provider", args.model),
        input_mode=safe_path_name(input_mode),
    )
    score_path = output_dir / args.score_template.format(
        model_name=safe_path_name(model_name),
        provider=generation_payload.get("provider", args.model),
        input_mode=safe_path_name(input_mode),
    )
    eval_log_path = output_dir / args.eval_log_template.format(
        model_name=safe_path_name(model_name),
        provider=generation_payload.get("provider", args.model),
        input_mode=safe_path_name(input_mode),
    )

    case_reports = [] if args.force_recompute else load_existing_case_reports(report_path)
    evaluated = {report.get("case_id") for report in case_reports}
    all_cases = generation_payload.get("cases", [])
    if args.max_cases and args.max_cases > 0:
        all_cases = all_cases[:args.max_cases]
    case_order = {str(case.get("case_id")): idx for idx, case in enumerate(all_cases)}
    workers = max(1, int(args.workers or 1))
    with eval_log_path.open("a", encoding="utf-8") as log_handle:
        write_eval_log(log_handle, {
            "event": "start",
            "generation_file": str(generation_file),
            "judge_mode": args.judge_mode,
            "force_recompute": args.force_recompute,
            "resume_evaluated_case_count": len(evaluated),
            "workers": workers,
            "max_cases": args.max_cases,
        })
        pending_cases = []
        for case in all_cases:
            case_id = case.get("case_id")
            if case_id in evaluated:
                print(f"[skip evaluated] {case_id}")
                continue
            pending_cases.append(case)

        if workers == 1:
            for case in pending_cases:
                case_id = case.get("case_id")
                try:
                    print(f"[evaluating] {case_id}", flush=True)
                    report = evaluate_case(case, llm.handler)
                    case_reports.append(report)
                    case_reports = sort_case_reports(case_reports, case_order)
                    evaluated.add(case_id)
                    write_eval_log(log_handle, {"event": "evaluated", "case_id": case_id, "case_report": report})
                    full_report = build_report_payload(generation_payload, generation_file, eval_log_path, args, case_reports)
                    save_json(report_path, full_report)
                    save_json(score_path, build_score_summary(full_report))
                    print(f"[evaluated] {case_id}", flush=True)
                except Exception as exc:
                    write_eval_log(log_handle, {
                        "event": "error",
                        "case_id": case_id,
                        "source_file": case.get("source_file"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print(f"[eval error] {case_id}: {type(exc).__name__}: {exc}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_case = {}
                for case in pending_cases:
                    case_id = case.get("case_id")
                    print(f"[evaluating] {case_id}", flush=True)
                    future = executor.submit(evaluate_case_worker, case, args)
                    future_to_case[future] = case

                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    case_id = case.get("case_id")
                    try:
                        report = future.result()
                        case_reports.append(report)
                        case_reports = sort_case_reports(case_reports, case_order)
                        evaluated.add(case_id)
                        write_eval_log(log_handle, {"event": "evaluated", "case_id": case_id, "case_report": report})
                        full_report = build_report_payload(generation_payload, generation_file, eval_log_path, args, case_reports)
                        save_json(report_path, full_report)
                        save_json(score_path, build_score_summary(full_report))
                        print(f"[evaluated] {case_id}", flush=True)
                    except Exception as exc:
                        write_eval_log(log_handle, {
                            "event": "error",
                            "case_id": case_id,
                            "source_file": case.get("source_file"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                        print(f"[eval error] {case_id}: {type(exc).__name__}: {exc}", flush=True)
        write_eval_log(log_handle, {"event": "finish", "case_count": len(case_reports)})

    case_reports = sort_case_reports(case_reports, case_order)
    full_report = build_report_payload(generation_payload, generation_file, eval_log_path, args, case_reports)
    save_json(report_path, full_report)
    save_json(score_path, build_score_summary(full_report))
    print("Saved report:", report_path)
    print("Saved score summary:", score_path)
    print("Saved eval log:", eval_log_path)
    print("Question count:", full_report["question_count"])
    return report_path
