import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from llm_clients import LLMHandler


SECTION_1 = "1.patient_information_and_chief_complaint"
SECTION_2 = "2.initial_differential_diagnoses"
SECTION_3 = "3.initial_recommended_exams"
SECTION_4 = "4.exam_results_and_differential_evidence"
SECTION_5 = "5.updated_differential_diagnoses"
SECTION_6 = "6.follow_up_recommended_exams"
SECTION_7 = "7.key_confirmatory_exam_information"
SECTION_8 = "8.final_diagnosis_and_diagnostic_basis"
SECTION_9 = "9.treatment_plan"

OUTPUT_SECTION_KEYS = [SECTION_2, SECTION_3, SECTION_5, SECTION_6, SECTION_8, SECTION_9]
REQUIRED_SECTION_KEYS = [SECTION_1, SECTION_2, SECTION_3, SECTION_4, SECTION_5, SECTION_6, SECTION_7, SECTION_8, SECTION_9]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_GENERATE_DIR = "generation"


class GenerationError(Exception):
    pass


def resolve_script_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", str(value).strip(), flags=re.UNICODE).strip("._")
    return safe or "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def get_sections(case_payload: dict[str, Any]) -> dict[str, Any]:
    sections = case_payload.get("sections", {})
    if not isinstance(sections, dict):
        raise GenerationError("case sections is missing or not an object")
    return sections


def strip_images(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_images(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_images(item) for key, item in value.items() if key != "images"}
    return value


def collect_image_urls(*values: Any) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("url")
            if isinstance(url, str) and url and url not in seen:
                urls.append(url)
                seen.add(url)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for value in values:
        walk(value)
    return urls


def format_block(title: str, value: Any) -> str:
    clean_value = strip_images(value)
    if isinstance(clean_value, dict) and set(clean_value.keys()) == {"text"}:
        body = str(clean_value.get("text", "")).strip()
    else:
        body = json.dumps(clean_value, ensure_ascii=False, indent=2)
    return f"[{title}]\n{body}"


def parse_json_response(raw_response: Optional[str]) -> Optional[dict[str, Any]]:
    if raw_response is None:
        return None
    text = str(raw_response).strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    extracted = extract_first_json_object(text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def write_log(log_handle: Optional[TextIO | list[dict[str, Any]]], entry: dict[str, Any]) -> None:
    if log_handle is None:
        return
    if isinstance(log_handle, list):
        log_handle.append(entry)
        return
    log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_handle.flush()


def validate_case(case_payload: dict[str, Any]) -> list[str]:
    try:
        sections = get_sections(case_payload)
    except GenerationError as exc:
        return [str(exc)]
    return [key for key in REQUIRED_SECTION_KEYS if key not in sections]


def gold_outputs(case_payload: dict[str, Any]) -> dict[str, Any]:
    sections = get_sections(case_payload)
    return {key: sections.get(key) for key in OUTPUT_SECTION_KEYS}


def make_case_entry(case_payload: dict[str, Any], case_file: Path) -> dict[str, Any]:
    return {
        "case_id": case_payload.get("case_id", case_file.stem),
        "source_file": case_payload.get("source_file", case_file.name),
        "source_url": case_payload.get("source_url", ""),
        "gold": gold_outputs(case_payload),
        "stages": {},
    }


def generate_stage(
    case_id: str,
    stage_name: str,
    system_prompt: str,
    user_prompt: str,
    image_urls: list[str],
    llm: LLMHandler,
    log_handle: Optional[TextIO | list[dict[str, Any]]],
) -> dict[str, Any]:
    raw_response = llm.handler.get_completion(system_prompt, user_prompt, image_urls=image_urls)
    if raw_response is None or not str(raw_response).strip():
        raise GenerationError(f"{stage_name} model output is empty")
    parsed_response = parse_json_response(raw_response)
    if parsed_response is None:
        raise GenerationError(f"{stage_name} response is not valid JSON")
    generation = {
        "stage": stage_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_urls": image_urls,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "parse_error": None,
    }
    write_log(log_handle, {"case_id": case_id, **generation})
    return generation


def load_existing_generation(output_path: Path, stage_order: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    payload = load_json(output_path)
    cases = payload.get("cases", [])
    errors = payload.get("errors", [])
    if not isinstance(cases, list):
        cases = []
    if not isinstance(errors, list):
        errors = []
    complete_cases = [
        case for case in cases
        if isinstance(case.get("stages"), dict)
        and all(isinstance(case["stages"].get(stage, {}).get("parsed_response"), dict) for stage in stage_order)
    ]
    return complete_cases, errors


def build_arg_parser(description: str, default_filename_template: str, default_log_template: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--generate_dir", type=str, default=DEFAULT_GENERATE_DIR)
    parser.add_argument("--filename_template", type=str, default=default_filename_template)
    parser.add_argument("--log_template", type=str, default=default_log_template)
    parser.add_argument("--image_input_mode", type=str, default="text", choices=["text", "text_image"])
    parser.add_argument("--history_source", type=str, default="generated", choices=["generated", "gold"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", type=str, default="openai", choices=["openai", "gemini", "deepseek", "claude"])
    parser.add_argument("--sk", type=str, default="")
    parser.add_argument("--apikey", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--openai_apikey", type=str, default="")
    parser.add_argument("--openai_base_url", type=str, default="")
    parser.add_argument("--openai_model", type=str, default="gpt-4o")
    parser.add_argument("--gemini_apikey", type=str, default="")
    parser.add_argument("--gemini_base_url", type=str, default="")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--claude_apikey", type=str, default="")
    parser.add_argument("--claude_base_url", type=str, default="")
    parser.add_argument("--claude_model", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--deepseek_apikey", type=str, default="")
    parser.add_argument("--deepseek_base_url", type=str, default="")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    return parser


def generate_case_worker(case_file: Path, args, build_case_stages: Callable) -> dict[str, Any]:
    memory_log: list[dict[str, Any]] = []
    case_id = case_file.stem
    try:
        case_payload = load_json(case_file)
        case_id = case_payload.get("case_id", case_file.stem)
        validation_errors = validate_case(case_payload)
        if validation_errors:
            return {
                "status": "error",
                "case_id": case_id,
                "error_entry": {
                    "case_id": case_id,
                    "source_file": case_file.name,
                    "error_type": "invalid_case_payload",
                    "errors": validation_errors,
                },
                "log_entries": [],
            }
        llm = LLMHandler(args, purpose="generation")
        case_entry = make_case_entry(case_payload, case_file)
        case_entry["stages"] = build_case_stages(args, llm, case_payload, memory_log)
        return {"status": "success", "case_id": case_id, "case_entry": case_entry, "log_entries": memory_log}
    except Exception as exc:
        return {
            "status": "error",
            "case_id": case_id,
            "error_entry": {
                "case_id": case_id,
                "source_file": case_file.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "log_entries": memory_log,
        }


def build_output_payload(args, llm: LLMHandler, experiment_name: str, dataset_dir: Path, log_path: Path, stage_order: list[str], cases: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "provider": args.model,
        "model_name": llm.model_name,
        "dataset_dir": str(dataset_dir),
        "generation_log_file": str(log_path),
        "image_input_mode": args.image_input_mode,
        "history_source": args.history_source,
        "case_count": len(cases),
        "error_count": len(errors),
        "stage_order": stage_order,
        "errors": errors,
        "cases": cases,
    }


def run_generation(args, experiment_name: str, stage_order: list[str], build_case_stages: Callable) -> Path:
    dataset_dir = resolve_script_path(args.dataset_dir)
    generate_dir = resolve_script_path(args.generate_dir)
    llm = LLMHandler(args, purpose="generation")
    output_dir = generate_dir / safe_path_name(llm.model_name) / safe_path_name(args.image_input_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename_template.format(
        model_name=safe_path_name(llm.model_name),
        provider=args.model,
        input_mode=safe_path_name(args.image_input_mode),
        experiment_name=experiment_name,
    )
    log_path = output_dir / args.log_template.format(
        model_name=safe_path_name(llm.model_name),
        provider=args.model,
        input_mode=safe_path_name(args.image_input_mode),
        experiment_name=experiment_name,
    )

    cases, errors = load_existing_generation(output_path, stage_order)
    completed_case_ids = {case.get("case_id") for case in cases}
    case_files = sorted(dataset_dir.glob("*.json"))
    case_order = {case_file.stem: idx for idx, case_file in enumerate(case_files)}
    workers = max(1, int(args.workers or 1))

    def sort_cases() -> None:
        cases.sort(key=lambda case: case_order.get(str(case.get("source_file", "")).replace(".json", ""), 10**9))

    with log_path.open("a", encoding="utf-8") as log_handle:
        write_log(log_handle, {"event": "start", "experiment_name": experiment_name, "model_name": llm.model_name, "workers": workers})
        pending = [path for path in case_files if path.stem not in completed_case_ids]
        if workers == 1:
            for case_file in pending:
                result = generate_case_worker(case_file, args, build_case_stages)
                for entry in result.get("log_entries", []):
                    write_log(log_handle, entry)
                if result["status"] == "success":
                    cases.append(result["case_entry"])
                    completed_case_ids.add(result["case_id"])
                    sort_cases()
                    print(f"[generated] {result['case_id']}", flush=True)
                else:
                    errors.append(result["error_entry"])
                    write_log(log_handle, {"event": "error", **result["error_entry"]})
                    print(f"[generation error] {result['case_id']}: {result['error_entry'].get('error_type')}", flush=True)
                save_json(output_path, build_output_payload(args, llm, experiment_name, dataset_dir, log_path, stage_order, cases, errors))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(generate_case_worker, case_file, args, build_case_stages) for case_file in pending]
                for future in as_completed(futures):
                    result = future.result()
                    for entry in result.get("log_entries", []):
                        write_log(log_handle, entry)
                    if result["status"] == "success":
                        cases.append(result["case_entry"])
                        completed_case_ids.add(result["case_id"])
                        sort_cases()
                        print(f"[generated] {result['case_id']}", flush=True)
                    else:
                        errors.append(result["error_entry"])
                        write_log(log_handle, {"event": "error", **result["error_entry"]})
                        print(f"[generation error] {result['case_id']}: {result['error_entry'].get('error_type')}", flush=True)
                    save_json(output_path, build_output_payload(args, llm, experiment_name, dataset_dir, log_path, stage_order, cases, errors))
        write_log(log_handle, {"event": "finish", "case_count": len(cases), "error_count": len(errors)})

    save_json(output_path, build_output_payload(args, llm, experiment_name, dataset_dir, log_path, stage_order, cases, errors))
    print("Saved generation file:", output_path)
    print("Saved generation log:", log_path)
    return output_path
