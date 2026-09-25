import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_GENERATION_FILE = "generation/gpt-5.4/text_image/gpt-5.4_text_image_main_generation.json"
DEFAULT_REPORT_DIR = "eval"
DEFAULT_REPORT_TEMPLATE = "{model_name}_{input_mode}_report.json"
DEFAULT_SCORE_TEMPLATE = "{model_name}_{input_mode}_score.json"
DEFAULT_EVAL_LOG_TEMPLATE = "{model_name}_{input_mode}_eval_log.jsonl"

SECTION_1 = "1.patient_information_and_chief_complaint"
SECTION_2 = "2.initial_differential_diagnoses"
SECTION_3 = "3.initial_recommended_exams"
SECTION_4 = "4.exam_results_and_differential_evidence"
SECTION_5 = "5.updated_differential_diagnoses"
SECTION_6 = "6.follow_up_recommended_exams"
SECTION_7 = "7.key_confirmatory_exam_information"
SECTION_8 = "8.final_diagnosis_and_diagnostic_basis"
SECTION_9 = "9.treatment_plan"


def resolve_script_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def normalize_text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = value.strip().lower()
    return re.sub(r"[\s（）()\-_,，。:：/、；;]+", "", value)


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", str(value).strip(), flags=re.UNICODE).strip("._")
    return safe or "unknown"


def round_floats(value, digits: int = 4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    return value


def format_plain_list(items: List[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1) if str(item).strip())


def parse_json_object_response(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None:
        return {"parse_error": "empty response", "raw_response": raw}
    text = str(raw).strip()
    candidates = [text]
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.rfind("```")
        if end > start:
            candidates.append(text[start:end].strip())
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                candidates.append(chunk)

    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                parsed["parse_error"] = None
                parsed["raw_response"] = raw
                return parsed
        except Exception as exc:
            last_error = str(exc)
    return {"parse_error": last_error or "invalid json", "raw_response": raw}


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def item_text(item: Any, key: str) -> str:
    if isinstance(item, dict):
        return str(item.get(key, item.get("name", item.get("exam", "")))).strip()
    return str(item).strip()


def text_list(items: Any, key: str) -> List[str]:
    values = []
    for item in as_list(items):
        text = item_text(item, key)
        if text:
            values.append(text)
    return values


def gold_items(items: Any, key: str, default_requirement: str = "required") -> List[Dict[str, str]]:
    parsed = []
    for item in as_list(items):
        text = item_text(item, key)
        if not text:
            continue
        requirement = default_requirement
        if isinstance(item, dict):
            requirement = str(item.get("requirement", default_requirement)).strip() or default_requirement
        parsed.append({"text": text, "requirement": requirement})
    return parsed


def gold_texts(items: List[Dict[str, str]], requirement: Optional[str] = None) -> List[str]:
    if requirement is None:
        return [item["text"] for item in items]
    return [item["text"] for item in items if item.get("requirement") == requirement]


def get_stage(stages: Dict[str, Any], name: str) -> Dict[str, Any]:
    payload = stages.get(name, {}) if isinstance(stages, dict) else {}
    return payload if isinstance(payload, dict) else {}


def parsed_stage(stages: Dict[str, Any], name: str) -> Dict[str, Any]:
    parsed = get_stage(stages, name).get("parsed_response", {})
    return parsed if isinstance(parsed, dict) else {}


def clamp_number(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = lower
    return max(lower, min(upper, number))


def mean_optional(values: List[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None
