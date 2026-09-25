import json
from typing import Any, Dict, List, Optional, Tuple

from eval_prompts import (
    REASONING_COMPLETENESS_PROMPT,
    REASONING_COMPLETENESS_SYSTEM_PROMPT,
    REASONING_EFFICIENCY_PROMPT,
    REASONING_EFFICIENCY_SYSTEM_PROMPT,
    REASONING_FACTUALITY_PROMPT,
    REASONING_FACTUALITY_SYSTEM_PROMPT,
)
from eval_utils import (
    SECTION_2,
    SECTION_3,
    SECTION_5,
    SECTION_6,
    SECTION_8,
    SECTION_9,
    as_list,
    get_stage,
    mean_optional,
    parse_json_object_response,
    text_list,
)


def truncate_text(text: Any, limit: int = 8000) -> str:
    rendered = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
    rendered = str(rendered)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + "\n...[truncated]"


def normalize_efficiency_category(raw: str) -> str:
    text = str(raw or "").strip()
    for category in ["Citation", "Repetition", "Reasoning", "Redundancy"]:
        if category.lower() in text.lower():
            return category
    return "Redundancy"


def simple_json_judgment(raw: str) -> Dict[str, Any]:
    parsed = parse_json_object_response(raw)
    judgment = str(parsed.get("judgment", "") or "").strip()
    if judgment.lower() == "correct":
        judgment = "Correct"
    elif judgment.lower() == "wrong":
        judgment = "Wrong"
    else:
        judgment = "Wrong"
    return {
        "judgment": judgment,
        "reason": str(parsed.get("reason", "") or ""),
        "parse_error": parsed.get("parse_error"),
        "raw_response": parsed.get("raw_response"),
    }


def ddx_reasoning_steps(items: Any) -> List[str]:
    steps = []
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        label = f"[{name}] " if name else ""
        for key, prefix in [
            ("reason_for", "Support"),
            ("supporting_evidence", "Support"),
            ("evidence", "Support"),
            ("reason_against", "Against"),
            ("opposing_evidence", "Against"),
            ("exclusion_reasons", "Against"),
        ]:
            for reason in text_list(item.get(key, []), key):
                steps.append(f"{label}{prefix}: {reason}")
    return steps


def exam_reasoning_steps(items: Any) -> List[str]:
    steps = []
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        exam = str(item.get("exam", "") or "").strip()
        label = f"[{exam}] " if exam else ""
        for key in ["reason", "why_order_this_exam", "basis_in_stage3_input"]:
            value = item.get(key)
            if isinstance(value, list):
                for reason in text_list(value, key):
                    steps.append(f"{label}Reason: {reason}")
            elif str(value or "").strip():
                steps.append(f"{label}Reason: {str(value).strip()}")
    return steps


def plain_reasoning_steps(items: Any, prefix: str = "") -> List[str]:
    return [f"{prefix}{item}".strip() for item in text_list(items, "basis")]


def treatment_gold_steps(section9: Any) -> List[str]:
    if not isinstance(section9, dict):
        return []
    steps = []
    for item in text_list(section9.get("treatment_plan", []), "treatment"):
        steps.append(f"Treatment plan: {item}")
    for item in text_list(section9.get("treatment_response", []), "response"):
        steps.append(f"Treatment response/basis: {item}")
    return steps


def reasoning_inputs_for_output(output_key: str, pred: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[List[str], List[str], str]:
    raw_gold = gold.get("raw_gold", {})
    if output_key == "output_2":
        return ddx_reasoning_steps(pred["output_2"]), ddx_reasoning_steps(raw_gold.get(SECTION_2, {}).get("differential_diagnoses", [])), "initial differential diagnosis"
    if output_key == "output_3":
        return exam_reasoning_steps(pred["output_3"]), exam_reasoning_steps(raw_gold.get(SECTION_3, {}).get("recommended_exams", [])), "initial examination recommendation"
    if output_key == "output_5":
        return ddx_reasoning_steps(pred["output_5"]), ddx_reasoning_steps(raw_gold.get(SECTION_5, [])), "refined differential diagnosis"
    if output_key == "output_6":
        return exam_reasoning_steps(pred["output_6"]), exam_reasoning_steps(raw_gold.get(SECTION_6, {}).get("recommended_exams", [])), "further examination recommendation"
    if output_key == "output_8":
        return (
            plain_reasoning_steps(pred["output_8"].get("diagnostic_basis", [])),
            plain_reasoning_steps(raw_gold.get(SECTION_8, {}).get("diagnostic_basis", [])),
            "final diagnosis rationale",
        )
    if output_key == "output_9":
        return (
            plain_reasoning_steps(pred["output_9"].get("treatment_basis", [])),
            treatment_gold_steps(raw_gold.get(SECTION_9, {})),
            "treatment planning rationale",
        )
    return [], [], output_key


def evaluate_reasoning_metrics(
    pred_steps: List[str],
    gold_steps: List[str],
    case_context: str,
    gold_context: str,
    goal: str,
    handler,
) -> Dict[str, Any]:
    if handler is None:
        return {
            "efficiency": None,
            "factuality": None,
            "completeness": None,
            "reasoning_score": None,
            "reasoning_step_count": len(pred_steps),
            "effective_reasoning_step_count": None,
            "correct_reasoning_step_count": None,
            "gold_reasoning_step_count": len(gold_steps),
            "covered_gold_reasoning_step_count": None,
            "evaluated_steps": [],
            "ground_truth_steps": [],
            "reason": "not evaluated in exact mode",
        }

    evaluated_steps = []
    for idx, step in enumerate(pred_steps):
        previous_steps = "\n".join(pred_steps[:idx])
        efficiency_prompt = REASONING_EFFICIENCY_PROMPT.format(
            current_step=step,
            previous_steps=truncate_text(previous_steps, 3000),
            case_context=case_context,
            goal=goal,
        )
        raw_efficiency = handler.get_completion(REASONING_EFFICIENCY_SYSTEM_PROMPT, efficiency_prompt)
        efficiency = normalize_efficiency_category(raw_efficiency)

        factuality = None
        factuality_reason = ""
        factuality_raw = None
        factuality_parse_error = None
        if efficiency == "Reasoning":
            factuality_prompt = REASONING_FACTUALITY_PROMPT.format(
                case_context=case_context,
                gold_context=gold_context,
                reasoning_step=step,
            )
            raw_factuality = handler.get_completion(REASONING_FACTUALITY_SYSTEM_PROMPT, factuality_prompt)
            parsed_factuality = simple_json_judgment(raw_factuality)
            factuality = parsed_factuality["judgment"] == "Correct"
            factuality_reason = parsed_factuality.get("reason", "")
            factuality_raw = parsed_factuality.get("raw_response")
            factuality_parse_error = parsed_factuality.get("parse_error")

        evaluated_steps.append({
            "step": step,
            "efficiency": efficiency,
            "efficiency_raw_response": raw_efficiency,
            "factuality": factuality,
            "factuality_reason": factuality_reason,
            "factuality_parse_error": factuality_parse_error,
            "factuality_raw_response": factuality_raw,
        })

    effective_count = sum(1 for step in evaluated_steps if step["efficiency"] == "Reasoning")
    correct_count = sum(1 for step in evaluated_steps if step["factuality"] is True)
    efficiency_score = effective_count / len(evaluated_steps) if evaluated_steps else 0.0
    factuality_score = correct_count / effective_count if effective_count else 0.0

    ground_truth_steps = []
    pred_reasoning = "\n".join(pred_steps)
    for gold_step in gold_steps:
        prompt = REASONING_COMPLETENESS_PROMPT.format(
            gold_step=gold_step,
            pred_reasoning=truncate_text(pred_reasoning, 8000),
        )
        raw_hit = handler.get_completion(REASONING_COMPLETENESS_SYSTEM_PROMPT, prompt)
        hit = "yes" in str(raw_hit).lower()
        ground_truth_steps.append({
            "step": gold_step,
            "hit": hit,
            "raw_response": raw_hit,
        })
    covered_count = sum(1 for step in ground_truth_steps if step["hit"])
    completeness_score = covered_count / len(ground_truth_steps) if ground_truth_steps else None
    reasoning_score = mean_optional([efficiency_score, factuality_score, completeness_score])

    return {
        "efficiency": efficiency_score,
        "factuality": factuality_score,
        "completeness": completeness_score,
        "reasoning_score": reasoning_score,
        "reasoning_step_count": len(pred_steps),
        "effective_reasoning_step_count": effective_count,
        "correct_reasoning_step_count": correct_count,
        "gold_reasoning_step_count": len(gold_steps),
        "covered_gold_reasoning_step_count": covered_count if ground_truth_steps else None,
        "evaluated_steps": evaluated_steps,
        "ground_truth_steps": ground_truth_steps,
    }


def evaluate_case_reasoning(pred: Dict[str, Any], gold: Dict[str, Any], handler, case: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    raw_gold = gold.get("raw_gold", {})
    stages = case.get("stages", {}) if isinstance(case, dict) else {}
    case_context = truncate_text({
        "stage1_prompt": get_stage(stages, "stage1_outputs_2_3").get("user_prompt", ""),
        "stage2_prompt": get_stage(stages, "stage2_outputs_5_6").get("user_prompt", ""),
        "stage3_prompt": get_stage(stages, "stage3_outputs_8_9").get("user_prompt", ""),
    }, 8000)
    gold_context = truncate_text(raw_gold, 8000)
    metrics = {}
    for output_key in ["output_2", "output_3", "output_5", "output_6", "output_8", "output_9"]:
        pred_steps, gold_steps, goal = reasoning_inputs_for_output(output_key, pred, gold)
        metrics[output_key] = evaluate_reasoning_metrics(
            pred_steps=pred_steps,
            gold_steps=gold_steps,
            case_context=case_context,
            gold_context=gold_context,
            goal=goal,
            handler=handler,
        )
    return metrics


def summarize_case_reasoning(reasoning_by_output: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    output_summary = {
        output_key: {
            "efficiency": metrics.get("efficiency"),
            "factuality": metrics.get("factuality"),
            "completeness": metrics.get("completeness"),
            "reasoning_score": metrics.get("reasoning_score"),
        }
        for output_key, metrics in reasoning_by_output.items()
    }

    def avg_outputs(keys: List[str], metric: str) -> Optional[float]:
        return mean_optional([output_summary.get(key, {}).get(metric) for key in keys])

    stage_map = {
        "stage1": ["output_2", "output_3"],
        "stage2": ["output_5", "output_6"],
        "stage3": ["output_8", "output_9"],
    }
    stage_summary = {
        stage: {
            "efficiency": avg_outputs(keys, "efficiency"),
            "factuality": avg_outputs(keys, "factuality"),
            "completeness": avg_outputs(keys, "completeness"),
            "reasoning_score": avg_outputs(keys, "reasoning_score"),
        }
        for stage, keys in stage_map.items()
    }
    overall = {
        metric: mean_optional([stage_summary[stage].get(metric) for stage in ["stage1", "stage2", "stage3"]])
        for metric in ["efficiency", "factuality", "completeness", "reasoning_score"]
    }
    return {
        **output_summary,
        **stage_summary,
        "overall": overall,
    }

