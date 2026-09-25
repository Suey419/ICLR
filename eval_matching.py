import re
from typing import Any, Dict, List, Optional, Tuple

from eval_prompts import (
    BOND_PROMPT,
    BOND_SYSTEM_PROMPT,
    DIAGNOSIS_LIST_MATCH_PROMPT,
    DIAGNOSIS_MATCH_SYSTEM_PROMPT,
    EXAM_LIST_MATCH_PROMPT,
    EXAM_MATCH_SYSTEM_PROMPT,
    OUTPUT8_SCORE_PROMPT,
    OUTPUT8_SCORE_SYSTEM_PROMPT,
    RANK_PROMPT,
    RANK_SYSTEM_PROMPT,
    TREATMENT_LIST_MATCH_PROMPT,
    TREATMENT_MATCH_SYSTEM_PROMPT,
    TREATMENT_QUALITY_PROMPT,
    TREATMENT_QUALITY_SYSTEM_PROMPT,
)
from eval_utils import (
    clamp_number,
    format_plain_list,
    gold_texts,
    normalize_text,
    parse_json_object_response,
    text_list,
)


def exact_match_result(predicted: List[str], gold: List[str]) -> Dict[str, Any]:
    used_gold = set()
    matched_pairs = []
    unmatched_predicted = []
    for pred in predicted:
        pred_norm = normalize_text(pred)
        match = None
        for gold_item in gold:
            if gold_item in used_gold:
                continue
            if pred_norm == normalize_text(gold_item):
                match = gold_item
                break
        if match is None:
            unmatched_predicted.append(pred)
        else:
            used_gold.add(match)
            matched_pairs.append({"predicted": pred, "gold": match, "is_match": True, "reason": "exact normalized string match"})
    return {
        "matched_pairs": matched_pairs,
        "unmatched_predicted": unmatched_predicted,
        "unmatched_gold": [item for item in gold if item not in used_gold],
        "parse_error": None,
        "raw_response": None,
    }


def parse_match_response(raw: Optional[str], predicted: List[str], gold: List[str]) -> Dict[str, Any]:
    parsed = parse_json_object_response(raw)
    if parsed.get("parse_error"):
        return {
            "matched_pairs": [],
            "unmatched_predicted": predicted,
            "unmatched_gold": gold,
            "parse_error": parsed.get("parse_error"),
            "raw_response": raw,
        }
    matched_pairs = parsed.get("matched_pairs", [])
    if not isinstance(matched_pairs, list):
        matched_pairs = []
    unmatched_predicted = parsed.get("unmatched_predicted", predicted)
    unmatched_gold = parsed.get("unmatched_gold", gold)
    return {
        "matched_pairs": matched_pairs,
        "unmatched_predicted": unmatched_predicted if isinstance(unmatched_predicted, list) else predicted,
        "unmatched_gold": unmatched_gold if isinstance(unmatched_gold, list) else gold,
        "parse_error": None,
        "raw_response": raw,
    }


def llm_list_match(predicted: List[str], gold: List[str], handler, system_prompt: str, prompt_template: str) -> Dict[str, Any]:
    if handler is None:
        return exact_match_result(predicted, gold)
    prompt = prompt_template.format(
        predicted_list=format_plain_list(predicted),
        gold_list=format_plain_list(gold),
    )
    raw = handler.get_completion(system_prompt, prompt)
    return parse_match_response(raw, predicted, gold)


def compute_prf(hit_pred_count: int, pred_count: int, hit_gold_count: int, gold_count: int) -> Dict[str, float]:
    precision = hit_pred_count / pred_count if pred_count else 0.0
    recall = hit_gold_count / gold_count if gold_count else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def matched_pairs(match_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        pair for pair in match_result.get("matched_pairs", [])
        if isinstance(pair, dict) and pair.get("is_match") is True
    ]


def subset_metrics(predicted: List[str], gold: List[Dict[str, str]], match_result: Dict[str, Any], requirement: Optional[str]) -> Dict[str, Any]:
    pairs = matched_pairs(match_result)
    gold_subset = gold_texts(gold, requirement)
    gold_subset_norm = {normalize_text(item) for item in gold_subset}
    hit_pred_norm = set()
    hit_gold_norm = set()
    for pair in pairs:
        pred = str(pair.get("predicted", "")).strip()
        gold_text = str(pair.get("gold", "")).strip()
        if not pred or not gold_text:
            continue
        if requirement is None or normalize_text(gold_text) in gold_subset_norm:
            hit_pred_norm.add(normalize_text(pred))
            hit_gold_norm.add(normalize_text(gold_text))
    metrics = compute_prf(len(hit_pred_norm), len(predicted), len(hit_gold_norm), len(gold_subset))
    return {
        **metrics,
        "hit_pred_count": len(hit_pred_norm),
        "hit_gold_count": len(hit_gold_norm),
        "gold_count": len(gold_subset),
        "pred_count": len(predicted),
    }


def recall_at_k_for_gold(gold_item: str, predicted: List[str], match_result: Dict[str, Any], k: int) -> float:
    if not gold_item:
        return 0.0
    target = normalize_text(gold_item)
    top_pred = {normalize_text(item) for item in predicted[:k]}
    for pair in matched_pairs(match_result):
        if normalize_text(pair.get("gold", "")) == target and normalize_text(pair.get("predicted", "")) in top_pred:
            return 1.0
    return 0.0


def rank_for_gold(gold_item: str, predicted: List[str], match_result: Dict[str, Any]) -> int:
    if not gold_item:
        return 11
    target = normalize_text(gold_item)
    matched_pred = {
        normalize_text(pair.get("predicted", ""))
        for pair in matched_pairs(match_result)
        if normalize_text(pair.get("gold", "")) == target
    }
    for idx, pred in enumerate(predicted[:5], start=1):
        if normalize_text(pred) in matched_pred:
            return idx
    return 11


def normalize_rank(raw: Optional[str]) -> Tuple[int, str]:
    text = "" if raw is None else str(raw).strip()
    if not text or "no" in text.lower():
        return 11, "No"
    match = re.search(r"\b([1-5])\b", text)
    if match:
        rank = int(match.group(1))
        return rank, str(rank)
    return 11, text


def judge_rank_deeprare(predicted: List[str], gold_diagnosis: str, handler) -> Dict[str, Any]:
    if not gold_diagnosis or not predicted:
        return {"rank": 11, "raw_rank": "No", "raw_response": None}
    if handler is None:
        rank = 11
        for idx, pred in enumerate(predicted[:5], start=1):
            if normalize_text(pred) == normalize_text(gold_diagnosis):
                rank = idx
                break
        return {"rank": rank, "raw_rank": str(rank) if rank <= 5 else "No", "raw_response": None}
    prompt = RANK_PROMPT.format(
        predicted_list=format_plain_list(predicted[:5]),
        gold_diagnosis=gold_diagnosis,
    )
    raw = handler.get_completion(RANK_SYSTEM_PROMPT, prompt)
    rank, raw_rank = normalize_rank(raw)
    return {"rank": rank, "raw_rank": raw_rank, "raw_response": raw}


def judge_bond_score(predicted: List[str], gold_diagnosis: str, handler) -> Dict[str, Any]:
    if not gold_diagnosis or not predicted:
        return {"bond_score": 1, "matched_prediction": "", "reason": "missing prediction or gold diagnosis", "raw_response": None}
    if handler is None:
        gold_norm = normalize_text(gold_diagnosis)
        matched = next((pred for pred in predicted[:5] if normalize_text(pred) == gold_norm), "")
        return {
            "bond_score": 5 if matched else 1,
            "matched_prediction": matched,
            "reason": "exact normalized match" if matched else "no exact normalized match in exact mode",
            "raw_response": None,
        }
    prompt = BOND_PROMPT.format(
        predicted_list=format_plain_list(predicted[:5]),
        gold_diagnosis=gold_diagnosis,
    )
    raw = handler.get_completion(BOND_SYSTEM_PROMPT, prompt)
    parsed = parse_json_object_response(raw)
    try:
        bond_score = int(float(parsed.get("bond_score", 1)))
    except Exception:
        bond_score = 1
    bond_score = max(1, min(5, bond_score))
    return {
        "bond_score": bond_score,
        "matched_prediction": str(parsed.get("matched_prediction", "") or ""),
        "reason": str(parsed.get("reason", "") or ""),
        "parse_error": parsed.get("parse_error"),
        "raw_response": parsed.get("raw_response"),
    }


def score_rank_metrics(rank: int, prefix: str = "") -> Dict[str, Any]:
    name = f"{prefix}_" if prefix else ""
    return {
        f"{name}accuracy": 1 if rank == 1 else 0,
        f"{name}recall_at_1": 1 if rank <= 1 else 0,
        f"{name}recall_at_3": 1 if rank <= 3 else 0,
        f"{name}recall_at_5": 1 if rank <= 5 else 0,
        f"{name}rank": rank,
    }


def score_ddx_output(predicted_items: Any, gold: List[Dict[str, str]], handler) -> Dict[str, Any]:
    predicted = text_list(predicted_items, "name")
    gold_all = gold_texts(gold)
    match_result = llm_list_match(predicted, gold_all, handler, DIAGNOSIS_MATCH_SYSTEM_PROMPT, DIAGNOSIS_LIST_MATCH_PROMPT)

    required = subset_metrics(predicted, gold, match_result, "required")
    all_metrics = subset_metrics(predicted, gold, match_result, None)
    golden_first = gold_all[0] if gold_all else ""
    first_rank = rank_for_gold(golden_first, predicted, match_result)

    return {
        "predicted": predicted,
        "gold_all": gold_all,
        "gold_required": gold_texts(gold, "required"),
        "required_precision": required["precision"],
        "required_recall": required["recall"],
        "required_f1": required["f1"],
        "all_precision": all_metrics["precision"],
        "all_recall": all_metrics["recall"],
        "all_f1": all_metrics["f1"],
        "golden_first_dx": golden_first,
        "golden_first_accuracy": 1 if first_rank == 1 else 0,
        "golden_first_recall_at_1": 1 if first_rank <= 1 else 0,
        "golden_first_recall_at_3": 1 if first_rank <= 3 else 0,
        "golden_first_recall_at_5": 1 if first_rank <= 5 else 0,
        "golden_first_rank": first_rank,
        "matched_pairs": match_result.get("matched_pairs", []),
        "unmatched_predicted": match_result.get("unmatched_predicted", []),
        "unmatched_gold": match_result.get("unmatched_gold", []),
        "match_parse_error": match_result.get("parse_error"),
        "match_raw_response": match_result.get("raw_response"),
    }


def score_exam_output(predicted_items: Any, gold: List[Dict[str, str]], handler) -> Dict[str, Any]:
    predicted = text_list(predicted_items, "exam")
    gold_all = gold_texts(gold)
    match_result = llm_list_match(predicted, gold_all, handler, EXAM_MATCH_SYSTEM_PROMPT, EXAM_LIST_MATCH_PROMPT)
    required = subset_metrics(predicted, gold, match_result, "required")
    all_metrics = subset_metrics(predicted, gold, match_result, None)
    extra_count = len([
        item for item in match_result.get("unmatched_predicted", [])
        if str(item).strip()
    ])
    return {
        "predicted": predicted,
        "gold_all": gold_all,
        "gold_required": gold_texts(gold, "required"),
        "required_precision": required["precision"],
        "required_recall": required["recall"],
        "required_f1": required["f1"],
        "all_precision": all_metrics["precision"],
        "all_recall": all_metrics["recall"],
        "all_f1": all_metrics["f1"],
        "extra_count": extra_count,
        "extra_rate": extra_count / len(predicted) if predicted else 0.0,
        "matched_pairs": match_result.get("matched_pairs", []),
        "unmatched_predicted": match_result.get("unmatched_predicted", []),
        "unmatched_gold": match_result.get("unmatched_gold", []),
        "match_parse_error": match_result.get("parse_error"),
        "match_raw_response": match_result.get("raw_response"),
    }


def output8_score_from_structured(parsed: Dict[str, Any]) -> float:
    final_dx = parsed.get("final_dx", {}) if isinstance(parsed.get("final_dx"), dict) else {}
    basis = parsed.get("basis", {}) if isinstance(parsed.get("basis"), dict) else {}
    entity_match = float(final_dx.get("entity_match", 0) or 0)
    hierarchical_distance = float(final_dx.get("hierarchical_distance", 3) or 3)
    component_completeness = float(final_dx.get("component_completeness", 0) or 0)
    hallucination = float(final_dx.get("hallucination", 1) or 0)
    factual_accuracy = float(basis.get("factual_accuracy", 0) or 0)
    evidence_completeness = float(basis.get("evidence_completeness", 0) or 0)
    reasoning_efficiency = float(basis.get("reasoning_efficiency", 1) or 1)
    evidence_hallucination = float(basis.get("evidence_hallucination", 1) or 0)
    score = (
        max(0.0, min(2.0, entity_match)) / 2.0 * 30.0
        + (3.0 - max(0.0, min(3.0, hierarchical_distance))) / 3.0 * 15.0
        + max(0.0, min(1.0, component_completeness)) * 15.0
        + (1.0 - max(0.0, min(1.0, hallucination))) * 10.0
        + max(0.0, min(1.0, factual_accuracy)) * 10.0
        + max(0.0, min(1.0, evidence_completeness)) * 15.0
        + max(1.0, min(3.0, reasoning_efficiency)) / 3.0 * 5.0
        - max(0.0, min(1.0, evidence_hallucination)) * 10.0
    )
    return max(0.0, min(100.0, score))


def score_output8(predicted: Dict[str, Any], gold_final_dx: str, gold_basis: List[str], handler) -> Dict[str, Any]:
    final_dx = str(predicted.get("final_diagnosis", "")).strip()
    pred_sequence = [final_dx] if final_dx else []
    rank_result = judge_rank_deeprare(pred_sequence, gold_final_dx, handler)
    bond_result = judge_bond_score(pred_sequence, gold_final_dx, handler)
    pred_basis = text_list(predicted.get("diagnostic_basis", []), "basis")

    if handler is None:
        structured = {
            "final_dx": {
                "entity_match": 2 if rank_result["rank"] == 1 else 0,
                "hierarchical_distance": 0 if rank_result["rank"] == 1 else 3,
                "component_completeness": 1.0 if rank_result["rank"] == 1 else 0.0,
                "hallucination": 0,
                "key_components_in_gold": [],
            },
            "basis": {
                "factual_accuracy": 0,
                "evidence_completeness": 0.0,
                "reasoning_efficiency": 1,
                "evidence_hallucination": 0,
            },
            "parse_error": None,
            "raw_response": None,
        }
    else:
        prompt = OUTPUT8_SCORE_PROMPT.format(
            gold_final_dx=format_plain_list([gold_final_dx]),
            pred_final_dx=format_plain_list(pred_sequence),
            gold_basis=format_plain_list(gold_basis),
            pred_basis=format_plain_list(pred_basis),
        )
        raw = handler.get_completion(OUTPUT8_SCORE_SYSTEM_PROMPT, prompt)
        structured = parse_json_object_response(raw)

    structured_score = output8_score_from_structured(structured)
    return {
        "gold_final_dx": gold_final_dx,
        "predicted_sequence": pred_sequence,
        "predicted_basis": pred_basis,
        "final_dx_accuracy": 1 if bond_result["bond_score"] == 5 else 0,
        "final_dx_clinically_useful": 1 if bond_result["bond_score"] >= 4 else 0,
        "final_dx_bond_score": bond_result["bond_score"],
        "final_dx_bond_matched_prediction": bond_result.get("matched_prediction", ""),
        "final_dx_bond_reason": bond_result.get("reason", ""),
        "final_dx_bond_parse_error": bond_result.get("parse_error"),
        "final_dx_bond_raw_response": bond_result.get("raw_response"),
        "final_dx_recall_at_1": 1 if rank_result["rank"] <= 1 else 0,
        "final_dx_recall_at_3": 1 if rank_result["rank"] <= 3 else 0,
        "final_dx_recall_at_5": 1 if rank_result["rank"] <= 5 else 0,
        "final_dx_rank": rank_result["rank"],
        "final_dx_rank_raw": rank_result["raw_rank"],
        "final_dx_rank_raw_response": rank_result["raw_response"],
        "structured_score": structured_score,
        "structured_eval": structured,
    }


def score_output9(predicted: Dict[str, Any], gold: List[Dict[str, str]], gold_final_dx: str, handler) -> Dict[str, Any]:
    predicted_treatments = text_list(predicted.get("treatment_plan", []), "treatment")
    gold_treatments = gold_texts(gold)
    match_result = llm_list_match(
        predicted_treatments,
        gold_treatments,
        handler,
        TREATMENT_MATCH_SYSTEM_PROMPT,
        TREATMENT_LIST_MATCH_PROMPT,
    )
    all_metrics = subset_metrics(predicted_treatments, gold, match_result, None)

    if handler is None:
        quality = {
            "treatment_quality_score": None,
            "safety_score": None,
            "unsafe_or_hallucinated_treatment_count": None,
            "reason": "not evaluated in exact mode",
            "parse_error": None,
            "raw_response": None,
        }
    else:
        prompt = TREATMENT_QUALITY_PROMPT.format(
            gold_final_dx=gold_final_dx,
            gold_treatment=format_plain_list(gold_treatments),
            pred_treatment=format_plain_list(predicted_treatments),
        )
        raw = handler.get_completion(TREATMENT_QUALITY_SYSTEM_PROMPT, prompt)
        parsed = parse_json_object_response(raw)
        quality = {
            "treatment_quality_score": clamp_number(parsed.get("treatment_quality_score"), 0, 100),
            "safety_score": clamp_number(parsed.get("safety_score"), 0, 100),
            "unsafe_or_hallucinated_treatment_count": parsed.get("unsafe_or_hallucinated_treatment_count"),
            "reason": parsed.get("reason", ""),
            "parse_error": parsed.get("parse_error"),
            "raw_response": parsed.get("raw_response"),
        }

    return {
        "predicted": predicted_treatments,
        "gold_all": gold_treatments,
        "treatment_precision": all_metrics["precision"],
        "treatment_recall": all_metrics["recall"],
        "treatment_f1": all_metrics["f1"],
        "matched_pairs": match_result.get("matched_pairs", []),
        "unmatched_predicted": match_result.get("unmatched_predicted", []),
        "unmatched_gold": match_result.get("unmatched_gold", []),
        "match_parse_error": match_result.get("parse_error"),
        "match_raw_response": match_result.get("raw_response"),
        **quality,
    }

