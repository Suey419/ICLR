from typing import Any, Dict

from eval_data import collect_case_gold, collect_case_outputs
from eval_matching import score_ddx_output, score_exam_output, score_output8, score_output9


def evaluate_case(case: Dict[str, Any], handler) -> Dict[str, Any]:
    pred = collect_case_outputs(case)
    gold = collect_case_gold(case)
    report = {
        "case_id": case.get("case_id"),
        "source_file": case.get("source_file"),
        "source_url": case.get("source_url", ""),
        "output_2": score_ddx_output(pred["output_2"], gold["output_2"], handler),
        "output_3": score_exam_output(pred["output_3"], gold["output_3"], handler),
        "output_5": score_ddx_output(pred["output_5"], gold["output_5"], handler),
        "output_6": score_exam_output(pred["output_6"], gold["output_6"], handler),
        "output_8": score_output8(pred["output_8"], gold["output_8_final_dx"], gold["output_8_basis"], handler),
        "output_9": score_output9(pred["output_9"], gold["output_9"], gold["output_8_final_dx"], handler),
        "gold": gold,
    }
    return report
