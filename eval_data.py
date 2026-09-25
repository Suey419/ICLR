from typing import Any, Dict

from eval_utils import (
    SECTION_2,
    SECTION_3,
    SECTION_5,
    SECTION_6,
    SECTION_8,
    SECTION_9,
    gold_items,
    parsed_stage,
    text_list,
)


def collect_case_outputs(case: Dict[str, Any]) -> Dict[str, Any]:
    stages = case.get("stages", {})
    stage1 = parsed_stage(stages, "stage1_outputs_2_3")
    stage2 = parsed_stage(stages, "stage2_outputs_5_6")
    stage3 = parsed_stage(stages, "stage3_outputs_8_9")
    output8 = stage3.get("final_diagnosis_and_diagnostic_basis", {})
    output9 = stage3.get("treatment_plan", {})
    return {
        "output_2": stage1.get("initial_differential_diagnoses", []),
        "output_3": stage1.get("initial_recommended_exams", []),
        "output_5": stage2.get("updated_differential_diagnoses", []),
        "output_6": stage2.get("follow_up_recommended_exams", []),
        "output_8": output8 if isinstance(output8, dict) else {},
        "output_9": output9 if isinstance(output9, dict) else {},
    }


def collect_case_gold(case: Dict[str, Any]) -> Dict[str, Any]:
    gold = case.get("gold", {})
    section2 = gold.get(SECTION_2, {})
    section3 = gold.get(SECTION_3, {})
    section5 = gold.get(SECTION_5, [])
    section6 = gold.get(SECTION_6, {})
    section8 = gold.get(SECTION_8, {})
    section9 = gold.get(SECTION_9, {})

    output2_items = section2.get("differential_diagnoses", []) if isinstance(section2, dict) else []
    output3_items = section3.get("recommended_exams", []) if isinstance(section3, dict) else []
    output6_items = section6.get("recommended_exams", []) if isinstance(section6, dict) else []
    final_dx = section8.get("final_diagnosis", "") if isinstance(section8, dict) else ""
    basis = section8.get("diagnostic_basis", []) if isinstance(section8, dict) else []
    treatment = section9.get("treatment_plan", []) if isinstance(section9, dict) else []

    return {
        "output_2": gold_items(output2_items, "name"),
        "output_3": gold_items(output3_items, "exam"),
        "output_5": gold_items(section5, "name"),
        "output_6": gold_items(output6_items, "exam"),
        "output_8_final_dx": str(final_dx).strip(),
        "output_8_basis": text_list(basis, "basis"),
        "output_9": gold_items(treatment, "treatment", default_requirement="all"),
        "raw_gold": gold,
    }
