import json
from typing import Any

from generation_pipeline import (
    SECTION_1,
    SECTION_4,
    SECTION_7,
    format_block,
)


SYSTEM_PROMPT = (
    "You are a rigorous senior attending physician in pulmonary and critical care medicine. "
    "Use only the information available at the current stage. Do not use later-stage information "
    "that has not been provided. Do not fabricate clinical details. Write all output values in "
    "English using standard medical terminology. Output only valid JSON."
)


def prompt_stage1(case_id: str, sections: dict[str, Any]) -> str:
    return f"""Based only on Section 1 of the case, generate the initial differential diagnoses and recommended examinations.

Task requirements:
1. Use only the information in "1. Patient Information and Chief Complaint".
2. Output "initial_differential_diagnoses": a ranked list. Each item must include rank, name, reason_for, and reason_against.
3. Output "initial_recommended_exams": recommended examinations. Each item must include exam and reason.
4. Do not output the final diagnosis or treatment plan.

Output JSON format:
{{
  "initial_differential_diagnoses": [
    {{"rank": 1, "name": "Disease name", "reason_for": ["Supporting evidence"], "reason_against": ["Opposing evidence or missing evidence"]}}
  ],
  "initial_recommended_exams": [
    {{"exam": "Recommended examination", "reason": "Reason for ordering the examination"}}
  ],
  "clinical_summary": "Brief summary of the current-stage clinical judgment"
}}

Case ID: {case_id}

{format_block(SECTION_1, sections[SECTION_1])}
"""


def prompt_stage2(case_id: str, sections: dict[str, Any], history_2: Any, history_3: Any) -> str:
    return f"""Based on Sections 1 and 4 of the case, plus the previous-stage differential diagnoses and examination recommendations, generate updated differential diagnoses and follow-up recommended examinations.

Task requirements:
1. Use the patient information, previous differential diagnoses, previous examination recommendations, and exam results.
2. Output "updated_differential_diagnoses": a ranked list. Each item must include rank, name, reason_for, and reason_against.
3. Output "follow_up_recommended_exams": further examinations with reasons.
4. Do not output the final diagnosis or treatment plan.

Output JSON format:
{{
  "updated_differential_diagnoses": [
    {{"rank": 1, "name": "Disease name", "reason_for": ["Supporting evidence"], "reason_against": ["Opposing evidence or missing evidence"]}}
  ],
  "follow_up_recommended_exams": [
    {{"exam": "Recommended follow-up examination", "reason": "How this examination helps confirm, exclude, or stratify the diagnosis"}}
  ],
  "clinical_summary": "Brief summary of how the new evidence changes the diagnostic reasoning"
}}

Case ID: {case_id}

{format_block(SECTION_1, sections[SECTION_1])}

[Previous Initial Differential Diagnoses]
{json.dumps(history_2, ensure_ascii=False, indent=2)}

[Previous Initial Recommended Examinations]
{json.dumps(history_3, ensure_ascii=False, indent=2)}

{format_block(SECTION_4, sections[SECTION_4])}
"""


def prompt_stage3(case_id: str, sections: dict[str, Any], history_2: Any, history_3: Any, history_5: Any, history_6: Any) -> str:
    return f"""Based on Sections 1, 4, and 7 of the case, plus previous-stage diagnostic and examination reasoning, generate the final diagnosis, diagnostic basis, and treatment plan.

Task requirements:
1. Use the patient information, previous diagnostic reasoning, exam results, follow-up recommendations, and key confirmatory information.
2. Output "final_diagnosis_and_diagnostic_basis": final_diagnosis, diagnostic_basis, differential_diagnoses, and summary.
3. Output "treatment_plan": treatment_plan, treatment_basis, and clinical_summary.
4. The treatment plan must be consistent with the final diagnosis and case evidence.

Output JSON format:
{{
  "final_diagnosis_and_diagnostic_basis": {{
    "final_diagnosis": "Final diagnosis",
    "diagnostic_basis": ["Diagnostic basis"],
    "differential_diagnoses": [
      {{"rank": 1, "name": "Disease name", "reason_for": ["Supporting evidence"], "reason_against": ["Opposing evidence"]}}
    ],
    "summary": "Diagnostic summary"
  }},
  "treatment_plan": {{
    "treatment_plan": ["Treatment item"],
    "treatment_basis": ["Treatment rationale"],
    "clinical_summary": "Treatment strategy summary"
  }}
}}

Case ID: {case_id}

{format_block(SECTION_1, sections[SECTION_1])}

[Previous Initial Differential Diagnoses]
{json.dumps(history_2, ensure_ascii=False, indent=2)}

[Previous Initial Recommended Examinations]
{json.dumps(history_3, ensure_ascii=False, indent=2)}

{format_block(SECTION_4, sections[SECTION_4])}

[Previous Updated Differential Diagnoses]
{json.dumps(history_5, ensure_ascii=False, indent=2)}

[Previous Follow-up Recommended Examinations]
{json.dumps(history_6, ensure_ascii=False, indent=2)}

{format_block(SECTION_7, sections[SECTION_7])}
"""
