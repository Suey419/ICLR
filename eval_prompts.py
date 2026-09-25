DIAGNOSIS_MATCH_SYSTEM_PROMPT = (
    "You are a strict senior rare pulmonary disease diagnosis evaluator. "
    "Judge semantic medical equivalence. Do not reward vague broad categories when the gold answer is specific."
)

DIAGNOSIS_LIST_MATCH_PROMPT = """Compare predicted diagnoses with gold diagnoses.

Strict rules:
1. Match only the same clinically meaningful disease entity.
2. Accept synonyms, abbreviations, Chinese/English variants, and equivalent disease names.
3. Do not match concept expansion or narrowing.
4. One gold item can be matched at most once.

Output valid JSON only:
{{
  "matched_pairs": [
    {{"predicted": "predicted item", "gold": "gold item", "is_match": true, "reason": "brief reason"}}
  ],
  "unmatched_predicted": ["predicted item"],
  "unmatched_gold": ["gold item"]
}}

Predicted diagnoses:
{predicted_list}

Gold diagnoses:
{gold_list}
"""

EXAM_MATCH_SYSTEM_PROMPT = "You are a senior pulmonary medicine examination evaluator."

EXAM_LIST_MATCH_PROMPT = """Compare predicted medical examination items with gold examination items.

Matching rules:
1. Match names, aliases, abbreviations, or clinically equivalent exam groups.
2. Match if the predicted exam covers the same required clinical information.
3. Do not match clinically different exams.
4. One gold item can be matched at most once.

Output valid JSON only:
{{
  "matched_pairs": [
    {{"predicted": "predicted item", "gold": "gold item", "is_match": true, "reason": "brief reason"}}
  ],
  "unmatched_predicted": ["predicted item"],
  "unmatched_gold": ["gold item"]
}}

Predicted exams:
{predicted_list}

Gold exams:
{gold_list}
"""

TREATMENT_MATCH_SYSTEM_PROMPT = "You are a strict senior pulmonary treatment plan evaluator."

TREATMENT_LIST_MATCH_PROMPT = """Compare predicted treatment plan items with gold treatment items.

Rules:
1. Match if the predicted treatment has the same clinically specific meaning as the gold item.
2. Accept synonyms, abbreviations, drug name variants, and route wording differences.
3. Do not match a broad class to a specific treatment unless the gold item itself is broad.
4. One gold item can be matched at most once.

Output valid JSON only:
{{
  "matched_pairs": [
    {{"predicted": "predicted item", "gold": "gold item", "is_match": true, "reason": "brief reason"}}
  ],
  "unmatched_predicted": ["predicted item"],
  "unmatched_gold": ["gold item"]
}}

Predicted treatments:
{predicted_list}

Gold treatments:
{gold_list}
"""

RANK_SYSTEM_PROMPT = "You are a specialist in rare pulmonary diseases. Return only one of: 1, 2, 3, 4, 5, No."

RANK_PROMPT = """Judge whether the gold diagnosis appears in the predicted diagnosis sequence.
Return the rank of the first medically equivalent predicted diagnosis from 1 to 5.
Return "No" if there is no match.

Predicted diagnosis sequence:
{predicted_list}

Gold diagnosis:
{gold_diagnosis}
"""

BOND_SYSTEM_PROMPT = (
    "You are a strict senior rare pulmonary disease differential diagnosis evaluator. "
    "Use the Bond 1-5 differential diagnosis grading scale. Output valid JSON only."
)

BOND_PROMPT = """Evaluate whether the predicted final diagnosis matches the gold final diagnosis using the Bond 1-5 scale.

Bond score definitions:
5 = DDx includes the correct diagnosis.
4 = DDx contains something very close, but not an exact match.
3 = DDx contains something closely related and might have been helpful.
2 = DDx contains something related, but unlikely to be helpful.
1 = Nothing in the DDx is related to the correct diagnosis.

Output valid JSON only:
{{
  "bond_score": 1,
  "matched_prediction": "best matching predicted diagnosis or empty string",
  "reason": "brief reason"
}}

Predicted final diagnosis:
{predicted_list}

Gold final diagnosis:
{gold_diagnosis}
"""

OUTPUT8_SCORE_SYSTEM_PROMPT = (
    "You are a strict senior rare pulmonary disease final diagnosis and diagnostic-basis evaluator. "
    "Output valid JSON only."
)

OUTPUT8_SCORE_PROMPT = """Evaluate final diagnosis and diagnostic basis.

Gold answer:
{gold_final_dx}

Model prediction:
{pred_final_dx}

Gold diagnostic basis:
{gold_basis}

Predicted diagnostic basis:
{pred_basis}

Strictly output the following JSON:
{{
  "final_dx": {{
    "entity_match": 0,
    "hierarchical_distance": 0,
    "component_completeness": 0.0,
    "hallucination": 0,
    "key_components_in_gold": []
  }},
  "basis": {{
    "factual_accuracy": 0,
    "evidence_completeness": 0.0,
    "reasoning_efficiency": 1,
    "evidence_hallucination": 0
  }}
}}
"""

TREATMENT_QUALITY_SYSTEM_PROMPT = "You are a strict senior pulmonary treatment quality and safety evaluator. Output valid JSON only."

TREATMENT_QUALITY_PROMPT = """Evaluate treatment quality and safety.

Gold final diagnosis:
{gold_final_dx}

Gold treatment plan:
{gold_treatment}

Predicted treatment plan:
{pred_treatment}

Output valid JSON only:
{{
  "treatment_quality_score": 0,
  "safety_score": 0,
  "unsafe_or_hallucinated_treatment_count": 0,
  "reason": "brief reason"
}}
"""

REASONING_EFFICIENCY_SYSTEM_PROMPT = "You are a reliable assistant for the analysis of medical reasoning processes."

REASONING_EFFICIENCY_PROMPT = """Classify the current medical reasoning step into exactly one category.

Categories:
- Citation
- Repetition
- Reasoning
- Redundancy

Output only one word:
Citation | Repetition | Reasoning | Redundancy

Current reasoning step:
{current_step}

Previous reasoning steps:
{previous_steps}

Known patient record:
{case_context}

Final reasoning goal:
{goal}
"""

REASONING_FACTUALITY_SYSTEM_PROMPT = "You are a strict senior pulmonary medicine reasoning factuality evaluator. Output valid JSON only."

REASONING_FACTUALITY_PROMPT = """Judge whether this effective medical reasoning step is factually correct.

Output valid JSON only:
{{
  "judgment": "Correct",
  "reason": "brief reason"
}}

Allowed judgment values: Correct | Wrong

Patient record:
{case_context}

Gold/reference facts:
{gold_context}

Reasoning step to judge:
{reasoning_step}
"""

REASONING_COMPLETENESS_SYSTEM_PROMPT = "You are a reliable medical reasoning coverage evaluator."

REASONING_COMPLETENESS_PROMPT = """Determine whether the model reasoning covers the core meaning or logic of the gold reasoning step.

Output only:
Yes | No

Gold reasoning step:
{gold_step}

Model reasoning process:
{pred_reasoning}
"""
