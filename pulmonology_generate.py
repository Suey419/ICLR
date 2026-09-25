from typing import Any

from generation_pipeline import (
    SECTION_1,
    SECTION_2,
    SECTION_3,
    SECTION_4,
    SECTION_5,
    SECTION_6,
    SECTION_7,
    build_arg_parser,
    collect_image_urls,
    generate_stage,
    get_sections,
    run_generation,
)
from generation_prompts import SYSTEM_PROMPT, prompt_stage1, prompt_stage2, prompt_stage3


EXPERIMENT_NAME = "rea_pulmo_staged_generation"
STAGE_ORDER = ["stage1_outputs_2_3", "stage2_outputs_5_6", "stage3_outputs_8_9"]


def history_value(args, sections: dict[str, Any], parsed: dict[str, Any], generated_key: str, gold_key: str) -> Any:
    if args.history_source == "gold":
        return sections[gold_key]
    return parsed.get(generated_key)


def build_case_stages(args, llm, case_payload: dict[str, Any], log_handle) -> dict[str, Any]:
    case_id = str(case_payload.get("case_id", ""))
    sections = get_sections(case_payload)
    stages: dict[str, Any] = {}

    stage1_images = collect_image_urls(sections[SECTION_1]) if args.image_input_mode == "text_image" else []
    stages["stage1_outputs_2_3"] = generate_stage(
        case_id=case_id,
        stage_name="stage1_outputs_2_3",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt_stage1(case_id, sections),
        image_urls=stage1_images,
        llm=llm,
        log_handle=log_handle,
    )
    stage1_parsed = stages["stage1_outputs_2_3"]["parsed_response"]
    history_2 = history_value(args, sections, stage1_parsed, "initial_differential_diagnoses", SECTION_2)
    history_3 = history_value(args, sections, stage1_parsed, "initial_recommended_exams", SECTION_3)

    stage2_images = collect_image_urls(sections[SECTION_1], sections[SECTION_4]) if args.image_input_mode == "text_image" else []
    stages["stage2_outputs_5_6"] = generate_stage(
        case_id=case_id,
        stage_name="stage2_outputs_5_6",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt_stage2(case_id, sections, history_2, history_3),
        image_urls=stage2_images,
        llm=llm,
        log_handle=log_handle,
    )
    stage2_parsed = stages["stage2_outputs_5_6"]["parsed_response"]
    history_5 = history_value(args, sections, stage2_parsed, "updated_differential_diagnoses", SECTION_5)
    history_6 = history_value(args, sections, stage2_parsed, "follow_up_recommended_exams", SECTION_6)

    stage3_images = (
        collect_image_urls(sections[SECTION_1], sections[SECTION_4], sections[SECTION_7])
        if args.image_input_mode == "text_image"
        else []
    )
    stages["stage3_outputs_8_9"] = generate_stage(
        case_id=case_id,
        stage_name="stage3_outputs_8_9",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt_stage3(case_id, sections, history_2, history_3, history_5, history_6),
        image_urls=stage3_images,
        llm=llm,
        log_handle=log_handle,
    )

    return stages


def parse_args():
    parser = build_arg_parser(
        description="Run PULSAR staged clinical generation.",
        default_filename_template="{model_name}_{input_mode}_main_generation.json",
        default_log_template="{model_name}_{input_mode}_main_generation_log.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(args, EXPERIMENT_NAME, STAGE_ORDER, build_case_stages)


if __name__ == "__main__":
    main()
