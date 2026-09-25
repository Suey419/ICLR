# PULSAR Benchmark

Unlike single-turn medical question-answering benchmarks, PULSAR evaluates how models progressively update their diagnostic reasoning as new clinical evidence becomes available.

PULSAR is a clinical reasoning benchmark for evaluating large language models on challenging respiratory medicine cases. The benchmark focuses on diagnostically challenging real-world cases, including patients evaluated across multiple healthcare institutions before a definitive diagnosis was established. The benchmark represents each case as a staged clinical workflow, covering initial presentation, differential diagnosis, diagnostic planning, updated reasoning after new evidence, final diagnosis, and treatment planning. 

The codebase is organized as a clean, upload-ready pipeline. Model clients, prompt templates, generation orchestration, and evaluation prompts are separated so the benchmark can be extended without editing one large script.

## Repository Structure

```text
PULSAR_benchmark/
  dataset/                    # Case JSON files downloaded from Hugging Face
  generation/                 # Generated model outputs, created at runtime
  eval/                       # Evaluation reports, created at runtime
  llm_clients.py              # OpenAI-compatible model client wrapper
  generation_prompts.py       # Prompt templates for staged generation
  generation_pipeline.py      # Generation utilities, JSON parsing, logging, batching
  pulmonology_generate.py     # CLI entry point for generation
  eval_prompts.py             # Prompt templates for LLM-based evaluation
  eval_utils.py               # Shared constants, JSON parsing, text normalization
  eval_data.py                # Extract predicted and reference outputs from cases
  eval_matching.py            # Matching, ranking, diagnosis, exam, and treatment scoring
  eval_reasoning.py           # Reasoning-process extraction and scoring helpers
  eval_case.py                # Per-case evaluation assembly
  eval_summary.py             # Aggregate score summaries and report serialization
  eval_runner.py              # Batch evaluation, resume logic, logging, parallel execution
  eval_cli.py                 # Evaluation argument parser
  pulmonology_eval.py         # Thin CLI entry point for evaluation
  README.md
```

Runtime folders such as `generation/` and `eval/` do not need to exist before running the scripts.

## Dataset

The repository does not automatically download the benchmark cases. Access the dataset through the PULSAR Hugging Face repository and follow the license and usage conditions provided there. PULSAR is intended for research evaluation. It is not a medical device and must not be used as a substitute for professional clinical judgment or for direct patient-care decisions.

The case files can be downloaded from Hugging Face:

```text
https://huggingface.co/datasets/PULSARbenchmark/PULSAR
```

Place the downloaded JSON case files under:

```text
PULSAR_benchmark/dataset/
```

## Requirements

```bash
pip install openai numpy
```

The model interface is OpenAI-compatible. API keys should be passed by environment variable or command-line argument.

```bash
export OPENAI_API_KEY="your_api_key"
```

## Benchmark Workflow

| Phase | Information provided to the model | Model output |
|---|---|---|
| Stage 1 | Patient information and chief complaint | Initial differential diagnoses and recommended examinations |
| Stage 2 | Initial examination results and differential evidence | Updated differential diagnoses and follow-up examinations |
| Stage 3 | Key confirmatory examination information | Final diagnosis, diagnostic basis, and treatment plan |

The model receives information progressively. Future clinical evidence is not exposed before the corresponding stage.

## Case Format

Each case is a JSON file with a top-level `case_id` and `sections` object. The generation and evaluation scripts expect the following section keys:

```json
{
  "case_id": "case_001",
  "sections": {
    "1.patient_information_and_chief_complaint": {},
    "2.initial_differential_diagnoses": {},
    "3.initial_recommended_exams": {},
    "4.exam_results_and_differential_evidence": {},
    "5.updated_differential_diagnoses": [],
    "6.follow_up_recommended_exams": {},
    "7.key_confirmatory_exam_information": {},
    "8.final_diagnosis_and_diagnostic_basis": {},
    "9.treatment_plan": {}
  }
}
```

Image URLs can be included anywhere inside a section as objects with a `url` field. They are used only when `--image_input_mode text_image` is selected.

## Run Generation

Generate staged model outputs from case JSON files:

```bash
python pulmonology_generate.py \
  --dataset_dir dataset \
  --generate_dir generation \
  --model openai \
  --openai_model gpt-4o \
  --image_input_mode text \
  --workers 1
```

For an OpenAI-compatible endpoint:

```bash
python pulmonology_generate.py \
  --dataset_dir dataset \
  --generate_dir generation \
  --model openai \
  --openai_model your-model-name \
  --base_url https://your-endpoint.example/v1 \
  --apikey "$OPENAI_API_KEY"
```

The default output path is:

```text
generation/{model_name}/{input_mode}/{model_name}_{input_mode}_main_generation.json
```

Each generated case contains three stage records:

```text
stage1_outputs_2_3
stage2_outputs_5_6
stage3_outputs_8_9
```

## Run Evaluation

PULSAR evaluates both intermediate reasoning and final clinical decisions. The evaluation pipeline assesses the following components using the available structured reference annotations:

- coverage of reference differential diagnoses;
- ranking and prioritization of differential diagnoses;
- appropriateness of initially recommended examinations;
- updating of diagnoses after new evidence;
- selection of follow-up and confirmatory examinations;
- final-diagnosis accuracy;
- completeness of the diagnostic basis;
- appropriateness of the treatment plan; and
- quality of the staged reasoning process.

The evaluation produces per-case results and aggregate summaries across all evaluated cases.

Evaluate a generation file:

```bash
python pulmonology_eval.py \
  --generation_file generation/gpt-4o/text/gpt-4o_text_main_generation.json \
  --report_dir eval \
  --judge_mode llm \
  --model openai \
  --openai_model gpt-4o
```

For fast string-based checking without an LLM judge:

```bash
python pulmonology_eval.py \
  --generation_file generation/gpt-4o/text/gpt-4o_text_main_generation.json \
  --judge_mode exact
```

Evaluation outputs are written to:

```text
eval/{model_name}/{input_mode}/
```

Each evaluation run produces:

```text
{model_name}_{input_mode}_report.json
{model_name}_{input_mode}_score.json
{model_name}_{input_mode}_eval_log.jsonl
```

## Useful Options

```bash
--workers 4              # Run cases in parallel
--max_cases 10           # Evaluate only the first 10 cases
--force_recompute        # Ignore existing evaluation reports
--history_source gold    # Use reference intermediate outputs during generation
--image_input_mode text_image
```

The `--history_source` argument controls which intermediate outputs are included in the subsequent-stage context:

- `predicted`: Subsequent stages use the model's own earlier outputs. This setting evaluates the complete end-to-end reasoning trajectory and allows earlier errors to propagate.
- `gold`: Subsequent stages use the reference intermediate outputs. This setting evaluates performance at each stage while reducing error propagation from previous stages.

Use the same setting when comparing models. The default value is `predicted`.

## API Key Handling

The scripts check the following sources for an API key:

```text
--apikey
--sk
--openai_apikey / --gemini_apikey / --deepseek_apikey / --claude_apikey
OPENAI_API_KEY
API_KEY
```

Do not commit API keys, generated private outputs, or unreleased case data.

## Reproducibility

For reproducible comparisons, report the evaluated model and version, image-input mode, history-source setting, judge mode and judge model, number of evaluated cases, and code commit hash. Model outputs and LLM-based evaluation results may vary across model versions and API providers.

## Citation

If you use PULSAR in your research, please cite the accompanying paper:

```bibtex
@article{pulsar2026,
  title   = {PULSAR: [Full paper title]},
  author  = {[Author list]},
  journal = {[Journal or conference]},
  year    = {2026}
}


