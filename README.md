# Philosophy Argument Evaluation

Evaluate LLM-generated philosophical arguments against human baselines using automated grading.

## Overview

This project:
1. Extracts philosophical arguments from academic papers (question + ~600 word answer pairs)
2. Prompts LLMs to generate answers to the same philosophical questions
3. Grades both LLM and human answers using a detailed rubric
4. Collects results for comparison and analysis

## Setup

```bash
conda activate evals
pip install httpx openai python-dotenv pymupdf
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
```

## Project Structure

```
philosophy_explore/
├── main_questions.jsonl          # 10 Q&A pairs (question, human_answer, source_paper)
├── response_rubric.md            # Grading rubric (10 criteria, 0-5 each, max 50)
├── generate_answers.py           # Generate LLM answers
├── grade_responses.py            # Grade answers using rubric
├── extract_arguments.py          # Extract arguments from PDFs (uses Claude Opus)
│
├── prompts/                      # Prompt templates for answer generation
│   ├── answer_with_rubric.txt    # Includes grading criteria in prompt
│   └── answer_without_rubric.txt # No rubric shown to model
│
├── data/
│   ├── all_results.csv           # Master CSV with all graded results
│   ├── human_grades/             # Human baseline grades (one file per grader model)
│   │   └── {grader_model}.jsonl
│   └── runs/                     # Experiment runs
│       └── {run_name}/
│           ├── config.json       # Model, prompt, samples, temperature
│           ├── answers.jsonl     # Generated answers
│           └── grades.jsonl      # Grades for this run
│
├── papers/                       # Source PDF papers (gitignored)
└── output/                       # PDF extraction output (gitignored)
```

## Usage

### Generate Answers

```bash
# OpenRouter models
python generate_answers.py \
  --model openrouter/google/gemma-3-12b-it \
  --prompt answer_without_rubric \
  --samples 5 \
  --temperature 1.0

# OpenAI models
python generate_answers.py \
  --model openai/gpt-4o \
  --prompt answer_with_rubric \
  --samples 3
```

Options:
- `--model`: Model identifier with provider prefix (`openai/` or `openrouter/`)
- `--prompt`: Prompt template name (without `.txt`)
- `--samples`: Number of samples per question (default: 1)
- `--temperature`: Sampling temperature (default: 1.0)
- `--run-name`: Custom run name (default: auto-generated)

### Grade Responses

```bash
# Grade model answers from a run
python grade_responses.py \
  --run gemma-3-12b-it_answer_without_rubric_20260129_203000 \
  --grader openrouter/google/gemma-3-12b-it

# Grade human baselines (do this once per grader model)
python grade_responses.py \
  --human \
  --grader openai/gpt-4o
```

Grading outputs to:
- `data/runs/{run_name}/grades.jsonl` (for model answers)
- `data/human_grades/{grader_model}.jsonl` (for human baselines)
- `data/all_results.csv` (appended for all grades)

## Grading Rubric

Each answer is scored 0-5 on 10 criteria (max 50 points):

1. **Thesis Clarity** - Clear position stated early
2. **Charitable Engagement** - Steelmans opposing views
3. **Objection Handling** - Anticipates and addresses objections
4. **Example Quality** - Concrete examples that do argumentative work
5. **Precision in Distinctions** - Carefully distinguishes similar concepts
6. **Constructive Contribution** - Offers positive proposals
7. **Argumentative Risk-Taking** - Defends substantive positions
8. **Problem Reframing** - Reveals hidden assumptions
9. **Explanatory Unification** - Single principle explains multiple phenomena
10. **Scope Honesty** - Accurate about what has/hasn't been shown

## Data Formats

### main_questions.jsonl
```json
{"question": "...", "human_answer": "...", "source_paper": "Author-Title-2020.pdf"}
```

### answers.jsonl
```json
{"question_id": "...", "question": "...", "answer": "...", "model": "openrouter/...", "prompt_variant": "answer_with_rubric", "sample_idx": 0, "is_human": false}
```

### grades.jsonl
```json
{"question_id": "...", "model": "...", "grader_model": "...", "scores": {"thesis_clarity": 4, ..., "total": 38}, "is_human": false}
```

### all_results.csv
Columns: `question_id, answer, model, prompt_variant, sample_idx, is_human, grader_model, thesis_clarity, ..., total, timestamp`

## Experiment Workflow

1. **Grade human baselines** (once per grader model):
   ```bash
   python grade_responses.py --human --grader openai/gpt-4o
   ```

2. **Generate model answers** with different configurations:
   ```bash
   python generate_answers.py --model openrouter/meta-llama/llama-3-70b-instruct --prompt answer_with_rubric --samples 5
   python generate_answers.py --model openrouter/meta-llama/llama-3-70b-instruct --prompt answer_without_rubric --samples 5
   ```

3. **Grade model answers**:
   ```bash
   python grade_responses.py --run <run_name> --grader openai/gpt-4o
   ```

4. **Analyze**: Load `all_results.csv` to compare model vs human scores, create histograms, etc.

## Notes

- OpenAI uses native `n` parameter for multi-sampling
- OpenRouter uses async gather (no native multi-sample support)
- Human baseline grades are stored separately to avoid re-grading per experiment
- All results accumulate in `all_results.csv` for easy analysis
