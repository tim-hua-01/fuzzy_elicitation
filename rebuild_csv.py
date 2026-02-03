#!/usr/bin/env python3
"""
Rebuild all_results.csv from v2 run directories.
Maps precision_in_distinctions -> precision_distinctions and handles errors.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file and return list of dicts."""
    results = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {filepath} line {line_num}: {e}")
    return results

def process_run(run_dir: Path, errors: List[str]) -> List[Dict]:
    """Process a single run directory and return rows for CSV."""
    answers_file = run_dir / "answers.jsonl"
    grades_file = run_dir / "grades.jsonl"
    
    if not answers_file.exists():
        errors.append(f"Missing answers.jsonl in {run_dir.name}")
        return []
    
    if not grades_file.exists():
        errors.append(f"Missing grades.jsonl in {run_dir.name}")
        return []
    
    try:
        answers = load_jsonl(answers_file)
        grades = load_jsonl(grades_file)
    except Exception as e:
        errors.append(f"Error loading files in {run_dir.name}: {e}")
        return []
    
    # Create lookup for grades by (question_id, sample_idx)
    grades_lookup = {}
    for grade in grades:
        key = (grade.get('question_id'), grade.get('sample_idx'))
        grades_lookup[key] = grade
    
    rows = []
    for answer in answers:
        question_id = answer.get('question_id')
        sample_idx = answer.get('sample_idx')
        key = (question_id, sample_idx)
        
        # Get matching grade
        grade = grades_lookup.get(key)
        
        if not grade:
            errors.append(f"No grade found for {run_dir.name}: question={question_id}, sample={sample_idx}")
            continue
        
        # Check for grading errors
        if grade.get('error'):
            errors.append(f"Grade error in {run_dir.name}: question={question_id}, sample={sample_idx}, error={grade['error']}")
            continue
        
        # Extract scores, mapping precision_in_distinctions -> precision_distinctions
        scores = grade.get('scores', {})
        if not scores:
            errors.append(f"No scores in {run_dir.name}: question={question_id}, sample={sample_idx}")
            continue
        
        # Map the field name if needed
        if 'precision_in_distinctions' in scores:
            scores['precision_distinctions'] = scores.pop('precision_in_distinctions')
            errors.append(f"Fixed field name in {run_dir.name}: question={question_id}, sample={sample_idx}")
        
        # Calculate character counts
        answer_text = answer.get('answer', '')
        reasoning_text = answer.get('reasoning', '')
        answer_char_count = len(answer_text) if answer_text else 0
        reasoning_char_count = len(reasoning_text) if reasoning_text else 0
        
        # Build CSV row
        row = {
            'question_id': question_id,
            'answer': answer_text,
            'model': answer.get('model', ''),
            'prompt_variant': answer.get('prompt_variant', ''),
            'sample_idx': sample_idx,
            'is_human': answer.get('is_human', False),
            'grader_model': grade.get('grader_model', ''),
            'thesis_clarity': scores.get('thesis_clarity', ''),
            'argumentative_soundness': scores.get('argumentative_soundness', ''),
            'dialectical_engagement': scores.get('dialectical_engagement', ''),
            'precision_distinctions': scores.get('precision_distinctions', ''),
            'substantive_contribution': scores.get('substantive_contribution', ''),
            'example_quality': scores.get('example_quality', ''),
            'total': scores.get('total', ''),
            'timestamp': grade.get('timestamp', ''),
            'answer_char_count': answer_char_count,
            'reasoning_char_count': reasoning_char_count,
        }
        rows.append(row)
    
    return rows

def main():
    data_dir = Path("/Users/timhua/Documents/aisafety_githubs/philosophy_explore/data/v2")
    runs_dir = data_dir / "runs"
    output_file = data_dir / "all_results.csv"
    
    errors = []
    all_rows = []
    
    # Process all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(run_dirs)} run directories...")
    for run_dir in run_dirs:
        print(f"  Processing {run_dir.name}...")
        rows = process_run(run_dir, errors)
        all_rows.extend(rows)
        print(f"    Added {len(rows)} rows")
    
    # Write CSV
    if all_rows:
        fieldnames = [
            'question_id', 'answer', 'model', 'prompt_variant', 'sample_idx',
            'is_human', 'grader_model', 'thesis_clarity', 'argumentative_soundness',
            'dialectical_engagement', 'precision_distinctions', 'substantive_contribution',
            'example_quality', 'total', 'timestamp', 'answer_char_count', 'reasoning_char_count'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\nWrote {len(all_rows)} rows to {output_file}")
    else:
        print("\nNo rows to write!")
    
    # Print errors
    print(f"\n{'='*80}")
    print(f"ERRORS ENCOUNTERED: {len(errors)}")
    print(f"{'='*80}")
    if errors:
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    else:
        print("No errors!")

if __name__ == "__main__":
    main()
