"""
Extract fine-grained skills from math problems via LLM.

Extends Didolkar et al. (NeurIPS 2024) from single-skill to multi-skill
extraction. Supports OpenAI, Anthropic, and a mock keyword-based extractor.

Usage:
    python 02_extract_skills_llm.py --api openai --model gpt-4o-mini
    python 02_extract_skills_llm.py --api anthropic --model claude-3-haiku-20240307
    python 02_extract_skills_llm.py --api mock
"""

import pandas as pd
import json
import os
import argparse
import time
from typing import List, Dict
from tqdm import tqdm

# --- Skill extraction prompt ---

SKILL_EXTRACTION_PROMPT = """You are an expert in mathematics education and cognitive assessment. Given a math problem, identify all specific cognitive skills a solver would need.

Guidelines:
- Be specific (e.g., "solving linear equations" not just "algebra")
- Focus on cognitive operations, not topic names
- Include both mathematical and reasoning skills
- Use lowercase, concise labels (2-5 words each)
- List as many or as few skills as the problem genuinely requires

Problem:
{problem}

Respond in JSON format:
{{
    "skills": ["skill_1", "skill_2", ...],
    "primary_skill": "the single most important skill",
    "reasoning": "brief explanation"
}}"""

# --- API clients ---

def extract_skills_openai(problem: str, model: str = "gpt-4o-mini") -> Dict:
    """Extract skills using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in mathematics education. Respond only with valid JSON."},
            {"role": "user", "content": SKILL_EXTRACTION_PROMPT.format(problem=problem)}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def extract_skills_anthropic(problem: str, model: str = "claude-3-haiku-20240307") -> Dict:
    """Extract skills using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "user", "content": SKILL_EXTRACTION_PROMPT.format(problem=problem)}
        ]
    )

    text = response.content[0].text
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    return {"skills": [], "error": "Could not parse response"}


def extract_skills_mock(problem: str, model: str = None) -> Dict:
    """Keyword-based mock extractor for testing without API access."""
    skills = []
    problem_lower = problem.lower()

    keyword_map = {
        ('equation', 'solve for', 'find x', 'variable'): "solving equations",
        ('factor', 'factorize'): "factoring expressions",
        ('triangle', 'circle', 'square', 'area', 'perimeter'): "geometric reasoning",
        ('probability', 'chance', 'likely'): "probability calculation",
        ('how many', 'how much', 'total', 'altogether'): "word problem comprehension",
        ('+', '-', '*', '/', 'sum', 'difference', 'product'): "arithmetic operations",
        ('fraction', 'ratio', 'percent'): "working with fractions/ratios",
        ('graph', 'plot', 'coordinate'): "coordinate geometry",
    }

    for keywords, skill in keyword_map.items():
        if any(w in problem_lower for w in keywords):
            skills.append(skill)

    if not skills:
        skills = ["mathematical reasoning", "problem comprehension"]

    return {
        "skills": skills[:4],
        "primary_skill": skills[0],
        "reasoning": "Mock extraction based on keyword matching"
    }


# --- Main extraction pipeline ---

def run_extraction(input_file, output_file, api="mock", model=None, limit=None, delay=0.5):
    """Run skill extraction on a dataset of math problems."""

    extractors = {
        "openai": extract_skills_openai,
        "anthropic": extract_skills_anthropic,
        "mock": extract_skills_mock,
    }
    if api not in extractors:
        raise ValueError(f"Unknown API: {api}. Choose from {list(extractors.keys())}")

    extract_fn = extractors[api]

    df = pd.read_csv(input_file)
    if limit:
        df = df.head(limit)

    # Handle column name variations (older format vs cdm_ready format)
    id_col = "id" if "id" in df.columns else "item_idx"
    problem_col = "problem" if "problem" in df.columns else "question"

    print(f"Processing {len(df)} problems with {api}...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extraction = extract_fn(row[problem_col], model) if model else extract_fn(row[problem_col])

            results.append({
                'item_idx': row[id_col],
                'source': row.get('source', ''),
                'subject': row.get('subject', ''),
                'problem': row[problem_col],
                'skills': extraction.get('skills', []),
                'primary_skill': extraction.get('primary_skill', ''),
                'num_skills': len(extraction.get('skills', [])),
                'reasoning': extraction.get('reasoning', '')
            })

            if api != "mock":
                time.sleep(delay)

        except Exception as e:
            print(f"  Error on {row[id_col]}: {e}")
            results.append({
                'item_idx': row[id_col],
                'source': row.get('source', ''),
                'subject': row.get('subject', ''),
                'problem': row[problem_col],
                'skills': [],
                'primary_skill': '',
                'num_skills': 0,
                'reasoning': f'Error: {str(e)}'
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    json_output = output_file.replace('.csv', '.json')
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {output_file}, {json_output}")
    return results_df


def analyze_extracted_skills(results_df):
    """Print frequency analysis of extracted skills."""
    from collections import Counter

    all_skills = []
    for skills in results_df['skills']:
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    print(f"\nProblems: {len(results_df)}, Avg skills/problem: {results_df['num_skills'].mean():.2f}")
    print(f"Unique skills: {len(skill_counts)}")
    print(f"\nTop 20 skills:")
    for skill, count in skill_counts.most_common(20):
        pct = count / len(results_df) * 100
        print(f"  {skill}: {count} ({pct:.1f}%)")

    for source in results_df['source'].unique():
        source_df = results_df[results_df['source'] == source]
        source_skills = []
        for skills in source_df['skills']:
            if isinstance(skills, str):
                skills = eval(skills)
            source_skills.extend(skills)
        print(f"\n{source}: {len(set(source_skills))} unique skills from {len(source_df)} problems")

    return skill_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract skills from math problems using LLM")
    parser.add_argument("--input", default="../data/combined/sample_small.csv",
                        help="Input CSV file with problems")
    parser.add_argument("--output", default="../data/combined/skills_extracted.csv",
                        help="Output CSV file for results")
    parser.add_argument("--api", choices=["openai", "anthropic", "mock"], default="mock",
                        help="API to use for extraction")
    parser.add_argument("--model", default=None,
                        help="Model name (uses default if not specified)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems to process")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    args = parser.parse_args()

    print(f"Input: {args.input}, API: {args.api}, Model: {args.model or 'default'}")

    results_df = run_extraction(
        input_file=args.input,
        output_file=args.output,
        api=args.api,
        model=args.model,
        limit=args.limit,
        delay=args.delay
    )

    analyze_extracted_skills(results_df)
