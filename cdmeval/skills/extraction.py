"""LLM-based skill extraction from math and reasoning problems."""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

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


def extract_skills_openai(problem: str, model: str = "gpt-4o-mini") -> Dict:
    """Extract skills using OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in mathematics education. "
                "Respond only with valid JSON.",
            },
            {"role": "user", "content": SKILL_EXTRACTION_PROMPT.format(problem=problem)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def extract_skills_anthropic(
    problem: str, model: str = "claude-3-haiku-20240307"
) -> Dict:
    """Extract skills using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "user", "content": SKILL_EXTRACTION_PROMPT.format(problem=problem)}
        ],
    )
    text = response.content[0].text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    return {"skills": [], "error": "Could not parse response"}


def extract_skills_mock(problem: str, model: Optional[str] = None) -> Dict:
    """Keyword-based mock extractor for testing without API access."""
    skills = []
    problem_lower = problem.lower()

    keyword_map = {
        ("equation", "solve for", "find x", "variable"): "solving equations",
        ("factor", "factorize"): "factoring expressions",
        ("triangle", "circle", "square", "area", "perimeter"): "geometric reasoning",
        ("probability", "chance", "likely"): "probability calculation",
        ("how many", "how much", "total", "altogether"): "word problem comprehension",
        ("+", "-", "*", "/", "sum", "difference", "product"): "arithmetic operations",
        ("fraction", "ratio", "percent"): "working with fractions/ratios",
        ("graph", "plot", "coordinate"): "coordinate geometry",
    }

    for keywords, skill in keyword_map.items():
        if any(w in problem_lower for w in keywords):
            skills.append(skill)

    if not skills:
        skills = ["mathematical reasoning", "problem comprehension"]

    return {
        "skills": skills[:4],
        "primary_skill": skills[0],
        "reasoning": "Mock extraction based on keyword matching",
    }


EXTRACTORS = {
    "openai": extract_skills_openai,
    "anthropic": extract_skills_anthropic,
    "mock": extract_skills_mock,
}


def run_extraction(
    input_file: str,
    output_file: str,
    api: str = "mock",
    model: Optional[str] = None,
    limit: Optional[int] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Run skill extraction on a dataset of math problems."""
    if api not in EXTRACTORS:
        raise ValueError(f"Unknown API: {api}. Choose from {list(EXTRACTORS.keys())}")

    extract_fn = EXTRACTORS[api]

    df = pd.read_csv(input_file)
    if limit:
        df = df.head(limit)

    id_col = "id" if "id" in df.columns else "item_idx"
    problem_col = "problem" if "problem" in df.columns else "question"

    print(f"Processing {len(df)} problems with {api}...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extraction = (
                extract_fn(row[problem_col], model)
                if model
                else extract_fn(row[problem_col])
            )
            results.append({
                "item_idx": row[id_col],
                "source": row.get("source", ""),
                "subject": row.get("subject", ""),
                "problem": row[problem_col],
                "skills": extraction.get("skills", []),
                "primary_skill": extraction.get("primary_skill", ""),
                "num_skills": len(extraction.get("skills", [])),
                "reasoning": extraction.get("reasoning", ""),
            })
            if api != "mock":
                time.sleep(delay)
        except Exception as e:
            print(f"  Error on {row[id_col]}: {e}")
            results.append({
                "item_idx": row[id_col],
                "source": row.get("source", ""),
                "subject": row.get("subject", ""),
                "problem": row[problem_col],
                "skills": [],
                "primary_skill": "",
                "num_skills": 0,
                "reasoning": f"Error: {str(e)}",
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    json_output = output_file.replace(".csv", ".json")
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {output_file}, {json_output}")
    return results_df


def analyze_extracted_skills(results_df: pd.DataFrame) -> Counter:
    """Print frequency analysis of extracted skills."""
    all_skills = []
    for skills in results_df["skills"]:
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    print(
        f"\nProblems: {len(results_df)}, "
        f"Avg skills/problem: {results_df['num_skills'].mean():.2f}"
    )
    print(f"Unique skills: {len(skill_counts)}")
    print("\nTop 20 skills:")
    for skill, count in skill_counts.most_common(20):
        pct = count / len(results_df) * 100
        print(f"  {skill}: {count} ({pct:.1f}%)")

    for source in results_df["source"].unique():
        source_df = results_df[results_df["source"] == source]
        source_skills = []
        for skills in source_df["skills"]:
            if isinstance(skills, str):
                skills = eval(skills)
            source_skills.extend(skills)
        print(f"\n{source}: {len(set(source_skills))} unique skills from {len(source_df)} problems")

    return skill_counts


# ════════════════════════════════════════════════════════════════════
#  Benchmark-aware extraction (v2 prompt)
# ════════════════════════════════════════════════════════════════════

QMATRIX_SYSTEM_PROMPT = """You are a psychometrician designing a Q-matrix for cognitive diagnostic assessment of AI systems. Your task: identify the specific cognitive skills each test item measures.

RULES:
1. Each skill must be a compound phrase (5-10 words): [specific_cognitive_process] + [specific_content_domain]
2. Extract 2-4 skills per item. At least one must be domain-specific.
3. FORBIDDEN generic labels (will be rejected):
   - "reading comprehension", "logical reasoning", "critical thinking"
   - "mathematical reasoning", "problem solving", "analytical thinking"
   - "scientific knowledge", "attention to detail", "numerical reasoning"
   - Any label that would apply to >15% of items in the benchmark
4. Specificity test: would fewer than 100 items in a 10,000-item test share this exact skill? If not, decompose further.
5. Use lowercase snake_case labels.

GOOD examples by benchmark type:

MATH: "factoring_higher_degree_polynomials_over_integers", "applying_pigeonhole_principle_to_combinatorial_bounds", "computing_modular_arithmetic_in_residue_classes"

BBH: "tracing_boolean_operator_precedence_in_nested_expressions", "identifying_causal_direction_from_correlational_evidence", "tracking_object_positions_through_spatial_transformations"

GPQA: "applying_gauss_law_to_cylindrical_charge_distributions", "predicting_reaction_products_via_retrosynthetic_analysis", "interpreting_phylogenetic_trees_from_molecular_sequence_data"

MuSR: "integrating_alibis_and_motives_to_identify_suspects", "tracking_object_locations_across_sequential_room_transfers", "resolving_team_allocation_under_mutual_exclusion_constraints"

IFEval: "enforcing_exact_word_count_in_structured_output", "maintaining_consistent_formatting_across_nested_lists", "embedding_required_keywords_while_preserving_coherence"

BAD examples (too generic, rejected):
- "problem solving" -> decompose to "applying_substitution_to_solve_nonlinear_systems"
- "reading comprehension" -> decompose to "extracting_temporal_ordering_from_narrative_clues"
- "scientific knowledge" -> decompose to "applying_conservation_of_angular_momentum_to_rotating_bodies"
"""

QMATRIX_USER_TEMPLATE = """Benchmark: {benchmark} | Subtask: {subtask}

Item:
{question_text}

Respond in JSON:
{{
  "skills": [
    {{
      "label": "5-10 word compound skill in snake_case",
      "process": "the specific cognitive operation",
      "domain": "the specific content area"
    }}
  ],
  "primary_skill": "label of the single most important skill"
}}"""


FORBIDDEN_LABELS = {
    "reading comprehension", "logical reasoning", "critical thinking",
    "mathematical reasoning", "problem solving", "analytical thinking",
    "scientific knowledge", "attention to detail", "numerical reasoning",
    "reading_comprehension", "logical_reasoning", "critical_thinking",
    "mathematical_reasoning", "problem_solving", "analytical_thinking",
    "scientific_knowledge", "attention_to_detail", "numerical_reasoning",
}


def extract_skills_v2_openai(
    question_text: str,
    benchmark: str,
    subtask: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Extract skills using the benchmark-aware v2 prompt via OpenAI."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QMATRIX_SYSTEM_PROMPT},
            {"role": "user", "content": QMATRIX_USER_TEMPLATE.format(
                benchmark=benchmark, subtask=subtask,
                question_text=question_text[:1500],
            )},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    # Normalise: extract just the label strings
    skills = result.get("skills", [])
    labels = []
    for s in skills:
        if isinstance(s, dict):
            labels.append(s.get("label", ""))
        elif isinstance(s, str):
            labels.append(s)

    # Filter forbidden
    labels = [l for l in labels if l.lower().replace("_", " ") not in FORBIDDEN_LABELS]

    return {
        "skills": labels,
        "primary_skill": result.get("primary_skill", labels[0] if labels else ""),
        "raw": result,
    }
