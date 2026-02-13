"""
Prepare combined MATH + GSM8K dataset for skill extraction.

Loads both datasets from HuggingFace, combines them, and creates
stratified samples at different sizes for downstream processing.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import os

os.makedirs('../data/combined', exist_ok=True)

# --- Load MATH dataset ---

print("Loading MATH dataset...")

math_subjects = ['algebra', 'counting_and_probability', 'geometry',
                 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

math_data = []
for subject in math_subjects:
    ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test", trust_remote_code=True)
    for idx, item in enumerate(ds):
        math_data.append({
            'id': f"MATH_{subject}_{idx}",
            'source': 'MATH',
            'subject': item['type'],
            'level': item['level'],
            'problem': item['problem'],
            'solution': item['solution']
        })

math_df = pd.DataFrame(math_data)
print(f"  MATH: {len(math_df):,} problems")

# --- Load GSM8K dataset ---

print("Loading GSM8K dataset...")

gsm8k = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

gsm8k_data = []
for idx, item in enumerate(gsm8k):
    gsm8k_data.append({
        'id': f"GSM8K_{idx}",
        'source': 'GSM8K',
        'subject': 'Word Problems',
        'level': 'Grade School',
        'problem': item['question'],
        'solution': item['answer']
    })

gsm8k_df = pd.DataFrame(gsm8k_data)
print(f"  GSM8K: {len(gsm8k_df):,} problems")

# --- Combine and summarize ---

combined_df = pd.concat([math_df, gsm8k_df], ignore_index=True)
print(f"  Combined: {len(combined_df):,} problems")

for source in combined_df['source'].unique():
    source_df = combined_df[combined_df['source'] == source]
    print(f"\n  {source} ({len(source_df):,}):")
    for subject in source_df['subject'].unique():
        count = len(source_df[source_df['subject'] == subject])
        print(f"    {subject}: {count}")

# --- Create stratified samples ---

def create_stratified_sample(df, n_per_group=50):
    """Stratified sample by source and subject."""
    samples = []
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        for subject in source_df['subject'].unique():
            subject_df = source_df[source_df['subject'] == subject]
            n_sample = min(n_per_group, len(subject_df))
            sample = subject_df.sample(n=n_sample, random_state=42)
            samples.append(sample)
    return pd.concat(samples, ignore_index=True)

sample_small = create_stratified_sample(combined_df, n_per_group=10)   # ~80 problems
sample_medium = create_stratified_sample(combined_df, n_per_group=50)  # ~400 problems

print(f"\nSample sizes: small={len(sample_small)}, medium={len(sample_medium)}, full={len(combined_df)}")

# --- Save ---

combined_df.to_csv('../data/combined/full_dataset.csv', index=False)
sample_small.to_csv('../data/combined/sample_small.csv', index=False)
sample_medium.to_csv('../data/combined/sample_medium.csv', index=False)

sample_small_json = sample_small.to_dict(orient='records')
with open('../data/combined/sample_small.json', 'w') as f:
    json.dump(sample_small_json, f, indent=2)

print("Saved to ../data/combined/")
