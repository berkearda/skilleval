# Skill-Aware LLM Evaluation Using Cognitive Diagnostic Models

Master's thesis project exploring the use of Cognitive Diagnostic Models (CDMs) to extract fine-grained skill mastery profiles for Large Language Models from existing benchmark data.

## Overview

Instead of evaluating LLMs with a single aggregate score, this project treats LLMs as "students" and benchmark problems as "items" in a CDM framework. The pipeline:

1. **Extracts cognitive skills** from math problems using LLM-based metacognition
2. **Clusters** 7,700+ free-text skills into 281 interpretable groups via sentence embeddings
3. **Builds a Q-matrix** mapping items to required skill clusters
4. **Fits a Neural CDM** (NCDM) on response data from 235 LLMs across 2,643 problems
5. **Produces skill mastery profiles** revealing per-LLM strengths and weaknesses

## Repository Structure

```
cdm_exploration/
├── scripts/
│   ├── 01_prepare_combined_dataset.py   # Merge and preprocess benchmarks
│   ├── 02_extract_skills_llm.py         # LLM-based skill extraction
│   ├── 03_cluster_skills.py             # Embedding + UMAP + HDBSCAN clustering
│   ├── 04_build_response_matrix.py      # Extract response matrix from RouterEval
│   ├── 05_fit_ncdm.py                   # Train Neural CDM
│   └── 06_generate_figures.py           # Report figures
├── data/cdm_ready/                      # Pipeline outputs
├── figures/report/                      # Publication-quality figures
├── reports/                             # LaTeX reports
└── notebooks/                           # Exploratory analysis
```

## Data

- **Benchmarks**: GSM8K (1,319 items) + MATH Level 5 (1,324 items)
- **Response data**: From [RouterEval](https://huggingface.co/datasets/routereval) (235 LLMs)
- **External dependency**: [EduCDM](https://github.com/bigdata-ustc/EduCDM) (clone into `cdm_exploration/repos/`)

## Key Results

| Metric | Value |
|--------|-------|
| Unique skills extracted | 7,723 |
| Skill clusters | 281 |
| NCDM test AUC | 0.945 |
| NCDM test accuracy | 89.4% |

## Setup

```bash
pip install anthropic sentence-transformers umap-learn hdbscan scikit-learn torch pandas matplotlib seaborn tqdm

# Clone EduCDM
git clone https://github.com/bigdata-ustc/EduCDM.git cdm_exploration/repos/EduCDM
```

## References

- Didolkar et al. (2024). *Metacognitive Capabilities of LLMs*. NeurIPS 2024.
- Wang et al. (2020). *Neural Cognitive Diagnosis for Intelligent Education Systems*. AAAI 2020.
- Li et al. (2024). *RouterEval*. EMNLP 2025.
