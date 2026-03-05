# CDMEval

**Evaluating LLMs with Cognitive Diagnostic Models**

A framework for fine-grained skill-level evaluation of large language models
using Cognitive Diagnostic Models (CDMs). CDMEval moves beyond aggregate accuracy
scores by decomposing LLM competence into interpretable skill mastery profiles,
enabling skill-aware routing of queries to the most suitable model.

<!-- TODO: Add pipeline diagram -->

## Overview

Standard LLM benchmarks report a single accuracy number per model. CDMEval
applies psychometric techniques from educational measurement to produce
multi-dimensional skill profiles for each LLM, then leverages these profiles
for informed model selection.

The framework operates in three layers:

**Layer 1 -- Automated Q-matrix construction.**
Raw math problems are annotated with fine-grained cognitive skills via LLM
metacognition (7,723 unique skill labels extracted from 2,643 items). Skills
are embedded with SBERT (`all-mpnet-base-v2`), then clustered with
Hierarchical Agglomerative Clustering (HAC, cosine distance, average linkage)
into a 50-skill taxonomy. The resulting binary Q-matrix maps each item to its
required skills.

**Layer 2 -- NCDM-based LLM skill profiling.**
A Neural Cognitive Diagnosis Model (NCDM) is fitted on the response matrix
(235 LLMs x 2,643 items) and the Q-matrix. The model learns a latent mastery
vector for each LLM across all 50 skills, with monotonicity constraints
ensuring that higher skill mastery always increases the probability of a
correct response.

**Layer 3 -- Skill-aware routing.**
A text-conditioned variant replaces item ID embeddings with frozen SBERT
projections, enabling prediction on unseen questions. Given a novel query, the
model ranks all LLMs by predicted P(correct) and routes to the top candidate.

## Key Results

All results below are on held-out test sets. Models trained on the MATH
(Level 5) + GSM8K response data from the
[RouterEval](https://github.com/withmartian/routereval) leaderboard.

### NCDM Evaluation (HAC-50, 15 epochs)

| Metric   | Value  |
|----------|--------|
| AUC      | 0.9486 |
| Accuracy | 0.8977 |
| RMSE     | 0.2741 |

### Clustering Method Comparison

| Method     |  K  | Silhouette | Test AUC |
|------------|----:|------------|----------|
| HAC        |  50 | 0.053      | 0.9486   |
| K-Means    |  50 | 0.144      | 0.9480   |
| HDBSCAN    | 281 | 0.160      | 0.9456   |
| HAC        | 100 | 0.088      | 0.9482   |
| HAC        | 281 | 0.154      | 0.9471   |

HAC K=50 achieves the highest downstream AUC despite lower silhouette,
indicating that intrinsic clustering quality does not directly predict
diagnostic model performance.

### Cold-Start Routing (529 held-out items)

| Method                       | Acc@1  | Acc@3  | Acc@5  |
|------------------------------|--------|--------|--------|
| Text-conditioned routing     | 0.5955 | 0.7032 | 0.7259 |
| Majority baseline (best LLM) | 0.5690 | --     | --     |
| Random                       | 0.2620 | 0.4574 | 0.5332 |
| Oracle (upper bound)         | 0.8866 | --     | --     |

The text-conditioned model routes to a correct LLM 59.5% of the time on
completely unseen items, outperforming both the always-pick-best-LLM baseline
(56.9%) and random selection (26.2%).

### Extended Analysis

| Analysis                              | Finding                            |
|---------------------------------------|------------------------------------|
| PCA on 50-skill mastery profiles      | 23 components explain 90% variance |
| Q-row approximation (NN Jaccard)      | Mean similarity = 0.551            |
| Per-skill AUC (text vs ID, 34 skills) | Mean text AUC = 0.817, ID = 0.812 |

## Installation

```bash
git clone https://github.com/berkearda/cdm-llm-evaluation.git
cd cdmeval
pip install -e .

# For development (linting, testing):
pip install -e ".[dev]"
```

Requires Python >= 3.8. Tested with PyTorch 2.4.1.

## Quick Start

The full pipeline is executed through Hydra-based tools in `tools/`. Each tool
reads its configuration from `configs/config.yaml` and accepts CLI overrides.

```bash
# 1. Build the response matrix from RouterEval data
python tools/build_response_matrix.py

# 2. Extract skills and cluster into a taxonomy
python tools/extract_skills.py skills.api=mock
python tools/cluster_skills.py

# 3. Train NCDM with HAC-50 clustering
python tools/train_hac_ncdm.py device=mps model.epochs=15

# 4. Train text-conditioned model and run routing demo
python tools/train_text_ncdm.py device=mps model.epochs=15
```

Individual tools can be listed with `ls tools/`.

## Project Structure

```
cdmeval/
  __init__.py
  data/
    dataloader.py         # PyTorch DataLoader factories
    prepare.py            # HuggingFace dataset loading
    response_matrix.py    # Response matrix I/O and triplet construction
    routereval.py         # RouterEval leaderboard processing
  evaluation/
    metrics.py            # AUC/Acc/RMSE computation, mastery extraction
    training.py           # NCDM and TextConditionedNet training loops
  modeling/
    pos_linear.py         # PosLinear (non-negative weight constraint)
    text_conditioned.py   # TextConditionedNet architecture
  skills/
    clustering.py         # HDBSCAN, K-Means, HAC clustering + Q-matrix
    extraction.py         # LLM-based skill extraction (OpenAI/Anthropic/mock)
  utils/
    device.py             # Device resolution and seeding
    visualization.py      # Publication-quality figure generation

configs/
  config.yaml             # Hydra configuration (paths, model, device)

tools/
  prepare_data.py         # Download MATH + GSM8K from HuggingFace
  extract_skills.py       # Run LLM skill extraction
  cluster_skills.py       # SBERT + UMAP + HDBSCAN taxonomy
  build_response_matrix.py# RouterEval -> binary response matrix
  train_ncdm.py           # Standard NCDM training
  train_hac_ncdm.py       # HAC-50 clustering + NCDM
  train_text_ncdm.py      # Text-conditioned NCDM + routing demo
  compare_clustering.py   # Clustering method sweep
  generate_figures.py     # Publication figures
  run_experiments.py      # Extended evaluation experiments

cdm_exploration/
  scripts/                # Original exploration scripts (01--10)
  data/                   # Raw and processed data
  figures/                # Generated figures
```

## Configuration

CDMEval uses [Hydra](https://hydra.cc/) for configuration management. Default
values live in `configs/config.yaml` and can be overridden from the command
line:

```bash
# Change device and training hyperparameters
python tools/train_ncdm.py device=cpu model.epochs=20 model.lr=0.001

# Change data paths
python tools/train_ncdm.py paths.cdm_ready=/path/to/data

# Change clustering parameters
python tools/train_hac_ncdm.py skills.n_clusters=100
```

## Data

CDMEval uses two data sources:

- **[MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)** (Level 5, 7 subjects) -- 1,324 competition-level math problems.
- **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** -- 1,319 grade-school math word problems.
- **[RouterEval](https://github.com/withmartian/routereval)** -- Per-item correctness data for 235 open-source LLMs on both benchmarks.

The combined dataset contains 2,643 items evaluated by 235 LLMs (621,105 response triplets).

## Citation

```bibtex
@misc{arda2025cdmeval,
  title     = {{CDMEval}: Evaluating Large Language Models with Cognitive
               Diagnostic Models},
  author    = {Arda, Nizamettin Berke},
  year      = {2025},
  institution = {ETH Z\"urich},
  note      = {Master's thesis, in progress}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
