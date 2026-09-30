# SkillEval

Skill-level evaluation of large language models with cognitive diagnosis models.

A single benchmark score hides what a model can and cannot do. SkillEval extracts the skills that each
benchmark question requires, groups them into 100 named skills, and fits a neural cognitive diagnosis model
on the answers of 3,811 open LLMs to 9,523 questions from MATH, BBH, GPQA, MuSR and IFEval. Every model gets
a mastery profile over the named skills. The profiles are used to predict performance on unseen questions, to
route each question to a cheaper model that can answer it, and to profile new models from a few answers.

![Overview of SkillEval](assets/overview.png)

- **Demo website:** source in [`demo/`](demo/); not online yet
- **Profile your own model:** [skilleval-cdm](https://github.com/berkearda/skilleval-cdm), a pip-installable tool
- **Trained model, Q-matrix and item embeddings:** [bearda/skilleval-cdm-assets](https://huggingface.co/datasets/bearda/skilleval-cdm-assets) on the Hugging Face Hub
- **Paper:** forthcoming

## Repository layout

| Path | Contents |
|---|---|
| `cdmeval/` | Python package: skill extraction and Q-matrix construction (`skills/`), the text-conditioned neural CDM and the baseline models (`modeling/`), training, metrics and routing evaluation (`evaluation/`), data loading (`data/`) |
| `configs/` | Hydra configuration used by the training scripts |
| `tools/` | Scripts for the experiments, analyses and figures (`run_*`, `diag_*`, `fig_*`); the `*.sbatch` files are the SLURM launchers used on a cluster |
| `cdm_exploration/experiments/` | Result files (JSON) behind the reported numbers; `experiment_log.json` records every run |
| `cdm_exploration/figures/` | Figures; the final ones are in `report/main_ready/` and `report/appendix_ready/` |
| `cdm_exploration/scripts/` | The first version of the pipeline |
| `release/` | The 3,811 evaluated LLMs (`llm_list.csv`) and the 100 skills (`skill_list.csv`) |
| `demo/` | The demo website |
| `tests/` | Tests |

The Python package is called `cdmeval`, the project's earlier name.

## Installation

```bash
pip install -e .
```

Python 3.9 to 3.11. Dependencies are listed in `pyproject.toml`.

## Data

The per-item responses come from RouterEval (Huang et al., 2025), which collects the item-level results of the
Open LLM Leaderboard v2. They are not redistributed here; `tools/build_expanded_matrix.py` builds the
3,811 x 9,523 response matrix from RouterEval, and the scripts expect their inputs under
`cdm_exploration/data/cdm_ready/`. The trained main model, the Q-matrix and the item-text embeddings are on
the Hugging Face Hub (link above).

Files that quote GPQA questions are not included, because the GPQA authors ask that its questions not be
posted in plain text. The human-evaluation sheets in `cdm_exploration/experiments/human_eval/` keep the
annotators' judgments without the question text.

## Where the main results come from

| Result | Scripts |
|---|---|
| Skill extraction and Q-matrix | `tools/extract_skills_v2.py` (prompt in `cdmeval/skills/extraction.py`), `tools/cluster_skills.py` |
| Main model and routing | `tools/train_expanded.py`, `tools/diag_table4_routing_summary.py` |
| Item-level prediction and baselines | `tools/run_ncdm_protocolA.py`, `tools/run_multi_seed.py`, `tools/run_irt_baseline_v2.py`, `tools/run_irtnet_headtohead.py`, `tools/run_knn_baseline.py` |
| Benchmark-level prediction | `tools/diag_benchpred_skilleval_vs_irtnet.py` |
| Stability of the profiles | `tools/compare_theta_stability.py` |
| Cost and accuracy of routing | `tools/run_pareto_multibaseline.py` |
| Profiling unseen LLMs | `tools/run_profiling_bayes.py`, `tools/run_profiling_bayes_adaptive500.py` |
| Cross-benchmark transfer | `tools/run_cross_benchmark_transfer.py` |
| Base and instruction-tuned models | `tools/run_alignment_tax.py` |
| Co-mastery between skills | `tools/run_skill_prerequisites.py` |
| Human validation of the skill assignments | `tools/make_human_eval_csv.py`, `tools/score_human_eval_agreement.py` |
| Release lists of LLMs and skills | `tools/build_release_lists.py` |

## Citation

A paper describing SkillEval is forthcoming; its citation will be added here.

## License

Code: MIT (see `LICENSE`). Data derived from the benchmarks follow the licenses of the original datasets.
