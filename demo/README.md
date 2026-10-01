# SkillEval demo site

Interactive view of the SkillEval skill profiles: a model-by-skill table, a page per skill and per model,
and a comparison view. The site is not online yet.

## Data

- `public/data/theta_matrix.json`: the skill mastery of each of the 3,811 LLMs on the 100 skills, from the main
  SkillEval model, and each LLM's share of the 9,523 items answered correctly.
- `public/data/skills.json`: the 100 skills with the names listed in `../release/skill_list.csv`, their primary
  benchmark, item counts and up to three example items each. GPQA questions are not included, because the GPQA
  authors ask that its questions not be posted in plain text.
- `public/data/home.json`: what the Home page shows (top 12 models, the Compare example, the grid's 14 skills), so
  the first visit does not download the full matrix.
- `public/data/weak_beats_strong.json`: the held-out items that some model with at most 13B parameters answers while
  the strongest single model fails, overall and per skill (the Methodology page's chart).
- `public/figures/`: the paper's benchmark-prediction figure.

The example items and the chart data are rebuilt from the public data with

```bash
python tools/download_data.py            # from the repository root
python demo/scripts/build_site_data.py
```

## Development

```bash
npm install
npm run dev      # http://localhost:5173/skilleval/
npm run build    # writes dist/
```

Built with Vite, React, TypeScript, Tailwind CSS, TanStack Table and Recharts.
