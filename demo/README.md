# SkillEval demo site

Interactive view of the SkillEval skill profiles, live at
[berkearda.github.io/skilleval](https://berkearda.github.io/skilleval/): a model-by-skill table, a page per
skill and per model, and a comparison view.

## Data

- `public/data/theta_matrix.json`: the skill mastery of each of the 3,811 LLMs on the 100 skills, from the main
  SkillEval model, and each LLM's share of the 9,523 items answered correctly.
- `public/data/skills.json`: the 100 skills with the names listed in `../release/skill_list.csv`, their primary
  benchmark, item counts and example items. GPQA questions are not included, because the GPQA authors ask that
  its questions not be posted in plain text.
- `public/figures/`: two figures from the paper.

## Development

```bash
npm install
npm run dev      # http://localhost:5173/skilleval/
npm run build    # writes dist/
```

Built with Vite, React, TypeScript, Tailwind CSS, TanStack Table and Recharts. Every push to `main` that
touches `demo/` rebuilds the site through `../.github/workflows/deploy-demo.yml`.
