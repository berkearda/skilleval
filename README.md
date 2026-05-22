# SkillEval demo site

Static interactive visualization for the SkillEval framework: per-skill
ability scores for a set of language models, with drill-downs into each
skill.

Status: work in progress. Not yet deployed. The repo is local-only until
the table view phase lands.

## Tech stack

- Vite + React 19 + TypeScript
- Tailwind CSS (with the Okabe-Ito palette as custom colors)
- shadcn/ui (new-york style, slate base)
- React Router (BrowserRouter, basename `/skilleval`)
- TanStack Table + TanStack Virtual (wired in, not yet used)
- Recharts (wired in, not yet used)
- lucide-react icons

## Local development

```
npm install
npm run dev
```

Then open http://localhost:5173/skilleval/

## Production build

```
npm run build
```

Outputs to `dist/`. The Vite base path is `/skilleval/` so the built
site is ready to drop into a GitHub Pages project page at
`https://<user>.github.io/skilleval/` whenever we choose to publish.

## Data

Placeholder JSON lives in `public/data/`:

- `theta_matrix.json` — `{ models: [{ id, name, family, tier, params, theta[] }, ...] }`
- `skills.json` — `{ skills: [{ id, label, description, primary_benchmark, n_items, alpha_mean, theta_variance_across_LLMs, example_items[] }, ...] }`

The real datasets are produced upstream and will overwrite these
placeholders during deployment.
