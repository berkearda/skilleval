// Type definitions matching the SkillEval JSON schemas.

export interface ExampleItem {
  item_idx: number
  benchmark: string
  subtask?: string
  primary_skill?: string
  text: string
}

export interface Model {
  id: number
  name: string
  hf_id?: string
  family: string
  tier: string
  params: number | null
  theta: number[]
  /** Fraction of the 9,523 items answered correctly (raw benchmark accuracy). */
  accuracy?: number
}

export interface Skill {
  id: number
  label: string
  description: string
  primary_benchmark: string
  n_items: number
  alpha_mean: number
  theta_variance_across_LLMs: number
  example_items: ExampleItem[]
  // Optional fields present on non-English-origin clusters (e.g. skill 67).
  label_english?: string
  language?: string
}

/** public/data/weak_beats_strong.json, written by scripts/build_site_data.py. */
export interface WeakBeatsStrong {
  rule: string
  n_test_items: number
  n_items: number
  pct: number
  strongest_model: string
  n_small_models: number
  min_test_items_per_skill: number
  skills: { id: number; n: number; wbs: number; pct: number }[]
}
