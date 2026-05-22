// Color helpers for the SkillEval table view.
// All helpers return inline-style hex colors (not Tailwind classes) so the
// per-cell mastery gradient can interpolate smoothly across 100 levels
// without needing 100 Tailwind utility classes.

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

/**
 * Mastery cell background color. theta in [0,1] -> a blue gradient that
 * works in both light and dark mode (in dark mode we shift to slightly
 * cooler hues so contrast stays readable on a dark page).
 */
export function getMasteryColor(theta: number, darkMode: boolean): string {
  const t = clamp01(theta)
  if (darkMode) {
    // From near-background (dark slate) to a bright cyan-ish blue.
    // start ~ #1f2937 (slate-800), end ~ #38bdf8 (sky-400)
    const r = Math.round(31 + (56 - 31) * t)
    const g = Math.round(41 + (189 - 41) * t)
    const b = Math.round(55 + (248 - 55) * t)
    return `rgb(${r}, ${g}, ${b})`
  }
  // Light mode: pale blue -> deep navy
  // start ~ #eff6ff (blue-50), end ~ #1e3a8a (blue-900)
  const r = Math.round(239 + (30 - 239) * t)
  const g = Math.round(246 + (58 - 246) * t)
  const b = Math.round(255 + (138 - 255) * t)
  return `rgb(${r}, ${g}, ${b})`
}

/**
 * Pick text color (white or near-black) based on cell luminance so the
 * value stays readable on top of the mastery gradient.
 */
export function getMasteryTextColor(theta: number, darkMode: boolean): string {
  const t = clamp01(theta)
  if (darkMode) {
    // On the dark-mode gradient, low theta is dark -> light text; very high
    // theta is bright cyan -> dark text reads better.
    return t > 0.7 ? '#0b1220' : '#f8fafc'
  }
  return t > 0.55 ? '#ffffff' : '#0f172a'
}

const FAMILY_COLORS: Record<string, string> = {
  Llama: '#E69F00', // okabe-orange
  Qwen: '#56B4E9', // okabe-skyblue
  Mistral: '#009E73', // okabe-green
  Gemma: '#CC79A7', // okabe-purple
  Phi: '#B7A100', // darker yellow for contrast on white
  Mixtral: '#D55E00', // okabe-vermillion
  Other: '#64748b', // slate-500
}

export function getFamilyColor(family: string): string {
  return FAMILY_COLORS[family] ?? FAMILY_COLORS.Other
}

const BENCHMARK_COLORS: Record<string, string> = {
  MATH: '#0072B2', // okabe-blue
  BBH: '#E69F00', // okabe-orange
  GPQA: '#009E73', // okabe-green
  MuSR: '#D55E00', // okabe-vermillion
  IFEval: '#CC79A7', // okabe-purple
  Other: '#64748b', // slate-500
}

export function getBenchmarkColor(benchmark: string): string {
  return BENCHMARK_COLORS[benchmark] ?? BENCHMARK_COLORS.Other
}

export const FAMILY_LIST = [
  'Llama',
  'Qwen',
  'Mistral',
  'Gemma',
  'Phi',
  'Mixtral',
  'Other',
] as const

export const TIER_LIST = ['Small', 'Mid', 'Large', 'Other'] as const

export const BENCHMARK_LIST = ['MATH', 'BBH', 'GPQA', 'MuSR', 'IFEval'] as const
