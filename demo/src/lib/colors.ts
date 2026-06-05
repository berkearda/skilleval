// Color helpers for the SkillEval table view.
// All helpers return inline-style colors (not Tailwind classes) so the
// per-cell mastery gradient can interpolate smoothly across 100 levels
// without needing 100 Tailwind utility classes.

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

// The SkillEval ramp: a single cool family (ice -> aqua -> mint -> teal ->
// sea blue -> deep navy) so data surfaces share a palette with the blue
// brand UI instead of importing a full-spectrum chart rainbow. Ordered in
// luminance, colorblind-safe (no red/green axis). Low -> high.
const RAMP_LIGHT = [
  [234, 241, 248],
  [201, 228, 229],
  [147, 211, 200],
  [88, 184, 172],
  [47, 149, 150],
  [42, 111, 151],
  [39, 76, 139],
  [31, 46, 102],
]

// Dark-mode mirror: low sits just above the blue-black background, high
// glows bright mint. Low -> high.
const RAMP_DARK = [
  [27, 39, 53],
  [30, 58, 77],
  [34, 92, 102],
  [46, 131, 127],
  [69, 168, 153],
  [111, 197, 169],
  [163, 221, 192],
  [217, 242, 226],
]

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function rampColor(stops: number[][], t: number): string {
  const x = clamp01(t) * (stops.length - 1)
  const i = Math.min(Math.floor(x), stops.length - 2)
  const f = x - i
  const r = Math.round(lerp(stops[i][0], stops[i + 1][0], f))
  const g = Math.round(lerp(stops[i][1], stops[i + 1][1], f))
  const b = Math.round(lerp(stops[i][2], stops[i + 1][2], f))
  return `rgb(${r}, ${g}, ${b})`
}

/**
 * Mastery cell background color. theta in [0,1] -> the SkillEval cool ramp
 * (deep navy = high in light mode; bright mint = high in dark mode).
 */
export function getMasteryColor(theta: number, darkMode: boolean): string {
  return rampColor(darkMode ? RAMP_DARK : RAMP_LIGHT, theta)
}

function relativeLuminance(rgb: string): number {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb)
  if (!m) return 0.5
  const ch = [+m[1], +m[2], +m[3]].map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
  })
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
}

/**
 * Pick text color (near-white or near-black) from the WCAG relative
 * luminance of the actual interpolated cell color, so the value stays
 * readable across the full ramp including the crossover band.
 */
export function getMasteryTextColor(theta: number, darkMode: boolean): string {
  const L = relativeLuminance(getMasteryColor(theta, darkMode))
  return L < 0.4 ? '#f8fafc' : '#0b1220'
}

/** Neutral surface for a missing (null/undefined) theta — never "low mastery". */
export const NULL_CELL_BG_LIGHT = 'hsl(240 5% 94%)'
export const NULL_CELL_BG_DARK = 'hsl(240 4% 16%)'

/** Convert an `rgb(r, g, b)` string to `rgba(r, g, b, a)` for translucency. */
export function rgbToRgba(rgb: string, alpha: number): string {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb)
  if (!m) return rgb
  return `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${alpha})`
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
