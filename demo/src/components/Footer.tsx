import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Check, Copy } from 'lucide-react'

// BibTeX "Last, First" names, in the paper's author order; replace the entry with
// the arXiv one once the paper is public.
const AUTHORS = [
  'Arda, Berke',
  'Cui, Peng',
  'Zheng, Qiaoyuan',
  'Debelak, Rudolf',
  'Akhtar, Mubashara',
  'Sachan, Mrinmaya',
]

const BIBTEX = `@misc{arda2026skilleval,
  title        = {SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models},
  author       = {${AUTHORS.join(' and ')}},
  year         = {2026},
  howpublished = {\\url{https://github.com/berkearda/skilleval}},
  note         = {Code and data; paper forthcoming}
}`

const kicker = 'text-xs font-medium uppercase tracking-wider text-muted-foreground'
const linkCls = 'text-sm text-muted-foreground hover:text-foreground'

export function Footer() {
  const [copied, setCopied] = useState(false)

  const onCopy = () => {
    void navigator.clipboard?.writeText(BIBTEX).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto max-w-screen-2xl px-6 py-8">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          <div className="flex flex-col gap-2">
            <span className={kicker}>SkillEval</span>
            <Link to="/" className={linkCls}>
              Home
            </Link>
            <Link to="/leaderboard" className={linkCls}>
              Leaderboard
            </Link>
            <Link to="/skills" className={linkCls}>
              Skills
            </Link>
            <Link to="/compare" className={linkCls}>
              Compare
            </Link>
            <Link to="/about" className={linkCls}>
              Methodology
            </Link>
          </div>

          <div className="flex flex-col gap-2">
            <span className={kicker}>Resources</span>
            <a href="https://github.com/berkearda/skilleval" className={linkCls}>
              Code and results
            </a>
            <a href="https://github.com/berkearda/skilleval/blob/main/release/skill_list.csv" className={linkCls}>
              The 100 skills
            </a>
            <span className="text-sm text-muted-foreground">Paper forthcoming</span>
          </div>

          <div className="flex flex-col gap-2">
            <span className={kicker}>Cite</span>
            <button
              type="button"
              onClick={onCopy}
              className="inline-flex w-fit items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-sm text-foreground transition-colors hover:bg-surface-elevated"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-brand" />
              ) : (
                <Copy className="h-3.5 w-3.5 text-muted-foreground" />
              )}
              {copied ? 'Copied' : 'Copy BibTeX'}
            </button>
          </div>
        </div>

        <div className="mt-6 border-t border-border pt-4 text-xs leading-5 text-muted-foreground tabular">
          Mastery profiles from the paper's main model, a fixed snapshot. Answers
          of 3,811 models to 9,523 items from MATH, BBH, GPQA, MuSR and IFEval,
          from the{' '}
          <a
            href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
            className="underline-offset-2 hover:text-foreground hover:underline"
          >
            Open LLM Leaderboard
          </a>{' '}
          v2 via{' '}
          <a
            href="https://huggingface.co/datasets/linggm/RouterEval"
            className="underline-offset-2 hover:text-foreground hover:underline"
          >
            RouterEval
          </a>
          . Code under the MIT licence.
        </div>
      </div>
    </footer>
  )
}
