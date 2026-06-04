import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Check, Copy } from 'lucide-react'

const BIBTEX = `@misc{skilleval,
  title  = {SkillEval: Skill-Level Ability Estimates for Language Models},
  note   = {Cognitive-diagnostic evaluation snapshot, 2026-05-22},
  year   = {2026},
  url    = {https://example.org/skilleval}
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
              Overview
            </Link>
            <Link to="/browse" className={linkCls}>
              Browse
            </Link>
            <Link to="/about" className={linkCls}>
              Methodology
            </Link>
          </div>

          <div className="flex flex-col gap-2">
            <span className={kicker}>Resources</span>
            <a href="#" className={linkCls}>
              Paper
            </a>
            <a href="#" className={linkCls}>
              Code
            </a>
            <a href="#" className={linkCls}>
              Data
            </a>
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

        <div className="mt-6 border-t border-border pt-4 text-xs text-muted-foreground tabular">
          Snapshot 2026-05-22. 3,811 models, 100 skills, 9,523 items across MATH,
          BBH, GPQA, MuSR, IFEval.
        </div>
      </div>
    </footer>
  )
}
