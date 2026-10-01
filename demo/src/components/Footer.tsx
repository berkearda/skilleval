import { Link } from 'react-router-dom'
import { AFFILIATIONS, REPO_URL } from '@/lib/citation'

const linkCls = 'underline-offset-2 hover:text-foreground hover:underline'

export function Footer() {
  return (
    <footer className="border-t border-border bg-background">
      <div className="page flex flex-col gap-3 py-6 text-xs leading-5 text-muted-foreground md:flex-row md:items-baseline md:justify-between md:gap-8">
        <p className="max-w-[80ch]">
          SkillEval, {AFFILIATIONS.join(' and ')}. Answers of 3,811 models to 9,523 items from
          the{' '}
          <a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard" className={linkCls}>
            Open LLM Leaderboard
          </a>{' '}
          v2 via{' '}
          <a href="https://huggingface.co/datasets/linggm/RouterEval" className={linkCls}>
            RouterEval
          </a>
          ; mastery profiles from the paper's main model, a fixed snapshot.
        </p>
        <nav className="flex shrink-0 flex-wrap gap-x-5 gap-y-1" aria-label="Resources">
          <a href={REPO_URL} className={linkCls}>
            Code
          </a>
          <a href={`${REPO_URL}/blob/main/release/skill_list.csv`} className={linkCls}>
            The 100 skills
          </a>
          <Link to="/about#cite" className={linkCls}>
            Cite
          </Link>
          <span>Paper forthcoming</span>
        </nav>
      </div>
    </footer>
  )
}
