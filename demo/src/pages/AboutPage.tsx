import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Check, Copy } from 'lucide-react'
import { PipelineDiagram } from '@/components/PipelineDiagram'
import { AUTHORS, BIBTEX, PAPER_TITLE } from '@/lib/citation'
import { WeakBeatsStrongChart } from '@/components/WeakBeatsStrongChart'
import { usePageTitle } from '@/hooks/usePageTitle'
import { useWeakBeatsStrong } from '@/hooks/useWeakBeatsStrong'

const BASE = import.meta.env.BASE_URL

function Figure({
  src,
  alt,
  caption,
  maxWidth,
}: {
  src: string
  alt: string
  caption: string
  maxWidth?: number
}) {
  return (
    <figure className="mt-4 max-w-4xl">
      {/* Figures come from the paper and are rendered on white; keep a white
          card in both themes so dark mode looks intentional. */}
      <div className="rounded-lg border border-border bg-white p-4 sm:p-6">
        <img
          src={`${BASE}figures/${src}`}
          alt={alt}
          loading="lazy"
          className="mx-auto h-auto w-full"
          style={maxWidth ? { maxWidth } : undefined}
        />
      </div>
      <figcaption className="mt-2 text-[13px] leading-5 text-muted-foreground">
        {caption}
      </figcaption>
    </figure>
  )
}

const linkCls = 'text-brand hover:underline'

function Citation() {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    void navigator.clipboard?.writeText(BIBTEX).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }
  return (
    <section id="cite" className="mt-10 max-w-4xl scroll-mt-20">
      <h2 className="text-xl font-semibold tracking-tight">Citation</h2>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
        {AUTHORS.map((a) => a.name).join(', ')}. <em>{PAPER_TITLE}</em>. 2026. The
        paper is forthcoming; until then, please cite it as follows.
      </p>
      <div className="relative mt-3 rounded-lg border border-border bg-card">
        <pre className="whitespace-pre-wrap break-words p-4 pr-24 font-mono text-xs leading-5 text-foreground">
          {BIBTEX}
        </pre>
        <button
          type="button"
          onClick={copy}
          className="absolute right-2 top-2 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-surface-elevated"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-brand" /> : <Copy className="h-3.5 w-3.5 text-muted-foreground" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </section>
  )
}

export function AboutPage() {
  usePageTitle('Method · SkillEval')
  const { data: wbsData, error: wbsError } = useWeakBeatsStrong()
  const wbs = wbsData?.wbs
  // the router does not scroll to #anchors by itself (footer and home link to #cite)
  const { hash } = useLocation()
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView()
  }, [hash])
  return (
    <article className="page py-12 leading-7 text-foreground">
      <h1 className="text-3xl font-semibold tracking-tight">Method</h1>

      <p className="mt-5 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
        SkillEval replaces the single benchmark score with a skill profile:
        a cognitive diagnostic model from psychometrics scores every model on
        100 named skills, estimated from 3,811 models answering 9,523 items
        across five public benchmarks.
      </p>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          How it works
        </h2>
        <PipelineDiagram />
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          Why profiles, not a single score
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          A single accuracy hides specialists.{' '}
          {wbs ? (
            <>
              Of the {wbs.n_test_items.toLocaleString()} held-out items,{' '}
              {wbs.n_items.toLocaleString()} ({Math.round(wbs.pct)}%) are
              answered correctly by at least one model with at most 13B
              parameters while the strongest single model fails.
            </>
          ) : (
            <>
              On many held-out items, a model with at most 13B parameters
              answers correctly while the strongest single model fails.
            </>
          )}
        </p>
        <figure className="mt-4 max-w-4xl">
          {wbsData ? (
            <WeakBeatsStrongChart wbs={wbsData.wbs} skills={wbsData.skills} />
          ) : wbsError ? (
            <div className="rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
              Error loading the chart data: {wbsError}
            </div>
          ) : (
            <div className="h-80 animate-pulse rounded-lg bg-muted" />
          )}
          <figcaption className="mt-2 text-[13px] leading-5 text-muted-foreground">
            The twelve skills where this happens most often, as a share of
            each skill's held-out items (skills with at least 10 held-out
            items). Colors mark the benchmark most of the skill's items come
            from.
          </figcaption>
        </figure>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          Do the profiles predict performance?
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          On held-out items, the accuracy predicted from the profiles tracks
          each model's observed accuracy: closely on most benchmarks, more
          loosely on GPQA and MuSR. The figure compares SkillEval with IRTNet,
          a neural IRT model whose ability dimensions have no names.
        </p>
        <Figure
          src="fig_benchmark_prediction_irtnet_row.png"
          alt="Predicted versus observed held-out accuracy per benchmark for SkillEval and IRTNet"
          caption="Predicted vs. observed held-out accuracy per benchmark for SkillEval and IRTNet (d = 232); r is the Pearson correlation across models, and the diagonal is perfect prediction."
          maxWidth={720}
        />
      </section>

      <section className="mt-10 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          Reading the numbers
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          Mastery θ runs from 0 to 1. It is the model's estimated level on a
          skill, not the share of that skill's items it answered correctly.
          Mean θ averages a model's mastery over all 100 skills equally; it is
          an entry point, not a verdict, and two models with the same mean can
          have very different strengths. Mastery estimates carry sampling
          noise, so small gaps between adjacent ranks are not meaningful.
        </p>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          The numbers come from the paper's main model and are a fixed
          snapshot, not a live leaderboard. The 100 skill names, the
          Q-matrix and the code are in the{' '}
          <a href="https://github.com/berkearda/skilleval" className={linkCls}>
            repository
          </a>
          .
        </p>
      </section>

      <section className="mt-10 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          Data and credits
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          The answers of the 3,811 models are the item-level results of the{' '}
          <a
            href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
            className={linkCls}
          >
            Open LLM Leaderboard
          </a>{' '}
          v2, as packaged by{' '}
          <a href="https://huggingface.co/datasets/linggm/RouterEval" className={linkCls}>
            RouterEval
          </a>{' '}
          (Huang et al., 2025; MIT licence). The 9,523 items come from MATH,
          BIG-Bench Hard, GPQA, MuSR and IFEval and keep their original
          licences. GPQA questions are not shown on this site, because the
          GPQA authors ask that they not be posted in plain text.
        </p>
      </section>

      <Citation />
    </article>
  )
}
