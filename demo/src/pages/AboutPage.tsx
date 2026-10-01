import { PipelineDiagram } from '@/components/PipelineDiagram'
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
    <figure className="mt-4">
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

export function AboutPage() {
  usePageTitle('Methodology · SkillEval')
  const { data: wbsData, error: wbsError } = useWeakBeatsStrong()
  const wbs = wbsData?.wbs
  return (
    <article className="mx-auto max-w-4xl px-6 py-12 leading-7 text-foreground">
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Methodology
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About SkillEval
      </h1>

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
        <figure className="mt-4">
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
    </article>
  )
}
