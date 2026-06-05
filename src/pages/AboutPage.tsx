import { PipelineDiagram } from '@/components/PipelineDiagram'

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

export function AboutPage() {
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
          A single accuracy hides specialists: on 38% of held-out items, some
          small model (≤13B) answers correctly where the strongest single
          model fails.
        </p>
        <Figure
          src="fig_weak_beats_strong.png"
          alt="Per-skill percentages of test items where a small model beats the strongest single model"
          caption="The twelve skills where small models most often beat the strongest single model. Colors mark the source benchmark."
          maxWidth={620}
        />
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          Do the profiles predict performance?
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          On held-out items, accuracy predicted from the profiles tracks true
          accuracy on every benchmark, clearly tighter than a classical
          two-parameter IRT baseline.
        </p>
        <Figure
          src="fig_benchmark_prediction.png"
          alt="Predicted versus true accuracy per benchmark for SkillEval and an IRT 2PL baseline"
          caption="Predicted vs. true held-out accuracy per benchmark. Blue is SkillEval; pink is the IRT 2PL baseline; the diagonal is perfect prediction."
          maxWidth={720}
        />
      </section>

      <section className="mt-10 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          Reading the numbers
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          Mean θ averages a model's mastery over all 100 skills equally; it is
          an entry point, not a verdict, and two models with the same mean can
          have very different strengths. Mastery estimates carry sampling
          noise, so small gaps between adjacent ranks are not meaningful. The
          numbers are a fixed snapshot for inspection, not a live leaderboard;
          the K=100 skill taxonomy and trained model parameters will be
          released once the paper is public.
        </p>
      </section>
    </article>
  )
}
