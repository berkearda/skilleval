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
    <figure className="mt-5">
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

      <section className="mt-6 max-w-[68ch] space-y-4 text-[15px] text-foreground/90">
        <p>
          SkillEval is a framework for skill-aware evaluation of large language
          models. It decomposes a model's benchmark performance into a profile
          over latent skills, using cognitive diagnostic models adapted from
          psychometrics. Each model is summarised by a vector of per-skill
          mastery values rather than a single headline accuracy, which lets two
          models with the same average diverge sharply on the things users
          actually care about.
        </p>
      </section>

      <section className="mt-12">
        <h2 className="text-xl font-semibold tracking-tight">
          How SkillEval works
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          Benchmark questions are mapped to the skills they require, and every
          model's correct/incorrect record over all items forms a performance
          matrix. A neural cognitive diagnostic model reads the item text to
          estimate each item's difficulty and discrimination, and jointly
          learns a per-skill ability vector θ for every model. Held-out
          responses are used to test the fit, and the learned profiles are
          what this site shows.
        </p>
        <Figure
          src="fig_pipeline.png"
          alt="SkillEval pipeline: questions are mapped to skills, LLM responses form a performance matrix, and a neural CDM learns per-skill ability vectors"
          caption="The SkillEval pipeline. Questions are tagged with the skills they require; 3,811 models answer 9,523 items; a neural cognitive diagnostic model estimates item difficulty and discrimination from the item text and learns a per-skill mastery profile for every model."
        />
      </section>

      <section className="mt-12">
        <h2 className="text-xl font-semibold tracking-tight">
          Why profiles, not a single score
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          A single accuracy hides specialists. On 38% of held-out test items,
          some small model (at most 13B parameters) with the right skill
          profile answers correctly where the strongest single model fails.
          The skills where that happens are exactly the ones a profile
          surfaces and a leaderboard average buries.
        </p>
        <Figure
          src="fig_weak_beats_strong.png"
          alt="Per-skill percentages of test items where a small model beats the strongest single model"
          caption="Items where some small model (≤13B) answers correctly while the strongest single model fails, for the twelve skills where this is most common. Colors mark the source benchmark."
          maxWidth={640}
        />
      </section>

      <section className="mt-12">
        <h2 className="text-xl font-semibold tracking-tight">
          What the profiles look like
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          Within every size tier, models differ far more per skill than their
          averages suggest. The map below shows mastery relative to each
          model's tier mean on high-variance skills: orange marks an
          above-tier strength, blue a below-tier weakness. No model is
          uniformly strong, which is the point of profiling.
        </p>
        <Figure
          src="fig_landscape.png"
          alt="Heatmap of per-skill mastery relative to tier mean for representative models"
          caption="Per-skill mastery relative to the model's size-tier mean, for representative small, mid, and large models on the highest-variance skills, grouped by domain. Orange is an above-tier strength; blue is a below-tier weakness."
        />
      </section>

      <section className="mt-12">
        <h2 className="text-xl font-semibold tracking-tight">
          Do the profiles predict performance?
        </h2>
        <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
          A profile is only useful if it is predictive. On held-out items,
          accuracy predicted from the skill profiles tracks true accuracy
          closely on every benchmark, and clearly tighter than a classical
          two-parameter IRT baseline fit to the same data.
        </p>
        <Figure
          src="fig_benchmark_prediction.png"
          alt="Predicted versus true accuracy per benchmark for SkillEval and an IRT 2PL baseline"
          caption="Predicted vs. true held-out accuracy per benchmark. Blue is SkillEval; pink is a two-parameter IRT baseline; the diagonal is perfect prediction. Correlations are shown per panel."
          maxWidth={760}
        />
      </section>

      <section className="mt-12 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          Using this site
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          This site is an interactive view over a fixed evaluation snapshot:
          3,811 LLMs assessed on 9,523 items drawn from five public benchmarks
          (MATH, BBH, GPQA, MuSR, IFEval), clustered into 100 named skills.
          The leaderboard is a sortable table over the full grid: browse by
          family, search for a specific model, or sort any column (including
          raw accuracy and mean θ) to find the specialists for a given skill.
          Click any row to open that model's full ranked skill profile, and
          any skill column header to open the skill's detail page with its
          top-ranked LLMs and example items drawn from the underlying cluster.
        </p>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          The framework is research code. Planned follow-ups include
          model-upload calibration (cold-start profiling of a new model from
          its answers on a small fixed item set) and an open release of the
          K=100 skill taxonomy together with the trained NCDM parameters after
          the paper is public. The numbers shown here are a snapshot for
          inspection, not a leaderboard ranking.
        </p>
      </section>

      <section className="mt-12 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          How the composite is formed
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          The headline number on each model is the mean of its mastery
          estimates over all 100 skills. Mean θ weights every skill equally and
          does not depend on which other models are in the table, so adding or
          removing models never reorders the ones that remain. It is a summary,
          and like any single number over 100 dimensions it hides structure:
          two models with the same mean θ can have very different strengths.
          The per-skill grid and the per-model profile exist so that structure
          stays visible. Treat mean θ as an entry point, not a verdict.
        </p>
      </section>

      <section className="mt-12 max-w-[68ch]">
        <h2 className="text-xl font-semibold tracking-tight">
          Reading the ranks
        </h2>
        <p className="mt-3 text-[15px] leading-7 text-foreground/90">
          The mastery scores are latent estimates with sampling noise, not
          exact measurements. Small gaps between adjacent ranks fall inside
          that noise and should not be read as real differences in ability.
          Use the ranking to find candidates and the skill profile to compare
          them, rather than treating one position as strictly better than the
          next.
        </p>
      </section>
    </article>
  )
}
