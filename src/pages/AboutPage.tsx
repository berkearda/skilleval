export function AboutPage() {
  return (
    <article className="mx-auto max-w-[70ch] px-6 py-12 leading-7 text-foreground">
      <h1 className="text-3xl font-semibold tracking-tight">About SkillEval</h1>

      <section className="mt-6 space-y-4 text-[15px] text-foreground/90">
        <p>
          SkillEval is a framework for skill-aware evaluation of large language
          models. It decomposes a model's benchmark performance into a profile
          over latent skills, using cognitive diagnostic models adapted from
          psychometrics. Each model is summarised by a vector of per-skill
          mastery values rather than a single headline accuracy, which lets
          two models with the same average diverge sharply on the things
          users actually care about.
        </p>

        <p>
          This site is an interactive view over a fixed evaluation snapshot:
          3,811 LLMs assessed on 9,523 items drawn from five public
          benchmarks (MATH, BBH, GPQA, MuSR, IFEval). The items are clustered
          into 100 named skills, and each model is scored on every skill via
          a neural cognitive diagnostic model. The Browse page is a sortable
          table over the full grid: browse by family, search for a specific
          model, or sort any column (including raw accuracy and mean θ) to
          find the specialists for a given skill. Click any row to open that
          model's full ranked skill profile, and any skill column header to
          open the skill's detail page with its top-ranked LLMs and example
          items drawn from the underlying cluster.
        </p>

        <p>
          The framework is research code. Planned follow-ups include
          model-upload calibration — cold-start profiling of a new model
          from its answers on a small fixed item set — and an open release
          of the K=100 skill taxonomy together with the trained NCDM
          parameters after the paper is public. The numbers shown here are
          a snapshot for inspection, not a leaderboard ranking.
        </p>
      </section>
    </article>
  )
}
