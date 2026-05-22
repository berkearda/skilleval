export function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-3xl font-semibold tracking-tight">About SkillEval</h1>
      <p className="mt-4 leading-relaxed text-muted-foreground">
        SkillEval is a framework for measuring large language models on the
        latent skills they actually exercise on real benchmarks. Instead of one
        headline accuracy number, it produces a per-skill score per model, so
        you can see where a model is genuinely strong, where it is bluffing,
        and where two models with the same average diverge sharply. This site
        is a small interactive demo of those scores.
      </p>
    </div>
  )
}
