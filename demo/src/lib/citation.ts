// The paper behind the site, as registered (same order as the submission).
export const PAPER_TITLE =
  'SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models'

export const AUTHORS: { name: string; bibtex: string; affiliation: 1 | 2 }[] = [
  { name: 'Berke Arda', bibtex: 'Arda, Berke', affiliation: 1 },
  { name: 'Peng Cui', bibtex: 'Cui, Peng', affiliation: 1 },
  { name: 'Qiaoyuan Zheng', bibtex: 'Zheng, Qiaoyuan', affiliation: 1 },
  { name: 'Rudolf Debelak', bibtex: 'Debelak, Rudolf', affiliation: 2 },
  { name: 'Mubashara Akhtar', bibtex: 'Akhtar, Mubashara', affiliation: 1 },
  { name: 'Mrinmaya Sachan', bibtex: 'Sachan, Mrinmaya', affiliation: 1 },
]

export const AFFILIATIONS = ['ETH Zurich', 'EPFL']

export const REPO_URL = 'https://github.com/berkearda/skilleval'
export const DATA_URL = 'https://github.com/berkearda/skilleval/tree/main/release'

// Replace with the arXiv entry once the paper is public.
export const BIBTEX = `@misc{arda2026skilleval,
  title        = {${PAPER_TITLE}},
  author       = {${AUTHORS.map((a) => a.bibtex).join(' and ')}},
  year         = {2026},
  howpublished = {\\url{${REPO_URL}}},
  note         = {Code and data; paper forthcoming}
}`
