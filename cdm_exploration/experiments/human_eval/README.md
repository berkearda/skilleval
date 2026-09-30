# Human evaluation of skill-question assignments

Three annotators judged the same 780 (skill, question) pairs: "is this skill needed to answer this question?"
The sheet was produced by `tools/make_human_eval_csv.py` (blank copy: `../human_eval_skills.csv`).

| File | Format as received | Answers |
|---|---|---|
| `raw/annotator_A.csv` | cp1252, `;` separated | yes / no (one typo "mo" = no) |
| `raw/annotator_B.csv` | UTF-8, `,` separated | 1 / 0 |
| `raw/annotator_C.csv` | UTF-8, `,` separated | 1 / 0 |

The raw files are what the annotators returned (received 29 July and 20 September 2026), with the question-text
column removed: GPQA's authors ask that its questions not be posted in plain text. Annotator identities are not recorded here.
Scoring: `python3 tools/score_human_eval_agreement.py` writes `../v2_human_eval_agreement.json`.
The skill names shown to the annotators are the relabelled names (oldtax_final), not the original cluster labels.
