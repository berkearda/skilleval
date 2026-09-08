#!/usr/bin/env python3
"""One clipping function, shared, because a flat cut deletes the question.

MuSR items are ~4,800-character narratives whose actual ask sits at the very
end. A flat `text[:N]` therefore hands a model a story that stops mid-sentence
with no question in it. Measured on this corpus: 506 questions exceed 4,000
characters, 502 of them MuSR (66% of that benchmark), and 493 of the 506 have
question-like text after the cut.

That produced a skill in the codebook that does not exist: "recognizing the
exact word or punctuation required to complete the final, abruptly cut-off
sentence". It is an artefact of the truncation, not a property of any question.

This project had already found the bug once, in the project log 2026-06-13:

    MuSR full text (~4,800 chars, question at the end) must NOT be cut by the
    [:1500] cap in the prompt - needs a head+tail or higher cap for the full run.

Step 1 was fixed. Steps 4, 6 and 8 then reintroduced it with flat caps. Keeping
one implementation here is the point: three copies of a rule drift apart, and
these three did.
"""
MARK = " …[middle omitted]… "


def clip(text: str, budget: int, head_frac: float = 0.55) -> str:
    """Keep the opening and the closing, drop the middle.

    The tail matters more than a proportional split suggests, because that is
    where the question lives. Below `budget` the text is returned untouched.
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= budget:
        return text
    if budget <= len(MARK) + 20:
        return text[:budget]
    head = int((budget - len(MARK)) * head_frac)
    tail = budget - len(MARK) - head
    return text[:head] + MARK + text[-tail:]
