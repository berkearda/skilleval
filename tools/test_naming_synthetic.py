"""Test the revised naming prompt on synthetic clusters with KNOWN correct names,
before running it on the real 100 clusters.

Four clusters, each a deliberate trap:
  A clean         -> should get the obvious operation name
  B costume       -> same operation, three themes (dance/soccer/gifts); must name the
                     operation, not a theme
  C mixed         -> two real operations 60/40; should name the majority, and the fit
                     check should show a weak gap (flagging it mixed)
  D fluke/plurality-> counting operation, but 2 of 10 mention fruit; must recover the
                     operation, not name it "fruit" like the original taxonomy did
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

NSYS = ("You rename one skill in a test-skill taxonomy so that the name follows a strict "
        "format. A skill is the smallest named mental operation a solver must master to "
        "answer a question correctly; the same skill can appear across different topics and "
        "formats. Judge by the operation the solver performs, never by the topic or story of "
        "the questions. You are given the skill's questions; the name must describe what they require.")

REWRITE = """This skill currently has a name that needs replacing. Write a new name for it based on its questions.

Questions tagged with this skill:
{ev}

Current name (for context only, do not copy its style): {old}

First, decide the single cognitive operation these questions share: what the solver actually has to DO to answer them, ignoring the topic, story, or subject. If they span more than one operation, choose the one most of them require. Then name that operation.

Name format, all rules mandatory:
1. A lowercase verb phrase: one verb plus one specific object, for example "track object positions after swaps".
2. Between 3 and 8 words. A bare verb alone is forbidden.
3. No "and", no "or", no slashes.
4. Never use these words: "problem solving", "reasoning", "analysis", "skills", "advanced".
5. Name the operation the solver performs, not the topic, story, or subject. Two questions about different things (say, dance and soccer) that need the same operation must get the same kind of name. The name must fit these questions and NOT fit unrelated ones.

Return JSON only: {{"operation": "<the shared operation in plain words>", "name": "<new name>", "definition": "<one sentence, at most 15 words>"}}"""

CLUSTERS = {
 "A clean (area of triangle)": {
   "old": "Analyzing Geometric Figures For Measurement",
   "expect": "compute the area of a triangle",
   "q": [
     "A triangle has base 6 cm and height 4 cm. What is its area?",
     "Find the area of a triangle whose base is 10 and height is 3.",
     "The base of a triangle measures 8 units and its height 5 units. Compute the area.",
     "A triangular sail has a base of 12 m and a height of 9 m. What is its area in square metres?",
     "Compute the area of a triangle with base 7 and height 2.",
     "A triangle has a base of 15 inches and a height of 6 inches. Find its area.",
   ]},
 "B costume (track swaps, 3 themes)": {
   "old": "Tracking Dance Partner Assignments",
   "expect": "track an item's position through a sequence of swaps",
   "q": [
     "Alice, Bob and Claire are dancing, each with a partner. Alice starts with Rob. Alice and Bob swap partners, then Bob and Claire swap partners. Who is Alice dancing with at the end?",
     "In a square dance, three dancers begin with set partners. The first and second dancers trade partners, then the second and third trade. Who ends up with the original partner of dancer one?",
     "Three soccer players start in the positions striker, midfield, defence. The striker and midfielder swap, then the midfielder and defender swap. What position does the first player end in?",
     "On a soccer team, players X, Y, Z hold three roles. X and Y switch roles, then Y and Z switch. Which role does X finish with?",
     "At a gift exchange, Ana, Ben and Cara each hold a present. Ana and Ben swap gifts, then Ben and Cara swap. Which present does Ana hold at the end?",
     "Three friends each start with a coloured ball. The first two swap balls, then the last two swap. What colour ball does the first friend end with?",
   ]},
 "C mixed (count vs solve quadratic, 4/3)": {
   "old": "Evaluating Numerical Expressions",
   "expect": "MIXED: majority is counting objects; minority is solving quadratics",
   "q": [
     "I have 3 apples, two pens and a notebook. How many objects do I have in total?",
     "There are 4 chairs, a table and 2 lamps in the room. How many pieces of furniture and objects are there?",
     "A box holds a ruler, three erasers and a stapler. How many items are in the box?",
     "On the shelf there are five books, a clock and two mugs. How many objects are on the shelf?",
     "Solve for x: x^2 - 5x + 6 = 0.",
     "Find the roots of x^2 + 2x - 8 = 0.",
     "What are the solutions of x^2 - 9 = 0?",
   ]},
 "D fluke (count category, 2 fruit)": {
   "old": "Distinguishing Fruits From Non Fruits",
   "expect": "count items of a category in a list (NOT a fruit-specific name)",
   "q": [
     "I have a banana, an apple, a carrot and a potato. How many fruits do I have?",
     "In my basket there are two apples, a pear, an onion and a cabbage. How many fruits are there?",
     "I own a trumpet, a couch, a flute and two chairs. How many musical instruments do I have?",
     "There are three violins, a lamp, a drum and a sofa. How many musical instruments are there?",
     "I have a cauliflower, a stalk of celery, a cabbage and a garlic. How many vegetables do I have?",
     "On the table: a leek, two carrots, a spoon and a fork. How many vegetables are on the table?",
     "I have a pen, three notebooks, a stapler and a mug. How many items in total?",
     "There are a hammer, two screwdrivers, a wrench and a book. How many tools are there?",
     "My drawer holds a sock, a glove, a scarf and a hat. How many clothing items are in the drawer?",
     "I see a red pen, a blue pen, a pencil and a marker. How many writing tools are there?",
   ]},
}


def main():
    jd = tp.Judges(tp.make_client(), {})

    def name_one(item):
        title, d = item
        ev = "\n".join(f"- {q}" for q in d["q"])
        raw = jd._text(tp.SMALL, NSYS, REWRITE.format(ev=ev, old=d["old"]), max_out=250)
        try:
            obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:
            obj = {"operation": "(parse fail)", "name": raw[:60], "definition": ""}
        vio = tp.violations(str(obj.get("name", "")).lower())
        return title, d, obj, vio

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(name_one, CLUSTERS.items()))

    for title, d, obj, vio in results:
        print("=" * 78)
        print(f"CLUSTER: {title}")
        print(f"  old (fluke) name : {d['old']}")
        print(f"  we expect        : {d['expect']}")
        print(f"  model operation  : {obj.get('operation')}")
        print(f"  model NAME       : {obj.get('name')}")
        print(f"  model definition : {obj.get('definition')}")
        print(f"  format ok        : {'yes' if not vio else 'NO -> ' + '; '.join(vio)}")

    # fit-check demo on the mixed cluster C: name fits its own majority but not the minority?
    print("\n" + "=" * 78)
    print("FIT CHECK on the mixed cluster C (does the gate flag it?)")
    C = CLUSTERS["C mixed (count vs solve quadratic, 4/3)"]
    obj = [o for t, d, o, v in results if t.startswith("C mixed")][0]
    name, defn = obj.get("name", ""), obj.get("definition", "")
    def judge(q):
        u = (f"Item:\n{q}\n\nSkill: {name}\nDefinition: {defn}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return jd._json(tp.SMALL, tp.VSYS, u, "verdict") == "yes"
    with ThreadPoolExecutor(max_workers=6) as ex:
        verds = list(ex.map(judge, C["q"]))
    print(f"  candidate name: '{name}'")
    for q, v in zip(C["q"], verds):
        print(f"   {'YES' if v else 'no '}  {q[:70]}")
    yes = sum(verds)
    print(f"  own-fit = {yes}/{len(verds)} = {yes/len(verds):.0%}  "
          f"(a mixed cluster should score in the middle, not high, so it fails the >55 gap)")
    print(f"\nspent ${tp.spend['usd']:.3f}")


if __name__ == "__main__":
    main()
