from pathlib import Path
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
E = REPO / "cdm_exploration/experiments"

def main():
    qmatrix = np.load(D / "qmatrix_v2_K100.npy")
    cluster_labels = json.load(open(D / "cluster_labels_v2_K100.json"))
    items_meta = json.load(open(D / "item_full_text_recovered.json"))
    pilot_indices = json.load(open(E / "pilot_step4_int.json"))
    print(qmatrix.shape)
    print(len(cluster_labels))
    print(len(items_meta))
    print(pilot_indices["per_item"][0])

    assert qmatrix.shape == (9523, 100)
    assert set(np.unique(qmatrix)) <= {0,1}
    assert abs(qmatrix.sum(axis=1).mean()-1.46) < 0.01

    subset = {pi["item_idx"] for pi in pilot_indices["per_item"]}
    meta ={r["item_idx"]: (r["benchmark"], r["subtask"]) for r in items_meta}
    print(len(subset))
    print(meta[0])

    full = "--all" in sys.argv          # full mode: every item, not just the pilot 998
    ids = sorted(meta) if full else sorted(subset)

    per_item = []
    for i in ids:
        row = qmatrix[i]
        cols = np.where(row == 1)[0]
        skills = [cluster_labels[str(j)] for j in cols]
        b, sub = meta[i]
        per_item.append({"item_idx":i, "benchmark":b, "subtask":sub, "skills":skills})
    print(len(per_item))
    print(per_item[0])
    print(sum(1 for r in per_item if not r["skills"]))
    used = set()
    for r in per_item:
        used.update(r["skills"])
    bank = [{"name": n, "definition": ""} for n in sorted(used)]
    print(len(bank))

    outname = "oldtax_full_format.json" if full else "oldtax_pilot_format.json"
    out = {"bank": bank, "per_item": per_item}
    json.dump(out, open(E / outname, "w"))
    print("wrote", E / outname)
    mine = json.load(open(E / outname))
    ref  = json.load(open(E / "pilot_step4_int.json"))
    print("mine bank[0]:", mine["bank"][0])
    print("ref  bank[0]:", ref["bank"][0])
    print("mine per_item[0] keys:", sorted(mine["per_item"][0]))
    print("ref  per_item[0] keys:", sorted(ref["per_item"][0]))


if __name__=="__main__":
    main()