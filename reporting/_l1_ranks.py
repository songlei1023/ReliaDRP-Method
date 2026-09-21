"""Recompute L1 mean ranks under several conventions, old vs new.

The printed draft column is reproduced from the OLD csv with the v3 row
substituted, so we can tell which subset the column was computed over.
"""
import csv, statistics as st

P = ["EXT_random", "EXT_LDO", "EXT_LCO"]
TAB = ["clclsa", "mlp", "delfos", "attn", "pancancer", "mgataf", "mmcl",
       "bandrp", "codeae", "crdnn", "fourierdrug", "vaen", "mograph",
       "tgsa", "deepdtf"]          # the 15 external rows printed in the table
OURS = "reliadrp"


def load(path, rcol, mcol, pcol):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r[rcol])
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            out.setdefault((r[pcol], r[mcol]), []).append(v)
    return out


def board(data, ours_vals):
    """-> {proto: [(model, mean), ...] desc} with ours injected as `reliadrp`."""
    b = {}
    for p in P:
        ms = {m: st.mean(v) for (pp, m), v in data.items() if pp == p}
        ms["reliadrp"] = ours_vals[p]
        b[p] = sorted(ms.items(), key=lambda kv: -kv[1])
    return b


def meanrank(b, pool):
    mr = {}
    for p in P:
        for i, (m, _) in enumerate([x for x in b[p] if x[0] in pool], 1):
            mr.setdefault(m, []).append(i)
    return {m: st.mean(v) for m, v in mr.items()}


new = load("l1_unified_seeded.csv", "test_r", "model", "proto")
old = load("extended_benchmark_all.csv", "pearson", "model", "protocol")
oldtune = load("l1_v3_tune_confirm.csv", "test_r", "config", "proto")

old_ours = {p: st.mean(oldtune[(p, "mix2_mf03")]) for p in P}
new_ours = {p: st.mean(new[(p, "reliadrpv3mix2mf03")]) for p in P}

DRAFT = {  # what the current draft prints
    "clclsa": 2.00, "mlp": 4.00, "delfos": 6.00, "attn": 7.33,
    "pancancer": 7.33, "mgataf": 7.67, "mmcl": 8.00, "bandrp": 8.33,
    "codeae": 8.33, "crdnn": 8.33, "fourierdrug": 9.67, "vaen": 13.33,
    "mograph": 13.67, "tgsa": 15.00, "deepdtf": 16.00,
}

conventions = {
    "16 shown (ours + 15 baselines)": set(TAB) | {"reliadrp"},
    "16 external (15 baselines + evi2)": set(TAB) | {"evi2"},
    "17 (+ both our controls)": set(TAB) | {"evi2", "reliadrp"},
    "16 external, no evi2": set(TAB),
}

ob = board(old, old_ours)
print("OLD csv -- which convention reproduces the draft column?")
for name, pool in conventions.items():
    r = meanrank(ob, pool)
    hit = sum(1 for m in TAB if abs(r.get(m, -99) - DRAFT[m]) < 0.02)
    print("  %-36s %2d/15 match" % (name, hit))

print()
nb = board(new, new_ours)
print("NEW csv -- seeded mean ranks per convention")
for name, pool in conventions.items():
    r = meanrank(nb, pool)
    print("  " + name)
    print("    ours=%.2f  " % r.get("reliadrp", -1)
          + " | ".join("%s=%.2f" % (m, r.get(m, -1)) for m in TAB))
