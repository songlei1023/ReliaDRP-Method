"""Regenerate the paper's L1 table body straight from the seeded csv.

Mean rank is taken over the sixteen rows printed in the table (ours plus the
fifteen external baselines) -- the convention that reproduces the draft
column, verified in `_l1_ranks.py`.  Writes `_l1_table_body.tex` and splices
it into the manuscript between the header rule and the closing rule.
"""
import csv, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "l1_unified_seeded.csv")
TEX = os.path.join(HERE, "paper", "ReliaDRP_ICASSP2027.tex")
BODY = os.path.join(HERE, "_l1_table_body.tex")

P = ["EXT_random", "EXT_LDO", "EXT_LCO"]
OURS = "reliadrpv3mix2mf03"

# csv name -> printed name; None = our row
ROWS = [
    (OURS, "\\textbf{ReliaDRP}"),
    ("clclsa", "CLCLSA"),
    ("attn", "AttnOmics"),
    ("mlp", "MLP (multi-omics)"),
    ("pancancer", "panCancerDR"),
    ("bandrp", "BANDRP"),
    ("delfos", "DELFOS"),
    ("codeae", "CODE-AE"),
    ("mgataf", "MGATAF"),
    ("mmcl", "MMCL-CDR"),
    ("crdnn", "CRDNN"),
    ("fourierdrug", "FourierDrug"),
    ("vaen", "VAEN"),
    ("mograph", "MoGraphDRP"),
    ("tgsa", "TGSA"),
    ("deepdtf", "DeepDTF"),
]


def load(col):
    out = {}
    with open(CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r[col])
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            out.setdefault((r["proto"], r["model"]), []).append(v)
    return out


r_ = load("test_r")
e_ = load("test_rmse")

# per-protocol board over the sixteen printed models
rank = {}
for p in P:
    order = sorted(((m, st.mean(r_[(p, m)])) for m, _ in ROWS), key=lambda kv: -kv[1])
    for i, (m, _) in enumerate(order, 1):
        rank.setdefault(m, []).append(i)
mrank = {m: st.mean(v) for m, v in rank.items()}

def cell(txt, bold):
    return ("$\\mathbf{%s}$" % txt) if bold else ("$%s$" % txt)

lines = []
for m, name in ROWS:
    bold = (m == OURS)
    pr = [cell("%.3f{\\pm}%.3f" % (st.mean(r_[(p, m)]), st.stdev(r_[(p, m)])), bold)
          for p in P]
    rm = [cell("%.3f" % st.mean(e_[(p, m)]), bold) for p in P]
    rk = cell("%.2f" % mrank[m], bold)
    lines.append("%-17s & %s & %s & %s \\\\" % (name, " & ".join(pr), " & ".join(rm), rk))

body = "\n".join(lines) + "\n"
with open(BODY, "w", encoding="utf-8") as f:
    f.write(body)
print(body)

tex = open(TEX, encoding="utf-8").read()
head = "& \\textbf{rank} \\\\\n\\hline\n"
i = tex.index(head) + len(head)
j = tex.index("\n\\hline\n\\end{tabular*}", i)
new = tex[:i] + body.rstrip("\n") + tex[j:]
assert new != tex and new.count("\\textbf{ReliaDRP}") == tex.count("\\textbf{ReliaDRP}")
open(TEX, "w", encoding="utf-8").write(new)
print("spliced rows:", len(lines))
