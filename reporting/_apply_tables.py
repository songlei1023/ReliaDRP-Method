r"""Rebuild the four layer tables as single-column floats and re-apply the prose.

Starts from `paper/_bak_before_worst6.tex`, drops the two wide `table*` floats,
splices four single-column tables at the four subsection anchors (L1, L2, L3,
L4, so the float numbers follow the layer order) and rewrites the sentences that
point at them.  Because the tables now carry the numbers, the prose around them
is trimmed of the duplicated values.
"""
import os

PAPER = os.path.join("paper", "ReliaDRP_ICASSP2027.tex")
BAK = os.path.join("paper", "_bak_before_worst6.tex")
B = {k: open("_body_%s.tex" % k, encoding="utf-8").read().rstrip("\n")
     for k in ("L1", "L2", "L3", "L4")}


def build(caption, label, spec, head, body):
    """One single-column float: footnotesize caption, scriptsize tabular."""
    return "\n".join([
        r"\begin{table}[!t]",
        r"\centering",
        r"\renewcommand{\arraystretch}{0.74}",
        r"\footnotesize",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\setlength{\tabcolsep}{0.9pt}",
        r"\scriptsize",
        r"\begin{tabular}{@{}%s@{}}" % spec,
        r"\hline",
        head,
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


T1 = build(
    r"L1 monotherapy, the ten models with the best mean rank over L1--L4 "
    r"(seeded re-run, $n{=}5$ seeds; Pearson mean${\pm}$sd).  \textbf{rk}/\textbf{avg}: "
    r"mean rank over the three L1 protocols / over all four layers.  Ranks and "
    r"bold values are taken on the full sixteen-model board.",
    "tab:l1", "lccccc",
    r"\textbf{Model} & \textbf{ID} & \textbf{DO} & \textbf{CO} & \textbf{rk} & \textbf{avg} \\",
    B["L1"])

T2 = build(
    r"L2 drug combinations, same ten models and columns as Table~\ref{tab:l1} "
    r"(Pearson mean${\pm}$sd; our row is the v3-tuned ReliaDRPCombo, $n{=}5$, "
    r"baselines $n{=}3$; DeepDTF is undefined drug-blind).",
    "tab:l2", "lccccc",
    r"\textbf{Model} & \textbf{ID} & \textbf{DO} & \textbf{CO} & \textbf{rk} & \textbf{avg} \\",
    B["L2"])

T3 = build(
    r"L3 single-cell sensitivity, same ten models as "
    r"Table~\ref{tab:l1} (AUROC mean${\pm}$sd, $n{=}5$ seeds; L3 has no cold-cell "
    r"protocol).  The in-domain "
    r"column spans only $0.008$ and cannot rank models.",
    "tab:l3", "lcccc",
    r"\textbf{Model} & \textbf{ID} & \textbf{DO} & \textbf{rk} & \textbf{avg} \\",
    B["L3"])

T4 = build(
    r"L4 unified transfer, same ten models (Pearson mean, $n{=}5$ seeds), seeded "
    r"initialisation.  C = pooled cell-line screens, AML = BeatAML2, "
    r"PDX = Bruna PDX biobank.  Best forward value in bold; reverse transfer is "
    r"below chance for every model.",
    "tab:l4", "lcccccc",
    r"\textbf{Model} & \textbf{C$\to$AML} & \textbf{C$\to$PDX} "
    r"& \textbf{AML$\to$C} & \textbf{PDX$\to$C} & \textbf{rk} & \textbf{avg} \\",
    B["L4"])

tex = open(BAK, encoding="utf-8").read()

# ---- drop the two old wide floats --------------------------------------
for anchor in ("\\caption{L4 unified transfer",
               "\\caption{Complete L1 monotherapy comparison"):
    i = tex.index("\\begin{table*}[!t]", tex.index(anchor) - 400)
    j = tex.index("\\end{table*}\n", i) + len("\\end{table*}\n")
    tex = tex[:i] + tex[j:]

# ---- splice the four new ones, in layer order --------------------------
for anchor, block in (
        ("\\subsection{L2: Combinations under Two Shifts}", T1),
        ("\\subsection{L3: A Negative Result under a Unified Protocol}", T2),
        ("\\subsection{L4: Unified Cross-Domain Transfer}", T3),
        ("\\subsection{Conditional-Coverage Diagnosis and the GC-CP Fix}", T4)):
    assert tex.count(anchor) == 1, anchor
    tex = tex.replace(anchor, block + "\n\n" + anchor, 1)

# ---- prose: the tables carry the numbers now, so drop the duplicates ----
edits = [
    ("Shift severity increases from random (in-domain) to LCO (cold cells) to LDO\n"
     "(drug-blind).  Across fifteen external models, ReliaDRP ranks 1/16 on all\n"
     "three L1-extended protocols and leads on both metrics\n"
     "(Table~\\ref{tab:l1pearson}).  Its mean rank is $1.00$, ahead of CLCLSA\n"
     "\\cite{clclsa} ($2.00$), AttnOmics ($4.33$) and the multi-omics MLP\n"
     "\\cite{mlpbaseline} ($4.67$); eight of the fifteen sit at rank $8$ or worse.\n"
     "With initialisation seeded the margins are $+0.059$ in-domain\n"
     "($0.742$/$0.683$), $+0.095$ drug-blind ($0.356$/$0.261$) and $+0.102$ cold\n"
     "cells ($0.657$/$0.555$), each past three standard errors of our seed spread;\n"
     "seeding leaves our row within $0.007$ but drops the baselines (MLP $-0.264$,\n"
     "DELFOS $-0.268$ in-domain).",
     "Tables~\\ref{tab:l1}--\\ref{tab:l4} tabulate all four layers over one set of ten\n"
     "models: the six with the worst mean rank across the layers (TGSA, VAEN,\n"
     "MGATAF, CRDNN, CLCLSA, panCancerDR) are omitted, except that the two 2026\n"
     "baselines are kept regardless, and every rank and bold value is still taken\n"
     "over all sixteen, so the printed rows cannot flatter the survivors.  Shift\n"
     "severity increases from random (ID) to cold cells (CO) to drug-blind (DO),\n"
     "and ReliaDRP ranks 1/16 on all three protocols on both metrics\n"
     "(Table~\\ref{tab:l1}; RMSE is released).  Its mean rank is $1.00$ against\n"
     "$2.00$ for the runner-up CLCLSA \\cite{clclsa}, omitted from the table; eight\n"
     "of the fifteen baselines sit at rank $8$ or worse.  Seeding the initialisation\n"
     "leaves the margins $+0.059$ (ID), $+0.095$ (DO) and $+0.102$ (CO), each past\n"
     "three standard errors of our seed spread, and drops the baselines\n"
     "(MLP $-0.264$, DELFOS $-0.268$ in-domain)."),
    ("Under the same harness ReliaDRPCombo reaches rank 4/16 in-domain\n"
     "($0.751{\\pm}0.002$; MoGraph leads at $0.813$), rank 1/16 under cold-cell shift\n"
     "($0.668{\\pm}0.003$) and rank 2/15 drug-blind ($0.524{\\pm}0.012$; BANDRP\n"
     "$0.538$, a margin we read as a tie).",
     "Under the same harness ReliaDRPCombo (Table~\\ref{tab:l2}) ranks 4/16\n"
     "in-domain behind MoGraphDRP, 1/16 under cold cells and 2/15 drug-blind,\n"
     "where BANDRP's $0.538$ is a margin we read as a tie."),
    ("Under one harness, one recipe and\n"
     "seeded initialisation for all sixteen models, two findings follow.  (i) The\n"
     "in-domain split is \\emph{saturated}: every model lies in $0.977$--$0.985$\n"
     "AUROC, a $0.008$ spread against per-seed sd $\\le0.003$, and ReliaDRP is last\n"
     "($0.977$); ranking there is not informative.  (ii) Drug-blind, where the\n"
     "protocol does discriminate, ReliaDRP reaches $0.714$ (rank 6/16) against\n"
     "$0.752$ for the best baseline; a twelve-setting sweep averages\n"
     "$0.727{\\pm}0.012$ and still trails it; validation cannot pick that setting\n"
     "(the configuration it selects scores $0.711$ on test, below the average over\n"
     "settings), so we report the setting-free average.",
     "Under one harness, one recipe and\n"
     "seeded initialisation for all sixteen models, two findings follow\n"
     "(Table~\\ref{tab:l3}).  (i) The in-domain split is \\emph{saturated} --- all\n"
     "sixteen models lie within $0.008$ AUROC, against per-seed sd $\\le0.003$ ---\n"
     "so ranking there is not informative, and ReliaDRP is last.  (ii) Drug-blind,\n"
     "where the protocol does discriminate, ReliaDRP is 6/16, behind CODE-AE; a\n"
     "twelve-setting sweep averages $0.727{\\pm}0.012$ and still trails it, and\n"
     "validation cannot pick that setting (the configuration it selects scores\n"
     "$0.711$ on test, below the average over settings), so we report the\n"
     "setting-free average."),
    # ---- L4: the table now carries the four directions -------------------
    ("(i) \\emph{Forward transfer into AML is a narrow,\n"
     "top-dense win}: ReliaDRP ranks 1/16 at $0.208{\\pm}0.021$ vs.\\ $0.197$ for\n"
     "CLCLSA --- a $+0.011$ margin inside a top five spanning $0.033$.  Seeding the\n"
     "initialisation (Section~\\ref{sec:lim}) rewrote this board: CRDNN and\n"
     "panCancerDR fall from ranks 2--3 ($0.222$/$0.218$) to $0.062$/$0.135$.\n"
     "(ii) \\emph{PDX costs less and we do not lead there}: its\n"
     "top five span $0.486$--$0.514$ (DELFOS best at $0.514$), and we sit 5/16 at\n"
     "$0.486$.",
     "(i) \\emph{Forward transfer into AML is a narrow,\n"
     "top-dense win}: ReliaDRP ranks 1/16, its $+0.011$ margin over the runner-up\n"
     "sitting inside a top five that spans $0.033$.  Seeding the initialisation\n"
     "(Section~\\ref{sec:lim}) rewrote this board: CRDNN and panCancerDR fall from\n"
     "ranks 2--3 ($0.222$/$0.218$) to $0.062$/$0.135$.  (ii) \\emph{PDX costs less\n"
     "and we do not lead there}: DELFOS is best and we sit 5/16."),
    # ---- Limitations (1): Table 3 carries the values now -----------------
    ("\\noindent\\textbf{(1) Single-cell ranking} is not where the model wins: under one\n"
     "harness ReliaDRP ranks 6/16 drug-blind ($0.714{\\pm}0.040$; $0.727$ over twelve\n"
     "settings) and last on a\n"
     "saturated in-domain split ($0.977$--$0.985$ for all sixteen models), where\n"
     "validation cannot select hyper-parameters (validation--test AUROC correlation\n"
     "$0.10$, or $0.34$ at batch $512$).",
     "\\noindent\\textbf{(1) Single-cell ranking} is not where the model wins: under one\n"
     "harness ReliaDRP is 6/16 drug-blind (Table~\\ref{tab:l3}; $0.727$ over twelve\n"
     "settings) and last on a saturated in-domain split, where\n"
     "validation cannot select hyper-parameters (validation--test AUROC correlation\n"
     "$0.10$, or $0.34$ at batch $512$)."),
    # ---- Discussion / Limitations: drop restatements ---------------------
    ("ReliaDRP ranks first on every L1 protocol, but a unified single-cell protocol\n"
     "removes an earlier first-place claim (L3, above) and re-bases the combination\n"
     "ranks (L2): evaluation fragility, not\n"
     "only model fragility.  ReliaDRPCombo is top-quartile on both OOD combination\n"
     "protocols, and GC-CP converts the in-domain marginal guarantee into a per-tier\n"
     "guarantee (tier gap $0.013$).",
     "ReliaDRP ranks first on every L1 protocol (Table~\\ref{tab:l1}), but a unified\n"
     "single-cell protocol removes an earlier first-place claim (L3) and re-bases the\n"
     "combination ranks (L2) --- evaluation fragility, not only model fragility.\n"
     "ReliaDRPCombo is top-quartile on both OOD combination protocols, and GC-CP\n"
     "gives a per-tier guarantee (tier gap $0.013$)."),
    ("\\textbf{(2) Benchmark defects were found\n"
     "and repaired}: a training-indexing bug and a recipe mismatch inflated early\n"
     "L2/L3 numbers (caught by a shuffled-label guard that returns to chance); both\n"
     "are re-run here under one harness (L2 holds, L3's claim is withdrawn).  A third\n"
     "defect --- unseeded\n"
     "\\emph{initialisation}, which decided the L4 board by lucky draws --- is fixed\n"
     "by re-runs at L1, L3 and L4.",
     "\\textbf{(2) Benchmark defects were found\n"
     "and repaired}: a training-indexing bug and a recipe mismatch inflated early\n"
     "L2/L3 numbers (caught by a shuffled-label guard that returns to chance); both\n"
     "are re-run under one harness (L2 holds, L3's claim is withdrawn), and a third\n"
     "defect --- unseeded \\emph{initialisation} --- is fixed by re-runs at L1, L3\n"
     "and L4."),
    # ---- two more restatements ------------------------------------------
    ("The protocol is fully unified (one basis for all directions, the\n"
     "L1 training contract, five seeds, 320 tabulated runs), with no missing cells\n"
     "(Table~\\ref{tab:l4}).",
     "The protocol is fully unified (one basis, the L1 training contract, five\n"
     "seeds, 320 tabulated runs), no missing cells (Table~\\ref{tab:l4})."),
    ("Seeds are $5$ for L1--L4 and our own hyper-parameter study; coverage studies\n"
     "use five calibration splits.  Model \\emph{initialisation} is seeded for every\n"
     "layer, so every reported number is a multi-seed mean reported with its seed\n"
     "spread, and sub-$0.03$ gaps are read as ties.",
     "Seeds are $5$ for L1--L4 and our own hyper-parameter study; coverage studies\n"
     "use five calibration splits.  Initialisation is seeded for every layer, so\n"
     "every number is a multi-seed mean with its seed spread, and sub-$0.03$ gaps\n"
     "are read as ties."),
]
for old, new in edits:
    assert tex.count(old) == 1, old[:70]
    tex = tex.replace(old, new, 1)

open(PAPER, "w", encoding="utf-8").write(tex)
print("single-col floats:", tex.count("\\begin{table}[!t]"),
      "| wide floats:", tex.count("\\begin{table*}[!t]"))
for lab in ("tab:l1", "tab:l2", "tab:l3", "tab:l4"):
    print("  ", lab, tex.count("\\label{%s}" % lab))
print("stale ref:", tex.count("tab:l1pearson"))
