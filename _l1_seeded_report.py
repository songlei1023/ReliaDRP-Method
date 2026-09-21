"""Analysis of the seeded L1 rerun (`l1_unified_seeded.csv`).

Produces, straight from the CSV:
  1. the per-protocol board (mean +- seed sd), with the historical value and
     the delta for every model;
  2. the mean-rank ordering;
  3. ready-to-paste LaTeX rows for the paper's L1 table;
  4. the val--test correlation per protocol (does selection even work?);
  5. the device audit (deepdtf's historical CPU fallback);
  6. the "setting-free" average per protocol (audit discipline).
"""

import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "l1_unified_seeded.csv")
OLD_EXT = os.path.join(HERE, "extended_benchmark_all.csv")
OLD_TUNE = os.path.join(HERE, "l1_v3_tune_confirm.csv")

PROTOS = ["EXT_random", "EXT_LDO", "EXT_LCO"]
PLABEL = {"EXT_random": "rnd", "EXT_LDO": "LDO", "EXT_LCO": "LCO"}
OURS = "reliadrpv3mix2mf03"

#: csv model name -> paper table name.  Order = paper table order.
PAPER = {
    "reliadrpv3mix2mf03": "ReliaDRP",
    "clclsa": "CLCLSA",
    "mlp": "MLP (multi-omics)",
    "delfos": "DELFOS",
    "attn": "AttnOmics",
    "pancancer": "panCancerDR",
    "mgataf": "MGATAF",
    "mmcl": "MMCL-CDR",
    "bandrp": "BANDRP",
    "codeae": "CODE-AE",
    "crdnn": "CRDNN",
    "fourierdrug": "FourierDrug",
    "vaen": "VAEN",
    "mograph": "MoGraphDRP",
    "tgsa": "TGSA",
    "deepdtf": "DeepDTF",
}


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load(path, rcol, mcol, pcol):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = num(r.get(rcol))
            if v is None:
                continue
            out.setdefault((r[pcol], r[mcol]), []).append(v)
    return out


def main():
    new_r = load(NEW, "test_r", "model", "proto")
    new_rmse = load(NEW, "test_rmse", "model", "proto")
    new_val = load(NEW, "val_r", "model", "proto")
    old = load(OLD_EXT, "pearson", "model", "protocol")
    old_tune = load(OLD_TUNE, "test_r", "config", "proto")

    L = []
    A = L.append
    A("# L1 seeded rerun --- analysis")
    A("")
    n_models = len({m for (_, m) in new_r})
    n_rows = sum(len(v) for v in new_r.values())
    A(f"- source: `{os.path.basename(NEW)}`  ({n_rows} rows, {n_models} models, "
      f"{len(PROTOS)} protocols, up to 5 seeds)")
    A(f"- historical baseline table: `{os.path.basename(OLD_EXT)}`; "
      f"historical row for ours: `{os.path.basename(OLD_TUNE)}` (config `mix2_mf03`)")
    A("")

    # ------------------------------------------------------------------ board
    board = {}
    for p in PROTOS:
        ms = {}
        for (pp, m), v in new_r.items():
            if pp == p:
                ms[m] = v
        ranked = sorted(ms.items(), key=lambda kv: -st.mean(kv[1]))
        board[p] = ranked
    # mean rank across protocols
    meanrank = {}
    for p in PROTOS:
        for i, (m, _) in enumerate(board[p], 1):
            meanrank.setdefault(m, []).append(i)
    mr = {m: st.mean(v) for m, v in meanrank.items()}

    A("## 1. Seeded board (Pearson, mean +- seed sd)")
    A("")
    head = "| rank | model | " + " | ".join(PLABEL[p] for p in PROTOS) \
           + " | mean rank |"
    A(head)
    A("|---|---|" + "---|" * (len(PROTOS) + 1))
    order = sorted(new_r and {m for (_, m) in new_r}, key=lambda m: mr.get(m, 99))
    for i, m in enumerate(order, 1):
        cells = []
        for p in PROTOS:
            v = new_r.get((p, m))
            cells.append("---" if not v else "%.4f+-%.4f" % (st.mean(v), st.stdev(v)))
        tag = " **<- ours**" if m == OURS else ""
        A("| %d | `%s`%s | %s | %.2f |" % (i, m, tag, " | ".join(cells), mr.get(m, float('nan'))))
    A("")

    # ------------------------------------------------------- ours vs baselines
    A("## 2. Our row versus the strongest baseline (seeded)")
    A("")
    A("| protocol | ours | strongest external | gap | rank | top-5 spread | verdict |")
    A("|---|---|---|---|---|---|---|")
    for p in PROTOS:
        rk = board[p]
        myv = st.mean(new_r[(p, OURS)])
        myrank = [i for i, (m, _) in enumerate(rk, 1) if m == OURS][0]
        ext = [(m, st.mean(v)) for m, v in rk if m != OURS]
        top5 = [st.mean(v) for _, v in rk[:5]]
        gap = myv - ext[0][1]
        sd_ours = st.stdev(new_r[(p, OURS)])
        se = sd_ours / (len(new_r[(p, OURS)]) ** 0.5)
        verdict = ("win" if gap > 3 * se else
                   "tie (within seed noise)" if abs(gap) <= 3 * se else "lose")
        A("| %s | %.4f+-%.4f | %s %.4f | %+.4f | %d/%d | %.4f | %s (3SE=%.3f) |"
          % (PLABEL[p], myv, sd_ours, ext[0][0], ext[0][1], gap, myrank,
             len(rk), top5[0] - top5[-1], verdict, 3 * se))
    A("")

    # -------------------------------------------------------- old vs new (all)
    A("## 3. Seeded versus historical value, every model")
    A("")
    A("| model | " + " | ".join("%s old -> new" % PLABEL[p] for p in PROTOS) + " |")
    A("|---|" + "---|" * len(PROTOS))
    for m in sorted(new_r and {x for (_, x) in new_r}, key=lambda z: mr.get(z, 99)):
        cells = []
        for p in PROTOS:
            nv = st.mean(new_r[(p, m)]) if (p, m) in new_r else None
            if m == OURS:
                ov = st.mean(old_tune[(p, "mix2_mf03")]) if (p, "mix2_mf03") in old_tune else None
            else:
                ov = st.mean(old[(p, m)]) if (p, m) in old else None
            cells.append("---" if (nv is None or ov is None)
                         else "%.4f -> **%.4f** (%+.4f)" % (ov, nv, nv - ov))
        A("| `%s` | %s |" % (m, " | ".join(cells)))
    A("")

    # ------------------------------------------------------------ val vs test
    A("## 4. Selection sanity: val--test correlation per protocol")
    A("")
    for p in PROTOS:
        vv, tt = [], []
        for (pp, m), v in new_val.items():
            if pp != p:
                continue
            for a, b in zip(v, new_r[(p, m)]):
                vv.append(a)
                tt.append(b)
        r = st.correlation(vv, tt) if len(vv) > 2 else float("nan")
        A(f"- **{PLABEL[p]}**: r(val, test) = {r:+.3f} over {len(vv)} runs "
          f"({'usable' if r > 0.4 else 'weak'})")
    A("")

    # --------------------------------------------------------------- devices
    A("## 5. Device audit")
    A("")
    dev = {}
    with open(NEW, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if num(r.get("test_r")) is None:
                continue
            dev.setdefault((r["model"], r.get("device", "?")), 0)
            dev[(r["model"], r.get("device", "?"))] += 1
    off = {k: v for k, v in dev.items() if k[1] != "cuda"}
    if off:
        for (m, d), n in sorted(off.items()):
            A(f"- `{m}` ran on **{d}** for {n} runs (historical CPU fallback for "
              f"`CUDA error: invalid configuration argument`).")
    else:
        A("- every run on cuda")
    A("")

    # -------------------------------------------------- setting-free average
    A("## 6. Setting-free average (audit discipline)")
    A("")
    A("Mean over the five seeds is reported above; the setting-free mean per")
    A("protocol over *every* model in the table is:")
    A("")
    for p in PROTOS:
        allv = [st.mean(v) for (pp, m), v in new_r.items() if pp == p]
        A("- %s: n=%d models, mean %.4f, sd across models %.4f"
          % (PLABEL[p], len(allv), st.mean(allv), st.stdev(allv)))
    A("")

    # ------------------------------------------------------------ latex rows
    A("## 7. LaTeX body for the paper table")
    A("")
    A("```latex")
    for m in order:
        if m not in PAPER:
            continue
        bold = (m == OURS)

        def cell(txt):
            return "$\\mathbf{%s}$" % txt if bold else "$%s$" % txt

        pc = [cell("%.3f{\\pm}%.3f" % (st.mean(new_r[(p, m)]),
                                       st.stdev(new_r[(p, m)]))) for p in PROTOS]
        rc = [cell("%.3f" % st.mean(new_rmse[(p, m)])) for p in PROTOS]
        name = "\\textbf{ReliaDRP}" if bold else PAPER[m]
        rk = "$\\mathbf{%.2f}$" % mr[m] if bold else "%.2f" % mr[m]
        A("%-17s & %s & %s & %s \\\\"
          % (name, " & ".join(pc), " & ".join(rc), rk))
    A("```")
    A("")

    out = os.path.join(HERE, "_l1_seeded_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
