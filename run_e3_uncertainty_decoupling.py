import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXPROOT = os.path.normpath(os.path.join(HERE, "..", "new_data", "exp"))
sys.path.insert(0, EXPROOT)

from models import BaseDRP, pred_evidential  # noqa: E402
from train_utils import load_protocol  # noqa: E402

SPL = os.path.join(DATA, "splits")
PRED = os.path.join(DATA, "results", "preds")
MODELS = os.path.join(DATA, "results", "models")
PROTOCOLS = ["P0", "P1", "P2"]


def load_rel(path):
    r = pd.read_csv(path, usecols=["reliability_tier", "reliability_w"])
    return r["reliability_tier"].to_numpy(), r["reliability_w"].to_numpy(float)


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def within_domain(rel_path):
    tier, w = load_rel(rel_path)
    rows = []
    for P in PROTOCOLS:
        f = os.path.join(PRED, f"{P}_evi.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ale, epi, y, mu = d["ale"], d["epi"], d["y"], d["mu"]
        te = np.load(os.path.join(SPL, f"{P}_test.npy"))
        w_te, tier_te = w[te], tier[te]
        err = np.abs(y - mu)
        rows.append({
            "protocol": P, "n": len(te),
            "spearman_ale_vs_failrel": round(spearman(ale, 1 - w_te), 4),
            "spearman_epi_vs_failrel": round(spearman(epi, 1 - w_te), 4),
            "spearman_ale_vs_err": round(spearman(ale, err), 4),
            "spearman_epi_vs_err": round(spearman(epi, err), 4),
            "spearman_ale_vs_epi": round(spearman(ale, epi), 4),
            "mean_ale": round(float(ale.mean()), 5),
            "mean_epi": round(float(epi.mean()), 5),
        })
        by_tier = {}
        for t in ["high", "mid", "low"]:
            m = tier_te == t
            if m.sum() < 10:
                continue
            by_tier[t] = {
                "n": int(m.sum()),
                "mean_ale": round(float(ale[m].mean()), 5),
                "mean_epi": round(float(epi[m].mean()), 5),
                "spearman_ale_epi": round(spearman(ale[m], epi[m]), 4),
            }
        rows[-1]["by_tier"] = by_tier
    return rows


def ood_decomposition(id_frac=0.5):
    tr, va, te0, cell_idx, drug_idx, y, cells, drugs = load_protocol("P0")
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))
    model = BaseDRP(cells.shape[1], drugs.shape[1], head="evidential")
    model.load_state_dict(torch.load(os.path.join(MODELS, "P0_evi.pt")))
    model.eval()

    def infer(rows):
        with torch.no_grad():
            mu, ale, epi, tot = pred_evidential(
                model, tc[torch.from_numpy(cell_idx[rows].astype(np.int64))],
                td[torch.from_numpy(drug_idx[rows].astype(np.int64))])
        return ale, epi

    ale_id, epi_id = infer(te0)
    out = []
    for ood in ["P1", "P2"]:
        tex = np.load(os.path.join(SPL, f"{ood}_test.npy"))
        ale_od, epi_od = infer(tex)
        n = min(len(ale_id), len(ale_od))
        rng = np.random.RandomState(0)
        a_id = rng.choice(len(ale_id), n, replace=False)
        a_od = rng.choice(len(ale_od), n, replace=False)
        lbl = np.concatenate([np.zeros(n), np.ones(n)])
        out.append({
            "setting": f"ID(P0test)_vs_{ood}test", "n_each": int(n),
            "auc_ale": round(float(roc_auc_score(lbl, np.concatenate([ale_id[a_id], ale_od[a_od]]))), 4),
            "auc_epi": round(float(roc_auc_score(lbl, np.concatenate([epi_id[a_id], epi_od[a_od]]))), 4),
            "mean_ale_id": round(float(ale_id[a_id].mean()), 5),
            "mean_ale_ood": round(float(ale_od[a_od].mean()), 5),
            "mean_epi_id": round(float(epi_id[a_id].mean()), 5),
            "mean_epi_ood": round(float(epi_od[a_od].mean()), 5),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default=os.path.join(HERE, "reliability", "response_reliability.csv"))
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    wd = within_domain(args.rel)
    oo = ood_decomposition()
    res = {"within_domain": wd, "ood_decomposition": oo}
    with open(os.path.join(args.outdir, "e3_uncertainty_decoupling.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "by_tier"} for r in wd])
    flat.to_csv(os.path.join(args.outdir, "e3_within_domain.csv"), index=False)
    pd.DataFrame(oo).to_csv(os.path.join(args.outdir, "e3_ood_decomposition.csv"), index=False)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
