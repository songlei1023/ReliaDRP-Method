import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SEED = 42
MODALITIES = ["expr", "cnv", "mut"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.normpath(os.path.join(here, "data", "processed_ext")))
    ap.add_argument("--chan_dim", type=int, default=128)
    ap.add_argument("--drug_dim", type=int, default=128)
    args = ap.parse_args()
    d = args.extdir
    spl = os.path.join(d, "splits")
    os.makedirs(spl, exist_ok=True)

    ext = pd.read_csv(os.path.join(d, "response_ext.csv"), dtype={"cell_id": str, "drug_id": str})
    cell_meta = pd.read_csv(os.path.join(d, "cell_index_ext.csv"), dtype=str)
    drug_meta = pd.read_csv(os.path.join(d, "drug_index_ext.csv"), dtype=str)
    cells_raw = np.load(os.path.join(d, "cells_multi_raw_ext.npy"))
    drugs_raw = np.load(os.path.join(d, "drug_raw_ext.npy"))
    with open(os.path.join(d, "ext_features_summary.json"), encoding="utf-8") as f:
        chan = json.load(f)["channel_genes"]
    sizes = [chan[m] for m in MODALITIES]
    offs = np.cumsum([0] + sizes)

    cell_key = {(r.dataset, r.cell_id): i for i, r in enumerate(cell_meta.itertuples())}
    drug_key = {r.drug_id: i for i, r in enumerate(drug_meta.itertuples())}
    ext["cell_row"] = [cell_key.get((a, b), -1) for a, b in zip(ext.dataset, ext.cell_id)]
    ext["drug_row"] = [drug_key.get(a, -1) for a in ext.drug_id]
    elig = ext[(ext.cell_row >= 0) & (ext.drug_row >= 0)].reset_index(drop=True)
    np.save(os.path.join(d, "pair_cell_row.npy"), elig["cell_row"].to_numpy(np.int64))
    np.save(os.path.join(d, "pair_drug_row.npy"), elig["drug_row"].to_numpy(np.int64))
    elig[["dataset", "cell_id", "drug_id", "y_auc", "r2_mean", "n_rep", "smiles"]].to_csv(
        os.path.join(d, "response_ext_eligible.csv"), index=False)

    n = len(elig)
    rng = np.random.RandomState(SEED)
    info = {"n_pairs_total": int(len(ext)), "n_pairs_eligible": int(n),
            "channel_sizes": dict(zip(MODALITIES, sizes))}

    def transform_cells(tr_rows):
        out = []
        for k, m in enumerate(MODALITIES):
            sl = cells_raw[:, offs[k]:offs[k + 1]]
            p = PCA(n_components=min(args.chan_dim, len(tr_rows), sl.shape[1]),
                    random_state=SEED).fit(sl[tr_rows])
            out.append(p.transform(sl).astype(np.float32))
            info.setdefault("explained", {}).setdefault(m, []).append(round(float(p.explained_variance_ratio_.sum()), 4))
        return np.hstack(out)

    def emit(name, tr, va, te):
        tr_cell = np.unique(elig["cell_row"].to_numpy()[tr])
        cells = transform_cells(tr_cell)
        dp = PCA(n_components=min(args.drug_dim, drugs_raw.shape[0]), random_state=SEED).fit(drugs_raw)
        np.save(os.path.join(spl, f"{name}_train.npy"), tr)
        np.save(os.path.join(spl, f"{name}_val.npy"), va)
        np.save(os.path.join(spl, f"{name}_test.npy"), te)
        np.save(os.path.join(spl, f"{name}_cells.npy"), cells)
        np.save(os.path.join(spl, f"{name}_drugs.npy"), dp.transform(drugs_raw).astype(np.float32))
        return {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                "cell_dim": int(cells.shape[1]), "drug_dim": int(args.drug_dim),
                "drug_var": round(float(dp.explained_variance_ratio_.sum()), 4)}

    perm = rng.permutation(n)
    n_te, n_va = int(0.1 * n), int(0.1 * n)
    info["EXT_random"] = emit("EXT_random", np.sort(perm[n_te + n_va:]), np.sort(perm[:n_va]), np.sort(perm[n_va:n_te + n_va]))

    for kind in ["EXT_LDO", "EXT_LCO"]:
        if kind == "EXT_LDO":
            groups = elig["drug_id"].unique()
        else:
            elig["g"] = elig["dataset"].astype(str) + "|" + elig["cell_id"].astype(str)
            groups = elig["g"].unique()
        hold = set(rng.permutation(groups)[:int(0.2 * len(groups))])
        is_test = (elig["drug_id"] if kind == "EXT_LDO" else elig["g"]).isin(hold).to_numpy()
        idx = np.arange(n)
        te = idx[is_test]
        rest = idx[~is_test]
        nv = int(0.1 * len(rest))
        info[kind] = emit(kind, np.sort(rest[nv:]), np.sort(rest[:nv]), np.sort(te))

    with open(os.path.join(spl, "ext_splits_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
