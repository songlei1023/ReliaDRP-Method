import argparse
import json
import os
import re

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data\raw"
DC = r"D:\BI\DRP\HDRP\data\_drugcomb"
KEY = ["drug_row", "drug_col", "cell_line_name"]
N_BITS = 2048
SEED = 42


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def morgan(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=N_BITS), dtype=np.float32)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=os.path.normpath(os.path.join(here, "data", "processed_ext")))
    ap.add_argument("--n_genes", type=int, default=3000)
    args = ap.parse_args()
    outd = os.path.join(args.ext, "L2")
    os.makedirs(outd, exist_ok=True)

    rel = pd.read_csv(os.path.join(args.ext, "L2_drugcomb_reliability.csv"),
                      dtype={"drug_row": str, "drug_col": str, "cell_line_name": str})
    print("combos", len(rel), flush=True)

    cells = pd.read_excel(os.path.join(DC, "DrugComb_cell_line_identifiers.xlsx"))
    cmap = {}
    for r in cells.itertuples():
        ach = str(r.depmap_id).strip()
        if not ach.startswith("ACH"):
            continue
        cmap.setdefault(norm(r.name), ach)
        if isinstance(r.synonyms, str):
            for s in r.synonyms.split(";"):
                if s.strip():
                    cmap.setdefault(norm(s), ach)
    drugs = pd.read_excel(os.path.join(DC, "DrugComb_drug_identifiers.xlsx"))
    dmap = {}
    for r in drugs.itertuples():
        if not isinstance(r.smiles, str) or not r.smiles:
            continue
        dmap.setdefault(norm(r.dname), r.smiles)
        if isinstance(r.synonyms, str):
            for s in r.synonyms.split(";"):
                if s.strip():
                    dmap.setdefault(norm(s), r.smiles)
    print("cell map", len(cmap), "drug map", len(dmap), flush=True)

    rel["ach"] = rel["cell_line_name"].map(lambda x: cmap.get(norm(x)))
    rel["sm_row"] = rel["drug_row"].map(lambda x: dmap.get(norm(x)))
    rel["sm_col"] = rel["drug_col"].map(lambda x: dmap.get(norm(x)))
    keep = rel.dropna(subset=["ach", "sm_row", "sm_col"]).reset_index(drop=True)
    print("mapped combos", len(keep), f"({len(keep)/len(rel):.3f})", flush=True)

    cell_ids = sorted(keep["ach"].unique())
    drug_names = sorted(set(keep["drug_row"]) | set(keep["drug_col"]))
    smi = {}
    for n in drug_names:
        s = dmap.get(norm(n))
        if s:
            smi[n] = s
    drug_ids = sorted(smi)
    cell_ix = {a: i for i, a in enumerate(cell_ids)}
    drug_ix = {d: i for i, d in enumerate(drug_ids)}
    keep["cell_idx"] = keep["ach"].map(cell_ix)
    keep["drow_idx"] = keep["drug_row"].map(drug_ix)
    keep["dcol_idx"] = keep["drug_col"].map(drug_ix)
    keep = keep.dropna(subset=["drow_idx", "dcol_idx"]).reset_index(drop=True)
    keep[["cell_idx", "drow_idx", "dcol_idx"]] = keep[["cell_idx", "drow_idx", "dcol_idx"]].astype(int)

    expr = pd.read_csv(os.path.join(RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"), low_memory=False)
    expr = expr.rename(columns={expr.columns[0]: "ModelID"}).set_index("ModelID")
    expr.index = expr.index.astype(str).str.strip()
    have = [a for a in cell_ids if a in expr.index]
    expr = expr.loc[have]
    genes = list(expr.columns)
    var = expr.var(0).to_numpy()
    top = np.argsort(var)[::-1][:args.n_genes]
    genes = [genes[i] for i in top]
    expr = expr[genes]
    cells_mat = expr.to_numpy(np.float32)
    cell_ids = list(expr.index)
    cell_ix = {a: i for i, a in enumerate(cell_ids)}
    keep = keep[keep["ach"].isin(cell_ix)].reset_index(drop=True)
    keep["cell_idx"] = keep["ach"].map(cell_ix)

    np.save(os.path.join(outd, "cells_expr.npy"), cells_mat)
    pd.DataFrame({"cell_id": cell_ids}).to_csv(os.path.join(outd, "cell_index.csv"), index=False)
    with open(os.path.join(outd, "genes_l2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(genes))

    feats = []
    for d in drug_ids:
        fp = morgan(smi[d])
        feats.append(fp if fp is not None else np.zeros(N_BITS, np.float32))
    drug_feat = np.vstack(feats).astype(np.float32)
    np.save(os.path.join(outd, "drug_morgan.npy"), drug_feat)
    pd.DataFrame({"drug": drug_ids, "smiles": [smi[d] for d in drug_ids]}).to_csv(
        os.path.join(outd, "drug_index.csv"), index=False)

    keep = keep.reset_index(drop=True)
    keep.to_csv(os.path.join(outd, "response_l2.csv"), index=False)

    n = len(keep)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_te, n_va = int(0.1 * n), int(0.1 * n)
    splits = {"L2_random": (perm[n_te + n_va:], perm[:n_va], perm[n_va:n_te + n_va])}
    cell_groups = keep["cell_idx"].unique()
    hold = set(rng.permutation(cell_groups)[:int(0.2 * len(cell_groups))])
    is_te = keep["cell_idx"].isin(hold).to_numpy()
    rest = np.where(~is_te)[0]
    te = np.where(is_te)[0]
    nv = int(0.1 * len(rest))
    splits["L2_LCO"] = (rest[nv:], rest[:nv], te)
    drug_groups = pd.concat([keep["drow_idx"], keep["dcol_idx"]]).unique()
    hold = set(rng.permutation(drug_groups)[:int(0.2 * len(drug_groups))])
    is_te = (keep["drow_idx"].isin(hold) | keep["dcol_idx"].isin(hold)).to_numpy()
    rest = np.where(~is_te)[0]
    te = np.where(is_te)[0]
    nv = int(0.1 * len(rest))
    splits["L2_LDO"] = (rest[nv:], rest[:nv], te)
    os.makedirs(os.path.join(outd, "splits"), exist_ok=True)
    for k, (tr, va, te) in splits.items():
        np.save(os.path.join(outd, "splits", f"{k}_train.npy"), np.sort(tr))
        np.save(os.path.join(outd, "splits", f"{k}_val.npy"), np.sort(va))
        np.save(os.path.join(outd, "splits", f"{k}_test.npy"), np.sort(te))

    summ = {"combos_total": int(len(rel)), "combos_mapped": int(n),
            "mapping_rate": round(n / len(rel), 4), "cells": int(keep.cell_idx.nunique()),
            "drugs": int(pd.concat([keep.drow_idx, keep.dcol_idx]).nunique()),
            "n_genes": len(genes), "splits": {k: [len(v[0]), len(v[1]), len(v[2])] for k, v in splits.items()}}
    with open(os.path.join(outd, "l2_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
