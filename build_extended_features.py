import argparse
import json
import os

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data\_drevalpy"
SOURCES = ["CCLE", "CTRPv1", "GDSC1", "BeatAML2", "PDX_Bruna"]
MODALITIES = [("expr", "gene_expression.csv"),
              ("cnv", "copy_number_variation_gistic.csv"),
              ("mut", "mutations.csv")]
N_BITS = 2048


def load_matrix(path):
    e = pd.read_csv(path, index_col=0, low_memory=False)
    if "cell_line_name" in e.columns:
        e = e.drop(columns=["cell_line_name"])
    e.index = e.index.astype(str)
    e = e[~e.index.duplicated(keep="first")]
    return e


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
    ap.add_argument("--ext", default=os.path.normpath(os.path.join(here, "data", "processed_ext", "response_ext.csv")))
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(here, "data", "processed_ext")))
    args = ap.parse_args()

    ext = pd.read_csv(args.ext, dtype={"cell_id": str, "drug_id": str})
    mats = {}
    channel_genes = {}
    for name, fname in MODALITIES:
        loaded = {}
        for ds in SOURCES:
            p = os.path.join(RAW, ds, fname)
            if os.path.exists(p):
                loaded[ds] = load_matrix(p)
        if not loaded:
            continue
        common = sorted(set.intersection(*[set(m.columns) for m in loaded.values()]))
        if name == "expr":
            common = common[:]  # keep all shared genes for expression
        mats[name] = {ds: m[common] for ds, m in loaded.items()}
        channel_genes[name] = common
        print(f"{name}: datasets={list(loaded)} genes={len(common)}", flush=True)

    cell_key_rows, cell_meta = [], []
    for ds in SOURCES:
        want = ext.loc[ext.dataset == ds, "cell_id"].unique()
        ref = None
        for name, _ in MODALITIES:
            if name in mats and ds in mats[name]:
                ref = mats[name][ds]
                break
        rows = [c for c in want if c in set(ref.index)]
        for cid in rows:
            vecs = []
            for name, _ in MODALITIES:
                m = mats.get(name, {}).get(ds)
                if m is not None and cid in m.index:
                    vecs.append(m.loc[cid].to_numpy(np.float32))
                else:
                    vecs.append(np.zeros(len(channel_genes[name]), np.float32))
            cell_key_rows.append(np.concatenate(vecs))
            cell_meta.append({"dataset": ds, "cell_id": str(cid)})
    cells_multi = np.vstack(cell_key_rows).astype(np.float32)
    cells_multi = np.nan_to_num(cells_multi, nan=0.0)
    np.save(os.path.join(args.outdir, "cells_multi_raw_ext.npy"), cells_multi)
    pd.DataFrame(cell_meta).to_csv(os.path.join(args.outdir, "cell_index_ext.csv"), index=False)

    drugs = ext[["drug_id", "smiles"]].dropna(subset=["drug_id"]).drop_duplicates("drug_id")
    feats, keep = [], []
    for r in drugs.itertuples():
        fp = morgan(r.smiles) if isinstance(r.smiles, str) else None
        feats.append(fp if fp is not None else np.zeros(N_BITS, np.float32))
        keep.append(r.drug_id)
    np.save(os.path.join(args.outdir, "drug_raw_ext.npy"), np.vstack(feats))
    pd.DataFrame({"drug_id": keep}).to_csv(os.path.join(args.outdir, "drug_index_ext.csv"), index=False)

    summary = {
        "channel_genes": {k: len(v) for k, v in channel_genes.items()},
        "cell_rows": int(cells_multi.shape[0]),
        "cell_raw_dim": int(cells_multi.shape[1]),
        "drug_rows": int(len(keep)),
        "modality_by_dataset": {ds: [name for name, _ in MODALITIES if name in mats and ds in mats[name]]
                                for ds in SOURCES},
    }
    with open(os.path.join(args.outdir, "ext_features_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
