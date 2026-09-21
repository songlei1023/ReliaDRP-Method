import os
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "processed_ext", "L3")
SEED = 42
rng = np.random.RandomState(SEED)

cells = np.load(os.path.join(D, "cells_l3.npy")).astype(np.float32)
y = np.load(os.path.join(D, "y_l3.npy"))
meta = pd.read_csv(os.path.join(D, "meta_l3.csv"))

Xb, yb, dsb = [], [], []
for ds, g in meta.groupby("dataset"):
    i = g.index.values
    yi = y[i]
    if len(np.unique(yi)) < 2:
        continue
    pos = i[yi == 1]
    neg = i[yi == 0]
    k = min(len(pos), len(neg))
    take = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    Xd = cells[take]
    mu = Xd.mean(0, keepdims=True)
    sd = Xd.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    Xd = (Xd - mu) / sd
    Xb.append(Xd.astype(np.float32))
    yb.append(y[take])
    dsb.append(np.array([ds] * len(take)))
Xb = np.vstack(Xb)
yb = np.concatenate(yb)
dsb = np.concatenate(dsb)
perm = rng.permutation(len(Xb))
Xb, yb, dsb = Xb[perm], yb[perm], dsb[perm]
print("balanced cells", Xb.shape, "pos rate", round(float(yb.mean()), 3),
      "datasets", len(np.unique(dsb)), flush=True)

out = os.path.join(D, "v2")
os.makedirs(os.path.join(out, "splits"), exist_ok=True)
np.save(os.path.join(out, "cells_l3.npy"), Xb)
np.save(os.path.join(out, "y_l3.npy"), yb)
pd.DataFrame({"dataset": dsb, "y": yb}).to_csv(os.path.join(out, "meta_l3.csv"), index=False)

n = len(Xb)
perm = rng.permutation(n)
n_te, n_va = int(0.1 * n), int(0.1 * n)
np.save(os.path.join(out, "splits", "L3_random_train.npy"), np.sort(perm[n_te + n_va:]))
np.save(os.path.join(out, "splits", "L3_random_val.npy"), np.sort(perm[:n_va]))
np.save(os.path.join(out, "splits", "L3_random_test.npy"), np.sort(perm[n_va:n_te + n_va]))
hold = list(rng.permutation(np.unique(dsb))[:max(1, int(0.2 * len(np.unique(dsb))))])
is_te = np.isin(dsb, hold)
rest = np.where(~is_te)[0]
te = np.where(is_te)[0]
nv = int(0.1 * len(rest))
np.save(os.path.join(out, "splits", "L3_ldo_train.npy"), np.sort(rest[nv:]))
np.save(os.path.join(out, "splits", "L3_ldo_val.npy"), np.sort(rest[:nv]))
np.save(os.path.join(out, "splits", "L3_ldo_test.npy"), np.sort(te))
print(json.dumps({"n": n, "held_out": hold, "pos_rate": round(float(yb.mean()), 3)}))
