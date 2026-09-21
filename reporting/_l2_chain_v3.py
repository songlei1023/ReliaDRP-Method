"""Chained low-impact runner for the L2 v3 study.

1. waits until the v3 screen (_l2_screen_v3.csv) has all 3 protocols x 2
   variants (or until the screen process is gone),
2. picks the variant with the best mean Pearson across protocols,
3. launches the 5-seed confirmation at below-normal priority with 2 threads.
"""

import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
SCREEN = os.path.join(HERE, "_l2_screen_v3.csv")
PROTOS = ["L2_random", "L2_LCO", "L2_LDO"]
VARIANTS = ["v3mix1", "v3mix2"]


def screen_running():
    try:
        import psutil
    except Exception:
        return True
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if "run_l2_v3_screen.py" in cl:
            return True
    return False


def load_rows():
    if not os.path.exists(SCREEN):
        return pd.DataFrame()
    try:
        return pd.read_csv(SCREEN)
    except Exception:
        return pd.DataFrame()


def main():
    deadline = time.time() + 6 * 3600
    while True:
        df = load_rows()
        have = {(r.proto, r.variant) for r in df.itertuples()} if len(df) else set()
        complete = all((p, v) in have for p in PROTOS for v in VARIANTS)
        if complete:
            print(f"screen complete: {len(df)} rows", flush=True)
            break
        if not screen_running():
            print(f"screen process gone with {len(df)} rows; proceeding", flush=True)
            break
        if time.time() > deadline:
            print("watcher timeout; proceeding with what we have", flush=True)
            break
        time.sleep(30)

    df = load_rows()
    if df.empty:
        print("no screen results at all; aborting", flush=True)
        return
    best, best_score = None, -9
    for v in VARIANTS:
        sub = df[df["variant"] == v]
        if sub.empty:
            continue
        n_proto = sub["proto"].nunique()
        score = float(sub["test_r"].mean()) + 0.01 * n_proto  # prefer full coverage
        print(f"variant {v}: n_proto={n_proto} mean_test_r={sub['test_r'].mean():.4f}",
              flush=True)
        if score > best_score:
            best, best_score = v, score
    print(f"selected variant: {best}", flush=True)

    cmd = [sys.executable, os.path.join(HERE, "l2", "run_l2_v3_confirm.py"), "--variant", best,
           "--seeds", "0,1,2,3,4", "--threads", "2", "--low-priority"]
    print("launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=False)
    print("confirm finished", flush=True)


if __name__ == "__main__":
    main()
