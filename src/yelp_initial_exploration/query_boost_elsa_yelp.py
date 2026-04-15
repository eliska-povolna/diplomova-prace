#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_scores(scores_path: Path, n_items: int):
    if scores_path.suffix == ".npy":
        s = np.load(scores_path).reshape(-1)
        if len(s) != n_items:
            raise ValueError("Score length mismatch vs items")
        return s
    else:
        df = pd.read_csv(scores_path)
        s = np.zeros(n_items, dtype=np.float32)
        s[df["i"].astype(int).values] = df["score"].values.astype(np.float32)
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--items_csv",
        type=Path,
        required=True,
        help="CSV with columns i,business_id,tag_str,name",
    )
    ap.add_argument("--scores", type=Path, required=True, help=".npy or .csv (i,score)")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument(
        "--alpha", type=float, default=0.7, help="weight for base ELSA score"
    )
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    items = pd.read_csv(args.items_csv)
    tags = items["tag_str"].fillna("").tolist()

    vec = TfidfVectorizer()
    X = vec.fit_transform(tags)  # (I,V)
    qv = vec.transform([args.query])  # (1,V)
    sim = cosine_similarity(X, qv).reshape(-1)  # (I,)

    base = load_scores(args.scores, len(items))
    final = args.alpha * base + (1.0 - args.alpha) * sim

    top_idx = np.argsort(-final)[: args.topk]
    out = items.loc[top_idx, ["i", "business_id", "name", "tag_str"]].copy()
    out["base_score"] = base[top_idx]
    out["query_sim"] = sim[top_idx]
    out["final"] = final[top_idx]
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
