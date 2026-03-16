
#!/usr/bin/env python3
import argparse, json, pickle
from pathlib import Path
import duckdb, pandas as pd, numpy as np

SEL_ATTR_PREFIXES = ("Restaurants", "GoodForKids", "GoodForGroups", "OutdoorSeating", "WiFi", "BikeParking")

def extract_item_tags(df: pd.DataFrame) -> pd.DataFrame:
    def split_cats(x):
        if pd.isna(x): return []
        return [c.strip() for c in str(x).split(",") if c.strip()]

    def attrs_to_tokens(attrs):
        toks = []
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if any(k.startswith(p) for p in SEL_ATTR_PREFIXES):
                    if isinstance(v, (str, int, float, bool)):
                        val = str(v).strip()
                        if val and val.lower() not in ("none","null","false","0","no"):
                            toks.append(f"attr:{k}={val}")
        return toks

    tags = []
    for _, row in df.iterrows():
        t = []
        for c in split_cats(row.get("categories", None)):
            t.append(f"category:{c}")
        at = row.get("attributes", None)
        if isinstance(at, str):
            try:
                import json as _json
                at = _json.loads(at)
            except Exception:
                at = None
        t += attrs_to_tokens(at if isinstance(at, dict) else {})
        tags.append(" ".join(sorted(set(t))))
    return pd.DataFrame({"business_id": df["business_id"].values, "name": df.get("name",""), "tag_str": tags})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", type=Path, required=True)
    ap.add_argument("--item2index", type=Path, required=True)
    ap.add_argument("--neuron_item_matrix", type=Path, required=True, help=".npy, shape (H,I) or (I,H)")
    ap.add_argument("--top_items_per_neuron", type=int, default=300)
    ap.add_argument("--top_tags_per_neuron", type=int, default=30)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_item_tags_csv", type=Path, required=True)
    args = ap.parse_args()

    con = duckdb.connect()
    business_glob = (args.parquet_dir / "business" / "state=*/*.parquet").as_posix()
    dfb = con.execute(f"SELECT business_id, name, categories, attributes FROM read_parquet('{business_glob}')").fetchdf()
    item_tags = extract_item_tags(dfb)

    with args.item2index.open("rb") as f:
        item2idx = pickle.load(f)
    item_df = pd.DataFrame({"business_id": list(item2idx.keys()), "i": list(item2idx.values())})
    item_meta = item_df.merge(item_tags, on="business_id", how="left").sort_values("i")
    tag_strs = item_meta["tag_str"].fillna("").tolist()

    W = np.load(args.neuron_item_matrix)
    if W.ndim != 2:
        raise ValueError("neuron_item_matrix must be 2D")
    H, I = W.shape
    if I != len(tag_strs) and H == len(tag_strs):
        W = W.T
        H, I = W.shape
    if I != len(tag_strs):
        raise ValueError(f"Items mismatch: matrix I={I} vs tags {len(tag_strs)}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"[^ ]+")
    X = vec.fit_transform(tag_strs)  # (I,V)
    vocab = np.array(vec.get_feature_names_out())

    out = {}
    for h in range(H):
        scores = W[h]  # (I,)
        top_idx = np.argpartition(scores, -args.top_items_per_neuron)[-args.top_items_per_neuron:]
        mean_tfidf = X[top_idx].mean(axis=0).A1
        top_tag_idx = np.argsort(-mean_tfidf)[:args.top_tags_per_neuron]
        tags = [(vocab[j], float(mean_tfidf[j])) for j in top_tag_idx if mean_tfidf[j] > 0]
        out[str(h)] = {"top_tags": tags}

    item_meta.to_csv(args.out_item_tags_csv, index=False)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", args.out_json, args.out_item_tags_csv)

if __name__ == "__main__":
    main()
