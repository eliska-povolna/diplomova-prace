
#!/usr/bin/env python3
import argparse
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix, save_npz

def build_id_map(series: pd.Series):
    uniq = series.drop_duplicates().reset_index(drop=True)
    return pd.Series(index=uniq.values, data=np.arange(len(uniq)), name="idx")

def main():
    ap = argparse.ArgumentParser(description="Build CSR and ID maps from Yelp Parquet + DuckDB")
    ap.add_argument("--parquet_dir", type=Path, required=True, help="Dir with parquet (business/, review/)")
    ap.add_argument("--db", type=Path, default=Path("yelp.duckdb"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pos_threshold", type=float, default=4.0, help="stars >= threshold => implicit 1")
    ap.add_argument("--any_review", action="store_true", help="If set, any review counts as implicit 1")
    ap.add_argument("--state_filter", type=str, default=None, help="Optional: only this state code")
    ap.add_argument("--year_min", type=int, default=None)
    ap.add_argument("--year_max", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.db))

    business_glob = (args.parquet_dir / "business" / "state=*/*.parquet").as_posix()
    review_glob = (args.parquet_dir / "review" / "year=*/*.parquet").as_posix()

    where = ["user_id IS NOT NULL", "business_id IS NOT NULL"]
    if not args.any_review:
        where.append(f"TRY_CAST(stars AS DOUBLE) >= {args.pos_threshold}")
    if args.state_filter:
        state_filter = args.state_filter.replace("'", "''")
        business_view = f"""
        SELECT business_id FROM read_parquet('{business_glob}') WHERE state = '{state_filter}'
        """
        con.execute("CREATE OR REPLACE TEMP VIEW business_f AS " + business_view)
        where.append("business_id IN (SELECT business_id FROM business_f)")

    if args.year_min or args.year_max:
        year_cond = []
        if args.year_min:
            year_cond.append(f"CAST(strftime(date, '%Y') AS INTEGER) >= {int(args.year_min)}")
        if args.year_max:
            year_cond.append(f"CAST(strftime(date, '%Y') AS INTEGER) <= {int(args.year_max)}")
        where.append("(" + " AND ".join(year_cond) + ")")

    where_sql = " AND ".join(where)
    q = f"""
    SELECT user_id, business_id, epoch_ms(CAST(date AS TIMESTAMP)) AS ts, 1 AS implicit
    FROM read_parquet('{review_glob}')
    WHERE {where_sql}
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        raise SystemExit("No interactions after filters. Relax filters or check paths.")

    uid_map = build_id_map(df["user_id"])
    iid_map = build_id_map(df["business_id"])

    with (args.out_dir / "user2index.pkl").open("wb") as f:
        pickle.dump(uid_map.to_dict(), f)
    with (args.out_dir / "item2index.pkl").open("wb") as f:
        pickle.dump(iid_map.to_dict(), f)

    df["u"] = df["user_id"].map(uid_map)
    df["i"] = df["business_id"].map(iid_map)
    df = df.dropna(subset=["u","i"])
    df["u"] = df["u"].astype(int)
    df["i"] = df["i"].astype(int)

    out_interactions = args.out_dir / "interactions_filtered.parquet"
    df[["user_id","business_id","ts","implicit","u","i"]].to_parquet(out_interactions, compression="zstd", index=False)

    n_users = int(df["u"].max()) + 1
    n_items = int(df["i"].max()) + 1
    vals = np.ones(len(df), dtype=np.float32)
    R = coo_matrix((vals, (df["u"].values, df["i"].values)), shape=(n_users, n_items)).tocsr()
    save_npz(args.out_dir / "processed_train.npz", R)

    print("Done.")
    print("Users:", n_users, "Items:", n_items, "NNZ:", R.nnz)
    print("Saved:", out_interactions, args.out_dir / "processed_train.npz", args.out_dir / "user2index.pkl", args.out_dir / "item2index.pkl")

if __name__ == "__main__":
    main()
