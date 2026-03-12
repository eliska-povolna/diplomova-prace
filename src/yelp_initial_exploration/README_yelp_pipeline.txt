
Quick usage:

1) Convert Yelp JSONL -> Parquet (via notebook or your own step).
   Result structure:
     yelp_parquet/
       business/state=XX/part-*.parquet
       review/year=YYYY/part-*.parquet
       user.parquet

2) Build CSR + ID maps (positive-only implicit by default, stars>=4):
   python yelp_build_csr.py --parquet_dir yelp_parquet --out_dir data_yelp --pos_threshold 4.0

   Outputs in data_yelp/:
     - processed_train.npz       (CSR user x item, implicit)
     - interactions_filtered.parquet
     - user2index.pkl
     - item2index.pkl

3) Train ELSA/EASE to get item factors and/or per-user scores as usual (adapt your train_elsa.py to read processed_train.npz).

4) Map neurons/tags (needs business metadata and neuron->item weights):
   python map_neurons_to_yelp_tags.py      --parquet_dir yelp_parquet      --item2index data_yelp/item2index.pkl      --neuron_item_matrix outputs/neuron_item_matrix.npy      --out_json outputs/neuron_tag_map.json      --out_item_tags_csv outputs/items_with_tags.csv

5) Query boost with tag TF-IDF:
   python query_boost_elsa_yelp.py      --items_csv outputs/items_with_tags.csv      --scores outputs/user123_scores.npy      --query "family friendly pizza with outdoor seating"      --alpha 0.7      --topk 50      --out_csv outputs/ranked.csv
