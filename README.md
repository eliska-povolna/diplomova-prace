# Interpretable POI Recommender System with Sparse Autoencoders

> **Diploma thesis** — Eliška Povolná, 2025/2026

## What is this about?

Modern recommender systems achieve high accuracy through complex latent representations, but these are often opaque and hard for users to influence. This thesis integrates **Sparse Autoencoders (SAE)** into a collaborative-filtering architecture (**ELSA**) to transform dense latent vectors into interpretable, user-controllable preference representations — applied to **Points of Interest (POI)** recommendation using **Yelp** data.

The resulting model learns sparse, semantically meaningful features that can serve as interactive "knobs" — users can directly steer recommendations by adjusting their preference profile, rather than treating the system as a black box.

---

## Repository structure

```
.
├── thesis/                  # LaTeX source of the diploma thesis (MFF UK template)
│   ├── thesis.tex           # Master document
│   ├── metadata.tex         # Thesis title, author, supervisor, keywords
│   ├── macros.tex           # Custom LaTeX macros
│   ├── literatura.bib       # BibTeX references (BibLaTeX / biber)
│   ├── uvod.tex             # Introduction chapter
│   ├── kap01.tex            # Chapter 1 — Theoretical foundations
│   ├── kap02.tex            # Chapter 2 — Related work
│   ├── kap03.tex            # Chapter 3 — System design
│   ├── kap04.tex            # Chapter 4 — Implementation
│   ├── kap05.tex            # Chapter 5 — Evaluation
│   ├── zaver.tex            # Conclusion
│   ├── img/                 # Figures & diagrams
│   ├── tex/                 # Additional TeX packages (e.g. pdfx)
│   ├── .latexmkrc           # latexmk configuration (LuaLaTeX + biber)
│   └── Makefile             # Build shortcut: `make` compiles thesis.pdf
│
├── src/                     # Python implementation
│   ├── models/              # Recommender model code
│   │   ├── collaborative_filtering.py   # ELSA (linear shallow autoencoder)
│   │   ├── sparse_autoencoder.py        # TopK SAE architecture
│   │   └── sae_cf_model.py              # Combined ELSA + TopK SAE model
│   ├── data/                # Data loading & preprocessing
│   │   ├── yelp_loader.py   # Load Yelp Parquet data via DuckDB
│   │   └── preprocessing.py # Build CSR matrix and ID maps
│   ├── yelp_initial_exploration/  # Initial Yelp data exploration scripts
│   │   ├── train_elsa.py    # ELSA training script (reference implementation)
│   │   ├── train_sae.py     # TopK SAE training on ELSA latents
│   │   ├── yelp_build_csr.py # Build CSR matrix from Parquet via DuckDB (CLI)
│   │   ├── map_neurons_to_yelp_tags.py  # Map SAE neurons to Yelp tags
│   │   ├── query_boost_elsa_yelp.py     # Query-boosted recommendation
│   │   └── requirements.txt
│   ├── ui/                  # Interactive UI for testing the models (TBD)
│   ├── train.py             # Training entry point (stub)
│   ├── evaluate.py          # Evaluation entry point (stub)
│   └── requirements.txt     # Python dependencies
│
├── notebooks/               # Jupyter notebooks — data exploration
│   └── 01_data_exploration.ipynb   # DuckDB-based Yelp exploration starter
│
├── configs/
│   └── default.yaml         # Training hyperparameters (ELSA + TopK SAE)
│
├── data/                    # Raw & processed data (git-ignored, see data/README.md)
│
└── .github/workflows/       # CI/CD pipelines
    ├── ci.yml               # Lint, type-check & test Python code
    └── thesis.yml           # Compile LaTeX thesis to PDF
```

---

## Quick start

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt
```

### Download & convert Yelp data

The Yelp Open Dataset must be downloaded manually from <https://www.yelp.com/dataset>.
See [`data/README.md`](data/README.md) for conversion instructions (JSON → Parquet).

### Build the CSR matrix (initial exploration)

```bash
# Matches src/yelp_initial_exploration/yelp_build_csr.py
python src/yelp_initial_exploration/yelp_build_csr.py \
    --parquet_dir yelp_parquet --out_dir data/processed
```

### Train ELSA + TopK SAE

```bash
# Reference training scripts in yelp_initial_exploration/:
python src/yelp_initial_exploration/train_elsa.py
python src/yelp_initial_exploration/train_sae.py
```

### Run the UI

```bash
# Instructions will be added once the frontend technology is decided.
# See src/ui/README.md
```

### Compile the thesis

```bash
cd thesis
make           # uses latexmk + biber via .latexmkrc
# or directly:
latexmk thesis.tex
```

---

## Key references

- Spišák M., Bartyzal R., Hoskovec A., Peška L. (2024). *On Interpretability of Linear Autoencoders.* RecSys '24, ACM.
- Wang X. et al. (2021). *From Knots to Knobs: Interpretable Collaborative Filtering via Disentangled Representations.* WWW '21.
- Ooge J., Dereu L., Verbert K. (2023). *Steering Recommendations and Visualising Its Impact.* IUI '23, ACM.
- Zeng J. et al. (2025). *Explainable next POI recommendation based on spatial–temporal disentanglement.* Knowledge-Based Systems.

---

## License

See [LICENSE](LICENSE).
