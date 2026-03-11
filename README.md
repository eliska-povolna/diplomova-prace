# Interpretable POI Recommender System with Sparse Autoencoders

> **Diploma thesis** — Eliska Povolna, 2025/2026

## What is this about?

Modern recommender systems achieve high accuracy through complex latent representations, but these are often opaque and hard for users to influence. This thesis integrates **Sparse Autoencoders (SAE)** into a collaborative-filtering architecture to transform dense latent vectors into interpretable, user-controllable preference representations — applied to **Points of Interest (POI)** recommendation using **Yelp** data.

The resulting model learns sparse, semantically meaningful features that can serve as interactive "knobs" — users can directly steer recommendations by adjusting their preference profile, rather than treating the system as a black box.

---

## Repository structure

```
.
├── thesis/                  # LaTeX source of the diploma thesis
│   ├── main.tex             # Master document
│   ├── chapters/            # Individual chapter files
│   ├── figures/             # Figures & diagrams
│   └── bibliography.bib    # BibTeX references
│
├── src/                     # Python implementation
│   ├── models/              # Recommender model code
│   │   ├── collaborative_filtering.py   # Baseline MF / CF model
│   │   ├── sparse_autoencoder.py        # SAE architecture
│   │   └── sae_cf_model.py              # Combined SAE-CF model
│   ├── data/                # Data loading & preprocessing
│   │   ├── yelp_loader.py   # Load & parse Yelp JSON dataset
│   │   └── preprocessing.py # Feature engineering, splits
│   ├── ui/                  # Interactive UI for testing the models (TBD)
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # Evaluation entry point
│   └── requirements.txt     # Python dependencies
│
├── notebooks/               # Jupyter notebooks — data exploration & experiments
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

### Download Yelp data

The Yelp Open Dataset must be downloaded manually from <https://www.yelp.com/dataset> and placed inside `data/raw/`. See [`data/README.md`](data/README.md) for details.

### Train a model

```bash
python src/train.py --config configs/default.yaml
```

### Run the UI

```bash
# Instructions will be added once the frontend technology is decided.
# See src/ui/README.md
```

### Compile the thesis

```bash
cd thesis
latexmk -pdf main.tex
```

---

## References

1. (Lead supervisor's paper under review — citation will be added upon publication)
2. Zhang, S. et al. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys*.
3. Knijnenburg, B. et al. (2012). Explaining the user experience of recommender systems. *User Modeling and User-Adapted Interaction*.
4. Jannach, D. & Jugovac, M. (2019). Measuring the business value of recommender systems. *ACM Transactions on Management Information Systems*.
5. Liu, Y. et al. (2017). Experimental evaluation of point-of-interest recommendation in location-based social networks. *Proceedings of the VLDB Endowment*.

---

## License

See [LICENSE](LICENSE).
