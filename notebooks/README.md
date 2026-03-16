# Notebooks

Jupyter notebooks for interactive data exploration and experiment analysis.

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Basic Yelp dataset statistics, category & geographic distribution, interaction sparsity |

## Running

```bash
cd notebooks
jupyter lab
```

Or, with VS Code's Jupyter extension, open any `.ipynb` file directly.

Notebooks import from `src/`, so either:
- run from the repo root with `jupyter lab`, or
- add `sys.path.insert(0, '..')` at the top of the notebook (already done in the provided starters).
