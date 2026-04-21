# Notebooks

Interactive Jupyter notebooks for data exploration, model training, neuron interpretation, and evaluation analysis.

## Pipeline Notebooks

Run in order after executing the command-line pipeline stages:

| Notebook | Prerequisites | Purpose |
|---|---|---|
| `01_data_exploration.ipynb` | DuckDB database initialized | Explore Yelp dataset: statistics, geographic distribution, interaction sparsity |
| `02_training.ipynb` | Data preprocessed (`python -m src.preprocess_data`) | Train ELSA + TopK SAE models with interactive monitoring |
| `03_neuron_labeling_demo.ipynb` | Models trained (`python -m src.label`) | Interpret and visualize learned neuron features interactively |
| `04_evaluation_analysis.ipynb` | Evaluation complete (`python -m src.evaluate`) | Compare ELSA vs SAE+ELSA metrics, visualize results |

## Setup

Before running notebooks, complete the pipeline stages in `../README.md`:

```bash
# 1. Setup database (one-time)
python -m src.setup_database --json-dir ~/Downloads/Yelp-JSON/yelp_dataset

# 2. Preprocess data
python -m src.preprocess_data --config configs/default.yaml

# 3. Train models
python -m src.train --config configs/default.yaml

# 4. Label neurons
python -m src.label

# 5. Evaluate
python -m src.evaluate --checkpoint outputs/YYYYMMDD_HHMMSS --split test
```

## Running

**Option A: VS Code (recommended)**
- Open any `.ipynb` file directly with VS Code's Jupyter extension

**Option B: Jupyter Lab/Notebook**
```bash
cd notebooks
jupyter lab
```

## Notes

- All notebooks import from `src/` (sys.path is set in first cell)
- Outputs are saved to `../outputs/` (project root), not `notebooks/outputs/`
- Run notebooks in sequence; each depends on previous pipeline stages
