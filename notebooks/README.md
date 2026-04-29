# Notebooks (Historical / Outdated)

These notebooks were used during exploration and prototyping and are now considered
historical artifacts. They may not reflect the current production pipeline, run-artifact
contracts, or local/cloud runtime behavior.

## Current Supported Workflow

Use the main project workflow from the root `README.md`:

1. CLI pipeline for preprocessing, training, labeling, and evaluation (`python -m src.*`).
2. Streamlit app for interactive analysis and inspection:
   - `Dataset Statistics` page for data exploration
   - `Results` page for evaluation metrics and comparisons
   - `Live Demo` and `Interpretability` for steering and feature analysis

## Notebook Status

- `01_data_exploration.ipynb`: replaced by Streamlit `Dataset Statistics`
- `02_training.ipynb`: historical training exploration
- `03_neuron_labeling_demo.ipynb`: historical interpretability exploration
- `04_evaluation_analysis.ipynb`: historical evaluation exploration

If you open these notebooks, treat outputs and code as exploratory references only.
