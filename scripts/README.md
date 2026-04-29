# Development Scripts

This folder contains utility scripts for **development and experimentation** purposes. These scripts are **not part of the main pipeline** and are used for:

- Baseline model training (ALS, EASE)
- Thesis visualization generation
- Photo indexing and cloud upload
- Grid search analysis
- Data preprocessing experiments
- Steering evaluation analysis

## Quick Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `train_baseline.py` | Train ALS/EASE baseline models with grid search | Development |
| `test_quick_baseline.py` | Quick sanity test of baseline pipeline on small data | Testing |
| `analyze_grid_search.py` | Analyze and plot grid search results | Development |
| `generate_thesis_charts.py` | Generate dataset/metrics figures for thesis | Thesis |
| `plot_steering_eval.py` | Generate steering evaluation charts | Evaluation |
| `generate_universal_mappings.py` | Create universal item/business ID mappings | Development |
| `precompute_photo_index.py` | Build and upload photo index to cloud | Deployment |
| `precompute_user_matrices.py` | Precompute CSR matrices for all users | Optimization |
| `upload_photos_to_gcs_resume.py` | Resume photo upload with checkpoint support | Deployment |
| `upload_to_cloud.py` | Upload experiment runs to GCS | Deployment |
| `reset_and_reload_data_resume.py` | Resume Cloud SQL database reload | Deployment |
| `baseline_gridsearch.yaml` | Configuration for baseline grid search | Config |

## Notes

- **None of these scripts are required** for the main pipeline (preprocess → train → label → evaluate)
- They are independent tools for development, testing, and cloud deployment
- Most require local data setup and environment configuration
- Some scripts (photo upload, cloud upload) require GCS credentials

## Integration with Main Pipeline

The main pipeline is defined in `src/`:
- `src/preprocess_data.py` - Preprocessing stage
- `src/train.py` - Training stage
- `src/label.py` - Labeling stage
- `src/evaluate.py` - Evaluation stage

The interactive UI is in `src/ui/main.py`.

For main pipeline documentation, see [../README.md](../README.md).
