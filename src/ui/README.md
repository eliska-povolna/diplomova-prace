# POI Recommender Streamlit UI

Interactive web application for exploring ELSA+SAE recommendations with real-time steering.

### 🚀 Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure the App

Edit `configs/default.yaml`:

```yaml
model:
  checkpoint_dir: outputs/20260326_093131/checkpoints

data:
  duckdb_path: ../../Yelp-JSON/yelp.duckdb
  parquet_dir: ../../Yelp-JSON/yelp_parquet
  
ui:
  steering_alpha: 0.3
```

#### 3. Download Yelp Dataset (if not already present)

Get `yelp_academic_dataset_business.json` from https://www.yelp.com/dataset and place in:
```
Yelp-JSON/yelp_dataset/
```

#### 4. Run the App

```bash
streamlit run src/ui/main.py
```

The app will start at `http://localhost:8501`

---

### 📄 Pages

#### 1. **🏠 Home**
- Welcome screen
- Quick dataset statistics
- How the system works (expandable sections)
- Navigation to other pages

#### 2. **📊 Results**
- Model evaluation metrics (Recall@20, NDCG@100)
- Ablation studies (effect of SAE sparsity)
- Inference latency distribution
- Quality vs model size trade-offs

#### 3. **🎛️ Live Demo** ⭐ Main Interactive Page
- User selection (top 50 by activity)
- Feature activation visualization
- **Steering sliders** for real-time preference adjustment
- Interactive map (Folium) with POI markers
- Recommendation cards with photos (from Yelp dataset)
- Optional user history display

#### 4. **🔍 Interpretability**
- Feature browser (0-63 neurons)
- LLM-generated labels for each neuron
- Top POIs activating each neuron
- Feature comparison tool
- Semantic relationships

---

### 🏗️ Architecture

```
src/ui/
├── main.py                  # App entry point + multi-page routing
├── cache.py                 # @st.cache_resource decorators
│
├── services/                # Backend (no Streamlit deps)
│   ├── inference_service.py # Steering algorithm + scoring
│   ├── data_service.py      # POI metadata + photos
│   ├── labeling_service.py  # Neuron labels (lazy LLM)
│   └── model_loader.py      # Checkpoint discovery
│
├── pages/                   # Streamlit pages
│   ├── home.py
│   ├── results.py
│   ├── live_demo.py         # Main interactive page
│   └── interpretability.py
│
└── components/              # Reusable UI components (future)
```

---

### 🎛️ Steering Algorithm

The core steering mechanism follows the **EasyStudy SAESteering** pattern:

```python
# 1. Get user latent from ELSA encoder
z_u = elsa.encode(user_interactions)

# 2. Extract SAE decoder basis vectors (normalized)
basis_vectors = F.normalize(sae.decoder.weight, dim=0).T

# 3. Build steering vector from slider overrides
v_steer = Σ(weight_i * basis_i)

# 4. Normalize
v_steer = F.normalize(v_steer)

# 5. Interpolate (alpha=0.3 by default)
z_steered = (1 - α) * z_u + α * v_steer

# 6. Score items via SAE decoder
scores = sae.decode(z_steered)
```

**Slider Range**: -1.0 to +2.0
- **-1.0**: Strongly avoid this feature
- **0.0**: No influence (default)
- **+2.0**: Strongly prefer this feature

---

### 📊 Data Integration

#### POI Metadata (from Yelp Dataset)

```json
{
  "business_id": "O8ks...",
  "name": "Amaro Pizzeria",
  "categories": "Italian, Restaurants",
  "latitude": 40.123,
  "longitude": -74.456,
  "stars": 4.2,
  "review_count": 150,
  "photos": [
    "https://s3-media1.fl.yelpassets.com/bphoto/...",
    "https://s3-media2.fl.yelpassets.com/bphoto/..."
  ]
}
```

**Photo Strategy**:
- Real Yelp URLs from `photos` field
- Served directly from Yelp CDN (no local storage)
- Lazy loading (only downloaded when viewed)
- ~98% of businesses have at least one photo

### Current Data & Cache Flow

- `src/ui/services/data_service.py` loads POI metadata lazily and falls back from Cloud SQL to local DuckDB.
- The service keeps model indices aligned with `item2index.pkl` from training, so recommendation and interpretation views refer to the same items.
- `src/precompute_ui_cache.py` can generate cached word clouds, neuron statistics, and test user embeddings under `outputs/<run_id>/precomputed_ui_cache/`.
- The UI reads the latest completed run’s outputs, not arbitrary intermediate files.

#### Performance Targets

- **Encoding latency**: ~0.15s per user
- **Scoring latency**: ~0.70s for 10K items  
- **Total recommendation**: <2s (interactive acceptable)

---

### 🔧 Configuration

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
MODEL_CHECKPOINT_DIR = "outputs/20260326_093131/checkpoints"
NEURON_LABELS_PATH = "outputs/neuron_labels.json"
```

---

### 🧪 Testing

**Manual Testing Checklist**:

- [ ] App launches without errors: `streamlit run main.py`
- [ ] All 4 pages render and are navigable
- [ ] Home page displays correct dataset stats
- [ ] Results page loads metrics or placeholder data
- [ ] User dropdown populates (top 50 users)
- [ ] Feature activation chart displays
- [ ] Steering sliders update without lag
- [ ] Map renders with POI markers
- [ ] POI cards display photos + details
- [ ] Interpretability page shows neuron labels
- [ ] Feature browser works (0-63 neuron range)
- [ ] Navigation buttons work across pages

---

### 🐛 Troubleshooting

**"Models not loading"**:
1. Check `model_checkpoint_dir` path exists
2. Verify `elsa_best.pt` + `sae_best.pt` in checkpoint folder
3. Confirm PyTorch version compatibility

**"POI data not loading"**:
1. Verify DuckDB path: `duckdb_path`
2. Check Parquet files exist in `parquet_dir`
3. Confirm Cloud SQL secrets are present if you expect the remote backend
4. Run: `duckdb yelp.duckdb "SELECT COUNT(*) FROM read_parquet('...')"`

**"Photos not displaying"**:
1. Confirm `yelp_academic_dataset_business.json` has `photos` field
2. Check network access to Yelp CDN
3. Try opening URLs manually in browser

**"Slow recommendations"**:
1. Reduce `num_recommendations` (slider max)
2. Check inference latency on "Results" tab
3. Increase `steering_alpha` for faster steering

---

### 📈 Future Enhancements

- [ ] Export recommendations as CSV
- [ ] User feedback loop (like/dislike)
- [ ] A/B testing framework
- [ ] Real-time metric updates
- [ ] Multi-user session management
- [ ] Feature relationship graph visualization
- [ ] Hypothesis testing interface

---

### 📚 References

- **IMPLEMENTATION_PLAN.md**: Full architecture + code examples
- **EasyStudy Integration**: steering algorithm source
- **Notebook**: `notebooks/03_neuron_labeling_demo.ipynb`
- **Model Code**: `src/models/sae_cf_model.py`
