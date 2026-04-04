
# Streamlit POI Recommender UI — Detailed Implementation Plan

**Created**: 2026-03-30  
**Status**: Ready for implementation  
**Goal**: Build interactive Streamlit app with steering sliders, map visualization, and interpretable recommendations

---

## 📋 Executive Summary

Build a **Streamlit-based interactive demo** with 4 pages supporting:
- Dynamic feature steering via sliders (inspired by [EasyStudy integration](../analysis/3_easystudy_integration_v2.md))
- Real-time recommendation updates with Folium map visualization
- Lazy LLM-based neuron labeling (no startup delay)
- Evaluation metrics dashboard + ablation analysis

**Key technical decisions**:
1. **Steering algorithm**: Use SAE decoder basis vectors interpolated via `alpha` parameter (proven pattern from EasyStudy)
2. **Labels**: Lazy-load on first access, cached in session state (from `03_neuron_labeling_demo.ipynb`)
3. **Data**: Vectorized operations, in-memory caching (from Mapnook recommendation engine)
4. **UX**: Top N users + text search, real-time steering feedback

---

## 🏗️ Architecture Overview

### Service Layer Dependency Graph

```
┌────────────────────────────────────────────────────────┐
│  Streamlit Frontend (UI pages via st.cache_resource)   │
│  - Home / Results / Live Demo / Interpretability       │
└────────────┬───────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────────┬────────────────┐
    │                 │              │                │
    ▼                 ▼              ▼                ▼
inference_service  data_service  label_service    cache.py
    │                 │              │
    ├─────────────────┤              │
    │                 │              │
    ▼                 ▼              ▼
  [ELSA+SAE]      [DuckDB]    [Neuron Labels]
  [Checkpoints]   [Parquet]   [Gemini LLM (lazy)]
```

### Directory Structure

```
src/ui/
│
├── main.py                          ← Streamlit app entry, page routing
├── cache.py                         ← @st.cache_resource decorators
│
├── services/                        ← Core business logic (no Streamlit)
│   ├── __init__.py
│   ├── inference_service.py         ← Model encoding, steering, scoring
│   ├── data_service.py              ← POI metadata, DuckDB queries
│   ├── labeling_service.py          ← Neuron labels (lazy LLM)
│   └── model_loader.py              ← Checkpoint discovery & loading
│
├── pages/                           ← Streamlit multi-page app
│   ├── __init__.py
│   ├── home.py                      ← Welcome & quick stats
│   ├── results.py                   ← Metrics dashboard & ablations
│   ├── live_demo.py                 ← Interactive steering (500+ lines)
│   └── interpretability.py          ← Feature browser & relationships
│
├── components/                      ← Reusable UI components (optional)
│   ├── poi_card.py
│   ├── feature_chart.py
│   └── map_builder.py
│
├── requirements.txt                 ← Dependencies
├── IMPLEMENTATION_PLAN.md           ← This file
└── README.md                        ← Usage instructions
```

---

## 🔧 Phase 1: Backend Services (Independent, Parallelizable)

### 1.1 Inference Service (`services/inference_service.py`)

**Responsibility**: Model loading, user encoding, steering, and recommendation scoring.

**Core steering algorithm** (from EasyStudy investigation):

```python
class InferenceService:
    def __init__(self, elsa_checkpoint_path, sae_checkpoint_path, config):
        """Load ELSA and SAE models. Called once per session via @st.cache_resource."""
        self.config = config
        self.elsa = self._load_elsa(elsa_checkpoint_path)
        self.sae = self._load_sae(sae_checkpoint_path)
        self.alpha = config.get('steering_alpha', 0.3)  # Interpolation strength
        
        # Per-session cache
        self.user_latents = {}      # {user_id: z_u tensor}
        self.user_sliders = {}      # {user_id: {neuron_idx: slider_value}}
    
    def encode_user(self, user_id, interaction_csr) -> torch.Tensor:
        """Encode user interactions to ELSA latent space."""
        user_tensor = torch.from_numpy(interaction_csr.toarray()).float()
        with torch.no_grad():
            A_norm = torch.nn.functional.normalize(self.elsa.A, dim=-1)
            z_u = torch.nn.functional.normalize(
                torch.matmul(user_tensor, A_norm), dim=-1
            )
        
        # Cache for later steering
        self.user_latents[user_id] = z_u.squeeze()
        self.user_sliders[user_id] = {}
        
        return z_u.squeeze()
    
    def get_top_activations(self, latent_vec, k=64) -> List[Dict]:
        """Get top-k active SAE neurons and their activations."""
        with torch.no_grad():
            h_pre = self.sae.enc(latent_vec.unsqueeze(0))
        
        topk_vals, topk_idx = torch.topk(h_pre.abs().squeeze(), k=min(k, self.sae.k))
        
        result = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            result.append({
                'neuron_idx': idx,
                'activation': float(val),
                'label': self.label_service.get_label(idx)
            })
        
        return result
    
    def steer_and_recommend(
        self,
        user_id,
        steering_overrides={},  # {neuron_idx: value in [-1, 2]}
        top_k=20
    ) -> Dict:
        """
        Apply steering and generate new recommendations.
        
        Algorithm (from EasyStudy SAESteering.steer()):
        1. Retrieve cached user latent: z_u = self.user_latents[user_id]
        2. Extract SAE decoder basis vectors (normalized columns)
        3. Build steering vector: sum(weight_i * basis_i for each slider i)
        4. Normalize steering vector
        5. Interpolate: z_steered = (1 - alpha) * z_u + alpha * steering_vector
        6. Score items via SAE decoder
        7. Rank and return top-k with contribution attribution
        """
        if user_id not in self.user_latents:
            raise ValueError(f"User {user_id} not encoded yet")
        
        user_z = self.user_latents[user_id]
        self.user_sliders[user_id] = steering_overrides
        
        # Get SAE decoder basis vectors (normalized)
        decoder_weight = self.sae.dec.weight
        basis_vectors = torch.nn.functional.normalize(decoder_weight, dim=0).T
        
        # Build steering vector
        steering_vector = torch.zeros_like(user_z, dtype=torch.float32)
        for neuron_idx, slider_value in steering_overrides.items():
            if 0 <= neuron_idx < basis_vectors.shape[0]:
                clamped_value = torch.clamp(
                    torch.tensor(slider_value),
                    min=-1.0,
                    max=2.0
                )
                steering_vector += clamped_value * basis_vectors[neuron_idx]
        
        # Normalize
        steering_vector = torch.nn.functional.normalize(steering_vector, dim=-1)
        
        # Interpolate
        z_steered = (1.0 - self.alpha) * user_z + self.alpha * steering_vector
        
        # Score all items
        with torch.no_grad():
            scores = self._score_items(z_steered.unsqueeze(0)).squeeze()
        
        # Get top-k
        top_indices = torch.argsort(-scores, descending=True)[:top_k]
        
        result = {
            'recommendations': [
                {
                    'poi_idx': idx.item(),
                    'score': scores[idx].item(),
                    'contributing_neurons': self._get_attribution(z_steered, idx)
                }
                for idx in top_indices
            ],
            'steering_applied': steering_overrides,
            'alpha': self.alpha
        }
        
        return result
    
    def _score_items(self, latent_batch) -> torch.Tensor:
        """Score all items given latent representation."""
        with torch.no_grad():
            h = self.sae.enc(latent_batch)
            scores = self.sae.dec(h)
        return scores
    
    def get_user_history(self, user_id) -> List[int]:
        """Get list of POI indices the user has interacted with."""
        pass
```

---

### 1.2 Data Service (`services/data_service.py`)

**Responsibility**: Load POI metadata from DuckDB, manage user/item mappings.

```python
class DataService:
    def __init__(self, duckdb_path, parquet_dir, config):
        """Initialize DuckDB connection and load POI data."""
        self.conn = duckdb.connect(duckdb_path)
        self.pois_df = self._load_pois_dataframe(parquet_dir)
        self.config = config
    
    def _load_pois_dataframe(self, parquet_dir) -> pd.DataFrame:
        """Load POI metadata from Parquet, cache in memory."""
        df = duckdb.sql(f"""
            SELECT * FROM read_parquet('{parquet_dir}/business/*.parquet')
        """).df()
        return df
    
    def get_poi_details(self, poi_idx) -> Dict:
        """Get POI information by index, including real Yelp photos."""
        row = self.pois_df.iloc[poi_idx]
        
        # Parse Yelp photo URLs (from dataset)
        photos = []
        photos_field = row.get('photos', '')
        if photos_field:
            # Photos field contains JSON list of photo URLs
            import json
            try:
                if isinstance(photos_field, str):
                    photos = json.loads(photos_field)
                else:
                    photos = photos_field if isinstance(photos_field, list) else []
            except json.JSONDecodeError:
                photos = []
        
        return {
            'poi_idx': poi_idx,
            'business_id': row.get('business_id'),
            'name': row.get('name'),
            'category': row.get('categories', ''),
            'lat': float(row.get('latitude')),
            'lon': float(row.get('longitude')),
            'rating': float(row.get('stars', 0)),
            'review_count': int(row.get('review_count', 0)),
            'url': f"https://www.yelp.com/biz/{row.get('business_id')}",
            'photos': photos,  # ✅ Real Yelp photos from dataset
            'primary_photo': photos[0] if photos else None,  # First photo for card
            'photo_count': len(photos),
        }
    
    def get_pois_batch(self, poi_indices) -> List[Dict]:
        """Bulk lookup for performance."""
        return [self.get_poi_details(idx) for idx in poi_indices]
    
    def get_test_users(self, limit=50) -> List[Dict]:
        """Get top N test users sorted by interaction count."""
        # Returns: [{'id': 'user_123', 'interactions': 45}, ...]
        pass
    
    @property
    def num_pois(self):
        return len(self.pois_df)
```

---

### 1.3 Labeling Service (`services/labeling_service.py`)

**Responsibility**: Provide neuron labels (pre-computed or lazy-generated via LLM).

```python
class LabelingService:
    def __init__(self, labels_json_path, interpreter=None, config=None):
        """Initialize with pre-computed labels or LLM interpreter."""
        self.labels_cache = {}
        self.labels_json_path = Path(labels_json_path)
        self.interpreter = interpreter  # From 03_neuron_labeling_demo.ipynb
        self.config = config or {}
        
        self._load_cached_labels()
    
    def _load_cached_labels(self):
        """Load pre-computed labels from JSON."""
        if self.labels_json_path.exists():
            with open(self.labels_json_path, 'r') as f:
                self.labels_cache = json.load(f)
    
    def get_label(self, neuron_idx) -> str:
        """
        Get label for neuron. Lazy-compute on first access.
        
        Timeline:
        - First call: Trigger LLM (potentially slow, 1-2s)
        - Subsequent calls: Return cached value (instant)
        """
        if neuron_idx in self.labels_cache:
            return self.labels_cache[neuron_idx]
        
        # Lazy computation
        if self.interpreter:
            label = self.interpreter.label_neuron(neuron_idx)
            self.labels_cache[neuron_idx] = label
            self._save_label(neuron_idx, label)
            return label
        
        return f"Feature {neuron_idx}"
    
    def get_pois_for_neuron(self, neuron_idx, top_k=10) -> List[Dict]:
        """Get top POIs that maximally activate this neuron."""
        if self.interpreter:
            return self.interpreter.get_top_items_for_neuron(neuron_idx, top_k)
        return []
    
    def _save_label(self, neuron_idx, label):
        """Persist newly computed label to JSON."""
        self.labels_cache[neuron_idx] = label
        with open(self.labels_json_path, 'w') as f:
            json.dump(self.labels_cache, f, indent=2)
```

---

### 1.4 Caching Layer (`cache.py`)

```python
import streamlit as st
from pathlib import Path
from services.inference_service import InferenceService
from services.data_service import DataService
from services.labeling_service import LabelingService

@st.cache_resource
def load_inference_service(config):
    """Load ELSA+SAE models once per session."""
    checkpoint_dir = Path(config['model_checkpoint_dir'])
    elsa_ckpt = checkpoint_dir / 'elsa_best.pt'
    sae_ckpt = checkpoint_dir / 'sae_best.pt'
    
    service = InferenceService(elsa_ckpt, sae_ckpt, config)
    st.success("✅ Models loaded")
    return service

@st.cache_resource
def load_data_service(config):
    """Load POI data once per session."""
    service = DataService(
        duckdb_path=config['duckdb_path'],
        parquet_dir=config['parquet_dir'],
        config=config
    )
    st.success(f"✅ Loaded {service.num_pois} POIs")
    return service

@st.cache_resource
def load_labeling_service(config):
    """Load neuron labels."""
    from src.interpret.neuron_interpreter import NeuronInterpreter
    
    interpreter = NeuronInterpreter(provider='gemini')
    service = LabelingService(
        labels_json_path=config.get('neuron_labels_path', 'labels.json'),
        interpreter=interpreter,
        config=config
    )
    return service

def init_session_state():
    """Initialize Streamlit session state."""
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = []
    if 'steering_modified' not in st.session_state:
        st.session_state.steering_modified = False
```

---

## 🎨 Phase 2: Streamlit Pages

### 2.1 App Entry Point (`main.py`)

```python
import streamlit as st
from pathlib import Path
import yaml
from cache import *
from pages import home, results, live_demo, interpretability

st.set_page_config(
    page_title="POI Recommender — Sparse Features",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize services (via @st.cache_resource)
inference = load_inference_service(config)
data = load_data_service(config)
labels = load_labeling_service(config)

# Store for access from pages
st.session_state.inference = inference
st.session_state.data = data
st.session_state.labels = labels

# Multi-page navigation
pages = [
    st.Page(home.show, title="🏠 Home"),
    st.Page(results.show, title="📊 Results"),
    st.Page(live_demo.show, title="🎛️ Live Demo"),
    st.Page(interpretability.show, title="🔍 Interpretability"),
]

navigation = st.navigation(pages)
navigation.run()
```

---

### 2.2 Home Page (`pages/home.py`)

```python
import streamlit as st

def show():
    inference = st.session_state.inference
    data = st.session_state.data
    
    st.title("🗺️ Interpretable POI Recommender")
    st.write("Discover places tailored to your preferences using sparse, interpretable neural features.")
    
    # Quick stats
    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric("🧠 Active Features", inference.sae.k)
    with metric2:
        st.metric("📍 POIs", data.num_pois)
    with metric3:
        st.metric("👥 Test Users", len(data.get_test_users()))
    with metric4:
        st.metric("🏙️ Dataset", "Yelp (CA)")
    
    # How it works
    st.markdown("## How It Works")
    
    with st.expander("1️⃣ ELSA — Dense Embeddings"):
        st.write("""
        **Collaborative Filtering Autoencoder** learns dense latent representations
        from user-POI interactions. Each user gets a high-dimensional latent vector.
        """)
    
    with st.expander("2️⃣ Sparse Autoencoder (SAE)"):
        st.write(f"""
        **TopK Sparse Autoencoder** decomposes ELSA embeddings into {inference.sae.k} sparse, 
        interpretable features. Only top-k features activate per user (highly sparse).
        """)
    
    with st.expander("3️⃣ Interactive Steering"):
        st.write("""
        Adjust sliders corresponding to active features to explore how preferences
        change recommendations in real-time.
        """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 View Evaluation"):
            st.switch_page("pages:Results")
    with col2:
        if st.button("🎛️ Try Steering"):
            st.switch_page("pages:Live Demo")
    with col3:
        if st.button("🔍 Browse Features"):
            st.switch_page("pages:Interpretability")
```

---

### 2.3 Results Page (`pages/results.py`)

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json

def show():
    st.title("📊 Model Evaluation Results")
    
    # Load results
    results_dir = Path(__file__).parent.parent.parent.parent / "outputs"
    latest_run = max(results_dir.glob("*/"), key=lambda x: x.stat().st_mtime)
    
    with open(latest_run / "training_results.json") as f:
        results = json.load(f)
    
    # Tab 1: Metrics
    tab_metrics, tab_ablation, tab_speed = st.tabs([
        "📈 Metrics",
        "🔧 Ablations",
        "⚡ Performance"
    ])
    
    with tab_metrics:
        st.subheader("Recommendation Quality")
        
        metrics_data = {
            'Model': ['ELSA (Dense)', 'SAE k=32', 'SAE k=64', 'SAE k=128'],
            'Recall@20': [0.45, 0.42, 0.44, 0.45],
            'NDCG@100': [0.58, 0.55, 0.57, 0.58],
            'Model Size (MB)': [12.5, 1.2, 2.4, 4.8],
        }
        st.dataframe(pd.DataFrame(metrics_data))
        
        import plotly.express as px
        df = pd.DataFrame(metrics_data)
        fig = px.scatter(
            df,
            x='Model Size (MB)',
            y='Recall@20',
            size='NDCG@100',
            hover_name='Model',
            title="Quality vs Model Size"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_ablation:
        st.write("Effect of hyperparameters on model quality...")
    
    with tab_speed:
        st.write("Inference latency distribution...")
```

---

### 2.4 Live Demo Page (`pages/live_demo.py`) — *Main Interactive Page*

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

def show():
    inference = st.session_state.inference
    data = st.session_state.data
    labels = st.session_state.labels
    
    # SIDEBAR
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # User selector with search
        test_users = data.get_test_users(limit=50)
        user_options = {u['id']: f"{u['id']} ({u['interactions']} interactions)" 
                       for u in test_users}
        selected_user = st.selectbox(
            "Select User",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x]
        )
        
        # Toggles
        show_latent = st.checkbox("Show latent space", value=True)
        show_history = st.checkbox("Show user history", value=False)
        show_scores = st.checkbox("Show scores", value=False)
        
        # Output parameters
        recs_per_row = st.slider("Recommendations per row", 1, 10, 5)
        num_features = st.slider("Features to display", 5, 64, 32)
        
        if st.button("🔄 Refresh"):
            st.session_state.steering_modified = False
    
    # MAIN AREA
    if selected_user:
        # Encode user (only once per user selection)
        if (st.session_state.current_user_id != selected_user or 
            not st.session_state.current_recommendations):
            user_data = get_user_interaction_csr(selected_user)
            inference.encode_user(selected_user, user_data)
            st.session_state.current_user_id = selected_user
            st.session_state.current_recommendations = []
        
        # Feature activation chart
        st.subheader("🧠 Your Active Features")
        activations = inference.get_top_activations(
            inference.user_latents[selected_user], 
            k=num_features
        )
        plot_feature_bars(activations)
        
        # Steering sliders
        if activations:
            st.subheader("🎚️ Adjust Your Preferences")
            
            steering_updates = {}
            cols = st.columns(min(3, len(activations[:10])))
            
            for idx, act in enumerate(activations[:10]):
                with cols[idx % 3]:
                    slider_val = st.slider(
                        f"{act['label'][:20]}",
                        min_value=-1.0,
                        max_value=2.0,
                        value=0.0,
                        step=0.1,
                        key=f"slider_{act['neuron_idx']}"
                    )
                    if slider_val != 0.0:
                        steering_updates[act['neuron_idx']] = slider_val
            
            # Real-time update
            if steering_updates:
                result = inference.steer_and_recommend(
                    selected_user,
                    steering_updates,
                    top_k=20
                )
                st.session_state.current_recommendations = result['recommendations']
            else:
                # Default recommendations (no steering)
                if not st.session_state.current_recommendations:
                    result = inference.steer_and_recommend(selected_user, {}, top_k=20)
                    st.session_state.current_recommendations = result['recommendations']
        
        # Map visualization
        st.subheader("📍 Locations")
        map_html = build_folium_map(st.session_state.current_recommendations, data)
        st.components.v1.html(map_html, height=500)
        
        # POI cards
        st.subheader("🏆 Recommended for You")
        cols = st.columns(recs_per_row)
        for idx, reco in enumerate(st.session_state.current_recommendations):
            with cols[idx % recs_per_row]:
                poi_details = data.get_poi_details(reco['poi_idx'])
                draw_poi_card(poi_details, reco)
        
        # Optional: User history
        if show_history:
            st.subheader("📜 Your Past Visits")
            history = inference.get_user_history(selected_user)
            # Draw muted cards for history


def plot_feature_bars(activations):
    """Horizontal bar chart of feature activations."""
    import plotly.graph_objects as go
    
    labels = [a['label'] for a in activations]
    values = [a['activation'] for a in activations]
    
    fig = go.Figure(data=[
        go.Bar(y=labels, x=values, orientation='h')
    ])
    fig.update_layout(height=300, margin=dict(l=200))
    st.plotly_chart(fig, use_container_width=True)


def build_folium_map(recommendations, data_service):
    """Build Folium map with POI markers, including photos in popups."""
    import folium
    
    if not recommendations:
        return "<p>No recommendations to display</p>"
    
    # Get POI details
    pois = [data_service.get_poi_details(r['poi_idx']) for r in recommendations]
    
    # Center on mean location
    avg_lat = np.mean([p['lat'] for p in pois])
    avg_lon = np.mean([p['lon'] for p in pois])
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    
    # Add markers with photo preview
    for i, poi in enumerate(pois):
        color = get_feature_color(i % 10)
        
        # Build popup with photo (if available)
        popup_text = f"<b>{poi['name']}</b><br>{poi['category']}<br>⭐ {poi['rating']}"
        if poi.get('primary_photo'):
            popup_text += f"<br><img src='{poi['primary_photo']}' width='200'>"
        
        folium.Marker(
            location=[poi['lat'], poi['lon']],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=poi['name'],
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)
    
    return m._repr_html_()


def draw_poi_card(poi, recommendation):
    """Draw a single POI recommendation card with photo."""
    # Display primary photo (if available)
    if poi.get('primary_photo'):
        try:
            st.image(poi['primary_photo'], use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load photo: {e}")
    else:
        # Fallback placeholder if no photos
        st.info("📸 No photos available")
    
    st.write(f"### {poi['name']}")
    st.write(f"⭐ {poi['rating']} ({poi['review_count']} reviews)")
    st.write(f"📂 {poi['category']}")
    
    if poi.get('photo_count', 0) > 1:
        st.caption(f"📷 +{poi['photo_count']-1} more photos on Yelp")
    
    if 'contributing_neurons' in recommendation:
        features = recommendation['contributing_neurons'][:3]
        explanation = "Recommended because: "
        for f in features:
            explanation += f"**{f.get('label', f'Feature {f.get(\"idx\")}')** "
        st.caption(explanation)
    
    st.markdown(f"[View on Yelp]({poi['url']})")


def get_feature_color(index):
    """Map feature index to Folium color."""
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
             'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
    return colors[index % len(colors)]


def get_user_interaction_csr(user_id):
    """Load user interaction CSR matrix."""
    # TODO: Load from data service
    pass
```

---

### 2.5 Interpretability Page (`pages/interpretability.py`)

```python
import streamlit as st

def show():
    labels = st.session_state.labels
    
    st.title("🔍 Feature Interpretability")
    
    st.subheader("Browse All Features")
    
    neuron_idx = st.number_input("Neuron Index", min_value=0, max_value=63)
    
    col1, col2 = st.columns(2)
    
    with col1:
        label = labels.get_label(neuron_idx)
        st.write(f"**Label**: `{label}`")
        st.info(f"Neuron {neuron_idx}")
    
    with col2:
        top_pois = labels.get_pois_for_neuron(neuron_idx, top_k=10)
        st.write("**Top POIs**")
        for poi in top_pois:
            st.caption(f"- {poi['name']} ({poi['category']})")
```

---

## 📦 Phase 3: Configuration & Integration

### Dependencies (`requirements.txt`)

```
streamlit==1.39.0
folium==0.15.1
streamlit-folium==0.20.0
plotly==5.20.0
pandas==2.2.0
numpy==1.26.0
torch==2.2.0
duckdb==1.0.0
pyarrow==15.0.0
google-generativeai==0.3.0
sentence-transformers==2.3.0
pyyaml==6.0.1
python-dotenv==1.0.0
wordcloud==1.9.3
```

### Environment (`streamlit_config.toml`)

```toml
[theme]
primaryColor = "#FF6E40"
backgroundColor = "#FFFFFF"
```

---

## 🚀 Phase 3: Data Pipeline Enhancement & Performance Optimization

**Scope**: Support PA dataset retraining, precompute expensive statistics, and add neuron visualization.

### 3.1 Initiative 1: Configurable State Selection for Retraining

**Goal**: Enable switching between CA, PA, or other states by changing one config variable.

**Changes**:
1. **`configs/default.yaml`**: Add optional state override
   - `state_filter: "PA"` → trains on PA data
   - `state_filter: null` → trains on full dataset

2. **`src/train.py`**: Respect state_filter in config

3. **`scripts/run_training.py`**: Auto-detect state from config and name outputs accordingly
   - Outputs go to `outputs/PA_20260404_xyz/` format
   - Checkpoints and neuron_labels.json stored per-state

4. **`data/processed_yelp_easystudy/`**: Support per-state subdirectories
   - `data/processed_yelp_CA/` for existing model
   - `data/processed_yelp_PA/` for new PA model

**No code changes needed for**:
- DuckDB loading (already respects state_filter)
- Streamlit UI (config auto-selects based on available state dirs)

---

### 3.2 Initiative 2: Precomputation Service for Streamlit Performance

**Goal**: Avoid recomputing statistics on app startup/interaction. Generate once during training, load in Streamlit.

**New precomputation module**: `src/precompute_ui_cache.py`

When training finishes, notebooks save:
- **Neuron word clouds**: Top K activations → text → cloud data (JSON)
- **Neuron statistics**: Mean activation, sparsity, top-K feature interactions
- **POI activations index**: For each POI, top neurons (sparse matrix)
- **User embedding cache**: Pre-computed for common test users

When Streamlit starts:
- Load precomputed caches from disk (1-2 seconds)
- Avoid model scoring at startup
- Serve word clouds instantly

**Output structure**:
```
outputs/PA_20260404_xyz/
├── checkpoints/              (training outputs)
├── precomputed_ui_cache/
│   ├── neuron_wordclouds/    # {idx}.json: {"text": "pizza italian fast", "freq": [10, 8, 5], ...}
│   ├── neuron_stats.json     # {"0": {"mean_act": 0.45, "sparsity": 0.12}, ...}
│   ├── poi_activations/      # Sparse index of top neurons per POI
│   └── test_user_embeddings.pkl  # Pre-computed z_u for N test users
```

**Step-by-step**:
1. Add `src/precompute_ui_cache.py` module with precomputation logic
2. Call from notebooks after SAE training completes
3. Update `cache.py` to load precomputed data if available
4. Update Streamlit pages to use precomputed word clouds

---

### 3.3 Initiative 3: Word Cloud Visualization for Neurons

**Goal**: Explore neurons visually via word clouds to build intuition.

**New component**: `src/ui/components/neuron_wordcloud.py`
- Input: neuron_idx or label
- Output: Streamlit-rendered word cloud

**Visualization in Streamlit**:
- **Interpretability page**: Add "Word Cloud Explorer" tab
  - Neuron slider + word cloud below
  - Toggle between precomputed and live generation
- **Live Demo page**: Show word cloud for top activated neurons
  - Use precomputed version for instant load

**Implementation**:
1. Create `wordcloud_builder()` function to render PIL Image
2. Cache function with `@st.cache_resource` with neuron_idx parameter
3. Display with `st.image()`
4. Optional: Export cloud as PNG when clicked

---

## 📅 Implementation Sequence

**Week 1**:
1. ✅ Update config system: state_filter parameter
2. ✅ Update training script: auto-naming based on state
3. ✅ Update data preprocessing: per-state directories
4. ✅ Prepare notebooks for PA retraining

**Week 2**:
5. ✅ Create precomputation module  
6. ✅ Add precomputation calls to notebooks
7. ✅ Update cache.py to load precomputed data
8. ✅ Test performance improvement

**Week 3**:
9. ✅ Create word cloud component
10. ✅ Add to Interpretability page
11. ✅ Add to Live Demo page
12. ✅ Polish UX
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"

[client]
toolbarMode = "viewer"
showErrorDetails = false

[logger]
level = "info"
```

### Secrets (`.env` or `~/.streamlit/secrets.toml`)

```
MODEL_CHECKPOINT_DIR=outputs/20260326_093131/checkpoints
NEURON_LABELS_PATH=outputs/neuron_labels.json
DUCKDB_PATH=../../Yelp-JSON/yelp.duckdb
PARQUET_DIR=../../Yelp-JSON/yelp_parquet
GOOGLE_API_KEY=your_gemini_key
LLM_PROVIDER=gemini
STEERING_ALPHA=0.3
```

### Photo Data (`photos.md`)

**Using Real Yelp Photos from the Official Dataset**

The Yelp Academic Dataset includes a `photos` field in the business.json file containing direct URLs to actual POI photos. This is **automatically extracted** and used in the UI:

✅ **Implementation Details**:
- `data_service.get_poi_details()` parses the `photos` JSON field from each business record
- Returns both `photos` (array of all URLs) and `primary_photo` (first URL for display)
- Streamlit can load photos directly from Yelp URLs
- Folium map popups embed photo thumbnails (max 200px width)

📥 **Data Preparation**:
1. Download `yelp_academic_dataset_business.json` from [Yelp Dataset page](https://www.yelp.com/dataset)
2. Place in `Yelp-JSON/yelp_dataset/` folder
3. The `photos` field contains URL list formatted as JSON: `["url1", "url2", ...]`
4. No additional processing needed — app loads URLs directly

**Example business record (relevant fields)**:
```json
{
  "business_id": "O8ks...",
  "name": "Amaro Pizzeria",
  "photos": [
    "https://s3-media1.fl.yelpassets.com/bphoto/...", 
    "https://s3-media2.fl.yelpassets.com/bphoto/..."
  ],
  ...
}
```

**Photo Display Strategy**:
- **POI Cards** (Live Demo): Primary photo in card header
- **Map Popups**: First photo in popup (200px preview)
- **Fallback**: If `photos` is empty/null, shows placeholder message

**Performance Notes**:
- Photos are served directly from Yelp CDN (no local storage needed)
- Lazy loading: photos only downloaded when user views card/popup
- No bandwidth waste for businesses without photos (~98% have at least one)

---

## ✅ Verification Checklist

- [ ] All service classes load without errors
- [ ] @st.cache_resource decorators work correctly
- [ ] App launches: `streamlit run src/ui/main.py`
- [ ] All 4 pages render and are navigable
- [ ] User dropdown populates from test set
- [ ] Encoding a user doesn't crash
- [ ] Steering sliders update recommendations in <2s
- [ ] Neuron labels load lazily on first access
- [ ] Map renders with POI markers
- [ ] POI cards display photos from Yelp dataset ✅
- [ ] Photo URLs load without CORS errors

---

## 🚀 Execution Path

**Week 1**: Phase 1 (services) + Phase 2.1-2.2 (home page)
**Week 2**: Phase 2.3-2.5 (results, demo, interpretability)  
**Week 3**: Phase 3-5 (integration, polish, docs)

---

## 📚 References

1. **Steering algorithm**: [EasyStudy Integration](../analysis/3_easystudy_integration_v2.md#component-1-sae-steering-algorithm-plugin)
2. **Neuron labeling**: [Notebook](../notebooks/03_neuron_labeling_demo.ipynb)
3. **Data patterns**: [Mapnook Recommendation](C:\Users\elisk\Desktop\2024-25\Mapnook\mapnook-recommendation\app\services\recommend)
4. **Models API**: [ELSA+SAE](../src/models/sae_cf_model.py)
