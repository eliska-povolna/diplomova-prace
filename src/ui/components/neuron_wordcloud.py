"""Word cloud visualization component for neurons."""

import logging
from pathlib import Path
from typing import Optional, Dict

import streamlit as st
from wordcloud import WordCloud
from PIL import Image

logger = logging.getLogger(__name__)


def get_wordcloud_text(
    neuron_idx: int,
    precomputed_wordcloud_dir: Optional[Path] = None,
) -> str:
    """Get word cloud text for a neuron.
    
    First tries precomputed cache, otherwise returns placeholder.
    
    Args:
        neuron_idx: Neuron index
        precomputed_wordcloud_dir: Path to precomputed word clouds (optional)
        
    Returns:
        Text suitable for WordCloud generation
    """
    # Try to load precomputed word cloud
    if precomputed_wordcloud_dir:
        wordcloud_file = precomputed_wordcloud_dir / f"neuron_{neuron_idx}.json"
        if wordcloud_file.exists():
            import json
            try:
                with open(wordcloud_file) as f:
                    data = json.load(f)
                return data.get("text", "")
            except Exception as e:
                logger.warning(f"Failed to load precomputed cloud for neuron {neuron_idx}: {e}")
    
    # Fallback: placeholder text
    return f"feature neuron {neuron_idx} representation embedding activation"


def render_wordcloud(
    text: str,
    width: int = 600,
    height: int = 300,
    colormap: str = "viridis",
) -> Image.Image:
    """Render text as a word cloud PIL Image.
    
    Args:
        text: Text for word cloud (space-separated words)
        width: Image width in pixels
        height: Image height in pixels
        colormap: Matplotlib colormap name
        
    Returns:
        PIL Image
    """
    if not text or len(text.strip()) == 0:
        # Return blank image if no text
        img = Image.new("RGB", (width, height), color="white")
        return img
    
    try:
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            colormap=colormap,
            relative_scaling=0.5,
            min_font_size=10,
        ).generate(text)
        
        # Convert to PIL Image
        img_array = wc.to_array()
        img = Image.fromarray(img_array)
        return img
    except Exception as e:
        logger.error(f"Failed to generate word cloud: {e}")
        # Return blank image on error
        return Image.new("RGB", (width, height), color="white")


@st.cache_resource
def cached_wordcloud_for_neuron(
    neuron_idx: int,
    text: str,
    width: int = 600,
    height: int = 300,
    colormap: str = "viridis",
) -> Image.Image:
    """Cached word cloud rendering (called by display_neuron_wordcloud).
    
    Streamlit caches the Image object to avoid re-rendering.
    """
    return render_wordcloud(text, width, height, colormap)


def display_neuron_wordcloud(
    neuron_idx: int,
    label: str = "",
    precomputed_wordcloud_dir: Optional[Path] = None,
    width: int = 600,
    height: int = 300,
    colormap: str = "viridis",
    show_info: bool = True,
) -> None:
    """Display a word cloud for a single neuron.
    
    Args:
        neuron_idx: Neuron index
        label: Optional label for the neuron (e.g., "Italian Restaurant")
        precomputed_wordcloud_dir: Path to precomputed word clouds
        width: Image width
        height: Image height
        colormap: Matplotlib colormap
        show_info: Show info text above cloud
    """
    if show_info:
        if label:
            st.markdown(f"**Neuron {neuron_idx}: {label}**")
        else:
            st.markdown(f"**Neuron {neuron_idx}**")
    
    # Get text for this neuron
    text = get_wordcloud_text(neuron_idx, precomputed_wordcloud_dir)
    
    # Render and cache
    img = cached_wordcloud_for_neuron(
        neuron_idx,
        text,
        width,
        height,
        colormap,
    )
    
    # Display
    st.image(img, width=width, use_column_width=False)


def display_neuron_wordcloud_grid(
    neuron_indices: list,
    labels: Optional[Dict[int, str]] = None,
    precomputed_wordcloud_dir: Optional[Path] = None,
    cols: int = 2,
    width: int = 400,
    height: int = 200,
) -> None:
    """Display multiple word clouds in a grid.
    
    Args:
        neuron_indices: List of neuron indices to display
        labels: Optional dict mapping neuron_idx to label
        precomputed_wordcloud_dir: Path to precomputed word clouds
        cols: Number of columns in grid
        width: Image width for each cloud
        height: Image height for each cloud
    """
    labels = labels or {}
    
    for i in range(0, len(neuron_indices), cols):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            idx = i + col_idx
            if idx >= len(neuron_indices):
                break
            
            neuron_idx = neuron_indices[idx]
            label = labels.get(neuron_idx, "")
            
            with col:
                display_neuron_wordcloud(
                    neuron_idx,
                    label,
                    precomputed_wordcloud_dir,
                    width,
                    height,
                    show_info=True,
                )
