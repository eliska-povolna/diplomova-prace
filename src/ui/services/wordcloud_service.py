"""
Wordcloud generation service for neuron features.

Generates visual word frequency clouds from business categories
that maximally activate each neuron.
"""

from pathlib import Path
from typing import Dict, Optional, List
import json
import logging

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

logger = logging.getLogger(__name__)


class WordcloudService:
    """Generate wordclouds for neuron features from category data."""

    def __init__(
        self, 
        category_metadata_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
    ):
        """
        Initialize wordcloud service.

        Args:
            category_metadata_path: Path to neuron_category_metadata.json 
                                  (exported from notebook 03)
            labels_path: Path to neuron_labels.json (for label display)
        """
        if not HAS_WORDCLOUD:
            logger.warning("wordcloud not installed - wordcloud generation will be disabled")
        
        self.category_metadata = {}
        self.labels = {}
        
        # Load category metadata
        if category_metadata_path:
            self._load_category_metadata(Path(category_metadata_path))
        
        # Load labels
        if labels_path:
            self._load_labels(Path(labels_path))

    def _load_category_metadata(self, metadata_path: Path):
        """Load category metadata from JSON file."""
        if not metadata_path.exists():
            logger.warning(f"Category metadata not found: {metadata_path}")
            return
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.category_metadata = data
            logger.info(f"Loaded category metadata for {len(data)} neurons")
        except Exception as e:
            logger.error(f"Failed to load category metadata: {e}")

    def _load_labels(self, labels_path: Path):
        """Load neuron labels from JSON file."""
        if not labels_path.exists():
            logger.warning(f"Labels not found: {labels_path}")
            return
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both flat dict and nested "neuron_labels" structure
            if 'neuron_labels' in data:
                self.labels = data['neuron_labels']
            else:
                self.labels = data
            
            logger.info(f"Loaded labels for {len(self.labels)} neurons")
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")

    def get_neuron_label(self, neuron_id: int) -> str:
        """Get readable label for a neuron."""
        label = self.labels.get(str(neuron_id))
        if label:
            return label
        return f"Feature {neuron_id}"

    def get_categories_for_neuron(self, neuron_id: int) -> List[str]:
        """Get list of categories that activate this neuron."""
        metadata = self.category_metadata.get(str(neuron_id), {})
        return metadata.get('categories', [])

    def generate_wordcloud(
        self, 
        neuron_id: int,
        width: int = 400,
        height: int = 300,
        background_color: str = 'white',
        colormap: str = 'viridis'
    ) -> Optional:
        """
        Generate wordcloud for a specific neuron.

        Args:
            neuron_id: Neuron index
            width: Image width in pixels
            height: Image height in pixels
            background_color: Background color
            colormap: Matplotlib colormap name

        Returns:
            WordCloud object (can be converted to image)
        """
        if not HAS_WORDCLOUD:
            logger.warning("wordcloud library not installed")
            return None

        categories = self.get_categories_for_neuron(neuron_id)
        
        if not categories:
            logger.warning(f"No categories for neuron {neuron_id}")
            return None

        # Repeat categories for better visibility (especially if only 2-3 categories)
        # This ensures they appear prominently in the wordcloud
        repeated_categories = categories * max(5, 10 // max(1, len(categories)))
        text = " ".join(repeated_categories)
        
        if not text.strip():
            logger.warning(f"Empty text for wordcloud (neuron {neuron_id})")
            return None

        try:
            wc = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                colormap=colormap,
                relative_scaling=0.5,
                min_font_size=8,
            ).generate(text)
            
            logger.debug(f"Generated wordcloud for neuron {neuron_id}")
            return wc
        except Exception as e:
            logger.error(f"Failed to generate wordcloud: {e}")
            return None

    def generate_wordcloud_fig(
        self,
        neuron_id: int,
        figsize: tuple = (8, 6),
        **kwargs
    ):
        """
        Generate matplotlib figure with wordcloud.

        Args:
            neuron_id: Neuron index
            figsize: Figure size (width, height)
            **kwargs: Additional arguments for generate_wordcloud

        Returns:
            matplotlib.figure.Figure or None
        """
        import matplotlib.pyplot as plt

        wc = self.generate_wordcloud(neuron_id, **kwargs)
        if wc is None:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_axis_off()
        
        # Add title with neuron label
        label = self.get_neuron_label(neuron_id)
        ax.set_title(f"Feature {neuron_id}: {label}", fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig

    def generate_wordcloud_pdf_bytes(
        self,
        neuron_id: int,
        **kwargs
    ) -> Optional[bytes]:
        """
        Generate wordcloud as PDF bytes for download.

        Returns:
            PDF bytes or None if generation failed
        """
        import io
        from matplotlib.backends.backend_pdf import PdfPages

        fig = self.generate_wordcloud_fig(neuron_id, **kwargs)
        if fig is None:
            return None

        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    def batch_generate_labels_dict(self) -> Dict[int, str]:
        """
        Generate mapping of all neuron IDs to their labels.

        Returns:
            Dict: {neuron_id: label_string}
        """
        labels_dict = {}
        for neuron_id_str in self.labels.keys():
            try:
                neuron_id = int(neuron_id_str)
                labels_dict[neuron_id] = self.labels[neuron_id_str]
            except (ValueError, KeyError):
                pass
        
        logger.info(f"Created labels dict with {len(labels_dict)} entries")
        return labels_dict

    @staticmethod
    def create_summary_wordcloud(
        labels_list: List[str],
        title: str = "Feature Summary",
        figsize: tuple = (10, 6)
    ):
        """
        Create a single wordcloud from multiple labels.

        Useful for showing aggregate feature summary.

        Args:
            labels_list: List of neuron label strings
            title: Title for the wordcloud
            figsize: Figure size

        Returns:
            matplotlib.figure.Figure
        """
        if not HAS_WORDCLOUD:
            return None

        import matplotlib.pyplot as plt

        text = " ".join(labels_list)
        
        try:
            wc = WordCloud(
                width=figsize[0] * 100,
                height=figsize[1] * 100,
                background_color='white',
                colormap='tab20'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(wc, interpolation='bilinear')
            ax.set_axis_off()
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.error(f"Failed to create summary wordcloud: {e}")
            return None
