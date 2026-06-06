"""
Attention mechanisms for exoplanet spectroscopy.

Implements:
- Multi-head self-attention for wavelength features
- Spectral feature extraction (peaks, slopes, line strengths)
- Adaptive weighting of absorption features
- Wavelength-dependent importance learning

Theory:
    Attention allows the model to focus on important spectral features
    rather than treating all wavelengths equally.

    For exoplanet spectroscopy:
    - Some wavelengths have strong biosignature features (O3 @ 0.6 μm)
    - Others are dominated by absorption (H2O @ 1.4 μm)
    - Model learns which to focus on

References:
    Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need.
    Advances in Neural Information Processing Systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SpectralAttentionHead(nn.Module):
    """
    Single attention head for spectral features.

    Theory:
        Attention(Q, K, V) = softmax(QK^T / √d_k)V

    where:
        Q = Query (what we're looking for)
        K = Key (what's available in spectrum)
        V = Value (spectral intensities)
        d_k = dimension of keys
    """

    def __init__(self, d_model: int, d_k: int):
        """
        Initialize attention head.

        Args:
            d_model: Model dimension (spectrum resolution)
            d_k: Dimension of keys/queries
        """
        super().__init__()
        self.d_k = d_k

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

        # Output projection
        self.W_o = nn.Linear(d_k, d_model)

    def forward(
        self,
        spectrum: torch.Tensor,  # [batch, wavelengths]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted spectral features.

        Args:
            spectrum: Input spectrum [batch, wavelengths]
            mask: Optional mask for padding or masking regions

        Returns:
            Tuple of (weighted_spectrum, attention_weights)
        """
        batch_size = spectrum.shape[0]

        # Project to Q, K, V
        Q = self.W_q(spectrum)  # [batch, d_k]
        K = self.W_k(spectrum)  # [batch, d_k]
        V = self.W_v(spectrum)  # [batch, d_k]

        # Compute attention scores
        # scores = QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights via softmax
        attention_weights = F.softmax(scores, dim=-1)  # [batch, 1, wavelengths]

        # Apply attention to values
        # context = attention_weights @ V
        context = torch.matmul(attention_weights, V)

        # Project back to spectrum dimension
        output = self.W_o(context)  # [batch, wavelengths]

        return output, attention_weights


class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-head attention for learning multiple spectral feature patterns.

    Different heads can learn to focus on:
    - Head 1: Strong absorption features (H2O, CO2)
    - Head 2: Weak absorption features (O3, CH4)
    - Head 3: Wavelength trends and slopes
    - etc.
    """

    def __init__(
        self,
        d_model: int,  # Spectrum resolution (512)
        num_heads: int = 4,
        d_k: int = 64,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Spectrum dimension
            num_heads: Number of attention heads
            d_k: Dimension per head
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k

        # Attention heads
        self.heads = nn.ModuleList([
            SpectralAttentionHead(d_model, d_k)
            for _ in range(num_heads)
        ])

        # Final linear layer to combine heads
        self.W_o = nn.Linear(num_heads * d_k, d_model)

        # Layer norm for residual connection
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        spectrum: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multi-head attention.

        Args:
            spectrum: Input spectrum [batch, wavelengths]
            mask: Optional mask

        Returns:
            Tuple of (output_spectrum, attention_weights_per_head)
        """
        # Apply each attention head
        head_outputs = []
        attention_maps = []

        for head in self.heads:
            output, attn_weights = head(spectrum, mask)
            head_outputs.append(output)
            attention_maps.append(attn_weights)

        # Concatenate head outputs
        # [batch, d_k] × num_heads → [batch, num_heads * d_k]
        combined = torch.cat(head_outputs, dim=-1)

        # Project back to spectrum dimension
        output = self.W_o(combined)

        # Residual connection + layer norm
        output = self.norm(spectrum + output)

        return output, attention_maps


class SpectralFeatureExtractor(nn.Module):
    """
    Extract meaningful features from spectrum using attention.

    Features extracted:
    - Peak locations and strengths
    - Continuum level
    - Slope/trends
    - Line strength ratios
    - Absorption feature patterns
    """

    def __init__(
        self,
        spectrum_dim: int = 512,
        num_attention_heads: int = 4,
        hidden_dim: int = 128,
    ):
        """
        Initialize spectral feature extractor.

        Args:
            spectrum_dim: Input spectrum resolution
            num_attention_heads: Number of attention heads
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Multi-head attention for feature learning
        self.attention = MultiHeadSpectralAttention(
            d_model=spectrum_dim,
            num_heads=num_attention_heads,
            d_k=spectrum_dim // num_attention_heads,
        )

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.spectrum_dim = spectrum_dim
        self.hidden_dim = hidden_dim

    def extract_peak_features(
        self,
        spectrum: torch.Tensor,
        window_size: int = 5,
    ) -> torch.Tensor:
        """
        Extract peak-based features from spectrum.

        Features:
        - Peak locations
        - Peak strengths
        - Peak widths
        - Number of peaks

        Args:
            spectrum: [batch, wavelengths]
            window_size: Window for peak detection

        Returns:
            Peak features [batch, num_peaks * 3]
        """
        # Detect peaks by finding local minima (absorption features)
        # Absorption = 1 - transit_depth
        batch_size = spectrum.shape[0]

        # Smooth spectrum for peak detection
        kernel = torch.ones(1, 1, window_size) / window_size
        spectrum_smooth = F.conv1d(
            spectrum.unsqueeze(1),
            kernel,
            padding=window_size // 2,
        ).squeeze(1)

        # Find where derivative changes sign (peaks/valleys)
        diff = spectrum_smooth[:, 1:] - spectrum_smooth[:, :-1]

        # Extract feature statistics
        peak_strength = torch.abs(spectrum).max(dim=1)[0]
        peak_mean = spectrum.mean(dim=1)
        peak_std = spectrum.std(dim=1)

        # Combine into feature vector
        features = torch.stack([peak_strength, peak_mean, peak_std], dim=1)

        return features

    def extract_continuum_features(
        self,
        spectrum: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract continuum-level features.

        Continuum is the baseline absorption level,
        important for determining atmospheric scale height.

        Args:
            spectrum: [batch, wavelengths]

        Returns:
            Continuum features [batch, 5]
        """
        # Fit continuum as polynomial background
        spectrum_np = spectrum.detach().cpu().numpy()

        continuum_level = spectrum.min(dim=1)[0]  # Baseline
        continuum_slope = (spectrum[:, -1] - spectrum[:, 0]) / self.spectrum_dim
        continuum_curvature = spectrum.std(dim=1) / spectrum.mean(dim=1)

        features = torch.stack([
            continuum_level,
            continuum_slope,
            continuum_curvature,
            spectrum.mean(dim=1),
            spectrum.max(dim=1)[0],
        ], dim=1)

        return features

    def forward(self, spectrum: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Extract all spectral features.

        Args:
            spectrum: [batch, wavelengths]

        Returns:
            Tuple of (combined_features, feature_dict)
        """
        # Apply attention
        attentive_spectrum, attention_maps = self.attention(spectrum)

        # Extract different feature types
        peak_features = self.extract_peak_features(spectrum)
        continuum_features = self.extract_continuum_features(spectrum)

        # Process through feature network
        network_features = self.feature_net(attentive_spectrum)

        # Combine all features
        combined = torch.cat([
            peak_features,
            continuum_features,
            network_features,
        ], dim=1)

        # Return with feature dictionary for interpretability
        feature_dict = {
            'peak_features': peak_features,
            'continuum_features': continuum_features,
            'network_features': network_features,
            'attention_maps': attention_maps,
            'attentive_spectrum': attentive_spectrum,
        }

        return combined, feature_dict


class WavelengthEmbedding(nn.Module):
    """
    Learn wavelength-dependent importance weights.

    Different wavelengths have different importance:
    - 0.6 μm: O3 band (biosignature!)
    - 1.4 μm: H2O strong absorption
    - 2.7 μm: H2O, CO2 combination
    - 4.3 μm: CO2 strong absorption

    This module learns which wavelengths matter for composition inference.
    """

    def __init__(
        self,
        spectrum_dim: int = 512,
        embedding_dim: int = 64,
    ):
        """
        Initialize wavelength embedding.

        Args:
            spectrum_dim: Number of wavelength points
            embedding_dim: Embedding dimension
        """
        super().__init__()

        # Positional encoding for wavelengths
        self.wavelength_embedding = nn.Embedding(spectrum_dim, embedding_dim)

        # Learn importance weights
        self.importance_net = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.spectrum_dim = spectrum_dim

    def forward(self, spectrum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute wavelength importance and apply weighting.

        Args:
            spectrum: [batch, wavelengths]

        Returns:
            Tuple of (weighted_spectrum, importance_weights)
        """
        batch_size = spectrum.shape[0]

        # Get wavelength indices
        wavelength_indices = torch.arange(
            self.spectrum_dim,
            device=spectrum.device,
        )

        # Embed wavelengths
        wl_embeddings = self.wavelength_embedding(wavelength_indices)  # [wl, emb]

        # Compute importance for each wavelength
        importance = self.importance_net(wl_embeddings).squeeze(-1)  # [wl]

        # Apply importance weighting
        # Broadcast to batch dimension
        importance_expanded = importance.unsqueeze(0).expand(batch_size, -1)
        weighted_spectrum = spectrum * importance_expanded

        return weighted_spectrum, importance


class AttentionVisualization:
    """
    Utilities for visualizing attention weights.

    Useful for understanding which wavelengths the model focuses on.
    """

    @staticmethod
    def get_attention_heatmap(
        attention_weights: torch.Tensor,
        wavelengths: Optional[List[float]] = None,
    ) -> dict:
        """
        Convert attention weights to interpretable format.

        Args:
            attention_weights: Attention tensor from model
            wavelengths: Optional wavelength array (micrometers)

        Returns:
            Dictionary with heatmap data
        """
        # Convert to numpy
        attn_np = attention_weights.detach().cpu().numpy()

        return {
            'attention_map': attn_np,
            'max_attention_idx': attn_np.argmax(axis=-1),
            'mean_attention': attn_np.mean(axis=0),
            'wavelengths': wavelengths,
        }

    @staticmethod
    def get_attention_statistics(
        attention_weights: List[torch.Tensor],
    ) -> dict:
        """
        Compute statistics across all attention heads.

        Args:
            attention_weights: List of attention tensors (one per head)

        Returns:
            Statistics dictionary
        """
        stats = {}

        for i, weights in enumerate(attention_weights):
            weights_np = weights.detach().cpu().numpy()
            stats[f'head_{i}_max'] = weights_np.max()
            stats[f'head_{i}_mean'] = weights_np.mean()
            stats[f'head_{i}_entropy'] = -((weights_np * np.log(weights_np + 1e-10)).sum())

        return stats
