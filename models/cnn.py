"""Beaconless adaptive-optics CNN models.

Architecture follows Optics Express 33(15):31010, 2025:

- 3-stage CNN: each stage is ``3x3 conv (stride 1, padding 0) -> BatchNorm2d ->
  ReLU -> 2x2 MaxPool``. Channels are config-driven, default ``[32, 64, 128]``
  (Table 1: kernel 3x3, stride 1, padding 0).
- Input is ``(B, 3, 512, 512)`` intensity images. After 3 max-pools the feature
  maps are ``62x62`` (128 channels), then ``AdaptiveAvgPool2d((18, 18))`` and
  flattened to ``128 * 18 * 18 = 41472``.
- A shared MLP of 4 hidden layers of 512 ReLU neurons maps the flattened encoding
  to ``n_modes`` (default 78) Zernike-mode outputs.

Two concrete networks are provided:

- :class:`CNN1` -- fixed-propagation-length network.
- :class:`CNNL` -- variable-length network with a length-sensing head that
  concatenates a scalar propagation distance into the shared MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in ``model``.

    Parameters
    ----------
    model : nn.Module
        The module whose parameters are counted.

    Returns
    -------
    int
        ``sum(p.numel() for p in model.parameters())``.
    """
    return sum(p.numel() for p in model.parameters())


class BaseBeaconlessCNN(nn.Module):
    """3-stage CNN -> AdaptiveAvgPool2d((18,18)) -> flatten -> MLP 4x512 ReLU -> n_modes.

    Config-driven channels: ``[32, 64, 128]``. Input: ``(B, 3, 512, 512)``
    intensity images.

    Parameters
    ----------
    n_modes : int, optional
        Number of output Zernike modes (default 78).
    channels : tuple, optional
        Channel counts per stage (default ``(32, 64, 128)``).
    pool_size : int, optional
        Side length of the adaptive average pool output (default 18).
    mlp_width : int, optional
        Width of each hidden MLP layer (default 512).
    mlp_depth : int, optional
        Number of hidden MLP layers (default 4).
    dropout : float, optional
        Dropout probability applied after each ReLU in the MLP (default 0.0).
    """

    def __init__(
        self,
        n_modes: int = 78,
        channels: tuple = (32, 64, 128),
        pool_size: int = 18,
        mlp_width: int = 512,
        mlp_depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.channels = tuple(channels)
        self.pool_size = pool_size
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.dropout = dropout

        # 3-stage CNN feature extractor.
        stages = []
        in_ch = 3
        for out_ch in self.channels:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            in_ch = out_ch
        self.features = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))

        # Flattened image encoding size (computed lazily in _build_mlp).
        self._flat_size = self.channels[-1] * self.pool_size * self.pool_size

        # Length-sensing head is defined only in CNNL; CNN1 has no such attribute.
        self._length_embed = 0

        self._build_mlp()

    def _build_mlp(self) -> None:
        """Build the shared MLP given the current flat size and length embedding."""
        in_features = self._flat_size + self._length_embed
        layers = []
        for _ in range(self.mlp_depth):
            layers.append(nn.Linear(in_features, self.mlp_width))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_features = self.mlp_width
        layers.append(nn.Linear(in_features, self.n_modes))
        self.mlp = nn.Sequential(*layers)

    def _image_encoding(self, images: torch.Tensor) -> torch.Tensor:
        """Run the CNN feature extractor and flatten the pooled encoding."""
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map intensity images to ``(B, n_modes)`` mode coefficients.

        Parameters
        ----------
        images : torch.Tensor
            Intensity images of shape ``(B, 3, 512, 512)``.

        Returns
        -------
        torch.Tensor
            Mode coefficients of shape ``(B, n_modes)``.
        """
        x = self._image_encoding(images)
        return self.mlp(x)


class CNN1(BaseBeaconlessCNN):
    """Fixed-propagation-length network.

    ``forward(images) -> (B, n_modes)``. No length-sensing head.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map intensity images to ``(B, n_modes)`` mode coefficients.

        Parameters
        ----------
        images : torch.Tensor
            Intensity images of shape ``(B, 3, 512, 512)``.

        Returns
        -------
        torch.Tensor
            Mode coefficients of shape ``(B, n_modes)``.
        """
        return super().forward(images)


class CNNL(BaseBeaconlessCNN):
    """Variable-length network with a length-sensing head.

    A scalar propagation length ``L`` is mapped through a 512-neuron ReLU MLP and
    concatenated with the flattened image encoding before the shared 4x512 MLP.

    Parameters
    ----------
    length_head_width : int, optional
        Width of the length-head MLP (default 512).
    """

    def __init__(self, *args, length_head_width: int = 512, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.length_head_width = length_head_width
        self.length_head = nn.Sequential(
            nn.Linear(1, self.length_head_width),
            nn.ReLU(inplace=True),
        )
        self._length_embed = self.length_head_width
        # Rebuild the MLP to account for the extra length-head inputs.
        self._build_mlp()

    def forward(self, images: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Map intensity images and propagation length to ``(B, n_modes)``.

        Parameters
        ----------
        images : torch.Tensor
            Intensity images of shape ``(B, 3, 512, 512)``.
        length : torch.Tensor
            Propagation distance in metres, shape ``(B,)`` float tensor.

        Returns
        -------
        torch.Tensor
            Mode coefficients of shape ``(B, n_modes)``.
        """
        x = self._image_encoding(images)
        length = length.view(-1, 1)
        length_enc = self.length_head(length)
        x = torch.cat([x, length_enc], dim=1)
        return self.mlp(x)
