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

Concrete networks provided:

- :class:`CNN1` -- fixed-propagation-length network.
- :class:`CNNL` -- variable-length network with a length-sensing head that
  concatenates a scalar propagation distance into the shared MLP.
- :class:`CNN1Freq` -- CNN1 augmented with a 2D-FFT log-magnitude spectral
  branch (:class:`FrequencyBranch`).
- :class:`CNN1Star` -- StarNet-style feature extractor (:class:`StarBlock`)
  with optional squeeze-and-excitation attention (:class:`SEBlock`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBN(nn.Sequential):
    """Conv2d optionally followed by BatchNorm2d (StarNet helper).

    When ``with_bn`` is true the conv has no bias (BN absorbs it); otherwise the
    conv keeps its bias.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        with_bn: bool = True,
        groups: int = 1,
    ) -> None:
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not with_bn,
            )
        ]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        super().__init__(*layers)


class StarBlock(nn.Module):
    """StarNet ``Block`` (Zhang et al., "Rewrite the Stars", CVPR 2025).

    Depthwise conv -> two parallel ``1x1`` projections -> "star operation"
    (element-wise product of the two projections) -> ``1x1``+BN -> depthwise
    conv -> residual add. The element-wise product of ``f1``/``f2``
    approximates a high-dimensional linear feature space at low cost.

    Layer ordering (from the timm implementation)::

        x = dwconv7x7(x) + BN
        x1, x2 = f1(x), f2(x)          # two parallel 1x1, no BN
        x = act(x1) * x2               # star operation
        x = g(x) + BN                  # 1x1 + BN
        x = dwconv2_7x7(x)             # depthwise 7x7, no BN
        out = x + residual
    """

    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        m = int(dim * mlp_ratio)
        self.dwconv = ConvBN(dim, dim, 7, padding=3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, m, 1, with_bn=False)
        self.f2 = ConvBN(dim, m, 1, with_bn=False)
        self.g = ConvBN(m, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, padding=3, groups=dim, with_bn=False)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = self.dwconv2(x)
        return x + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration (Hu et al., CVPR 2018).

    Global-average-pool -> bottleneck ``Linear`` -> sigmoid -> channel-wise
    scaling. A lightweight, parameter-efficient attention used to reweight
    feature channels after a spatial stage.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1, 1)
        return x * se


class FrequencyBranch(nn.Module):
    """2D-FFT frequency-domain feature extractor (log-magnitude spectrum).

    Transforms each input plane with ``torch.fft.rfft2``, takes the
    log-magnitude spectrum ``log(1 + |F|)`` (a compact, well-scaled
    representation of the spatial-frequency content), pools it to a fixed
    ``pool x pool`` grid, refines the bands with a small ``1x1`` conv, and
    flattens. Exposes the global / periodic structure of turbulence-degraded
    intensity images that local spatial convolution kernels under-sample.

    This is the frequency-domain branch used by :class:`CNN1Freq` (motivated by
    FFT feature extraction, FNet / Fourier Neural Operator style spectral
    processing).

    Parameters
    ----------
    in_ch : int, optional
        Number of input channels (default 3, one per measurement plane).
    pool : int, optional
        Side length of the adaptive-average-pooled log-magnitude grid
        (default 8).
    refine_ch : int, optional
        Width of the ``1x1`` conv that recombines the spectral bands
        (default 16).
    """

    def __init__(self, in_ch: int = 3, pool: int = 8, refine_ch: int = 16) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.pool = int(pool)
        self.refine_ch = int(refine_ch)
        self.avgpool = nn.AdaptiveAvgPool2d((self.pool, self.pool))
        self.refine = nn.Sequential(
            nn.Conv2d(self.in_ch, self.refine_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.refine_ch),
            nn.ReLU(inplace=True),
        )
        self.freq_size = self.refine_ch * self.pool * self.pool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, N, N)`` intensity images to ``(B, freq_size)`` features.

        Parameters
        ----------
        images : torch.Tensor
            Input intensity images (values in ``[0, 1]``).

        Returns
        -------
        torch.Tensor
            Flattened log-magnitude spectral features of shape
            ``(B, freq_size)``.
        """
        # rfft2 along the last two dims -> (B, C, N, N//2+1) complex.
        F = torch.fft.rfft2(images, norm="ortho")
        mag = torch.log1p(F.abs())            # (B, C, N, N//2+1)
        pooled = self.avgpool(mag)            # (B, C, pool, pool)
        refined = self.refine(pooled)         # (B, refine_ch, pool, pool)
        return torch.flatten(refined, 1)


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


class CNN1Freq(BaseBeaconlessCNN):
    """CNN1 augmented with a frequency-domain (2D-FFT log-magnitude) branch.

    Alongside the standard spatial 3-stage CNN encoder (``_image_encoding``),
    a :class:`FrequencyBranch` computes the log-magnitude 2D-FFT spectrum of
    the input planes. The flattened spatial and spectral encodings are
    concatenated before the shared MLP, so the head can exploit both local
    spatial features and global / periodic frequency structure.

    Parameters
    ----------
    freq_pool : int, optional
        Pool size for the spectral branch (default 8).
    freq_refine_ch : int, optional
        Refinement conv width for the spectral branch (default 16).
    """

    def __init__(
        self,
        *args,
        freq_pool: int = 8,
        freq_refine_ch: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.freq_pool = int(freq_pool)
        self.freq_refine_ch = int(freq_refine_ch)
        self.freq_branch = FrequencyBranch(
            in_ch=3, pool=self.freq_pool, refine_ch=self.freq_refine_ch
        )
        self._freq_embed = self.freq_branch.freq_size
        # Rebuild the MLP with the extra spectral inputs prepended.
        self._build_mlp()

    def _build_mlp(self) -> None:
        """Build the shared MLP given the current flat, length and freq sizes."""
        in_features = self._flat_size + self._length_embed + getattr(
            self, "_freq_embed", 0
        )
        layers = []
        for _ in range(self.mlp_depth):
            layers.append(nn.Linear(in_features, self.mlp_width))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_features = self.mlp_width
        layers.append(nn.Linear(in_features, self.n_modes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map intensity images to ``(B, n_modes)`` via spatial + spectral encodings.

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
        freq = self.freq_branch(images)
        x = torch.cat([x, freq], dim=1)
        return self.mlp(x)


class CNN1Star(BaseBeaconlessCNN):
    """StarNet-style multi-stage feature extractor with optional SE attention.

    Replaces the 3-stage vanilla CNN with a hierarchy of :class:`StarBlock`
    stages (each stage = a stride-2 downsample conv + ``depths[i]`` StarBlocks).
    A final :class:`SEBlock` can be inserted after the last stage to reweight
    channels. The pooled, flattened embedding feeds the shared MLP head as in
    the base model.

    Parameters
    ----------
    base_dim : int, optional
        Channel width of the first stage (doubles each stage, default 32).
    depths : tuple, optional
        Number of StarBlocks per stage (default ``(1, 1, 2)``).
    mlp_ratio : int, optional
        Expansion ratio inside each StarBlock (default 4).
    use_se : bool, optional
        Whether to append an :class:`SEBlock` after the last stage
        (default False).
    se_reduction : int, optional
        SE bottleneck reduction ratio (default 16).
    pool_size : int, optional
        Adaptive average pool side length before flattening (default 12).
    kernel : int, optional
        Downsample-conv kernel size between stages (default 3).
    """

    def __init__(
        self,
        n_modes: int = 78,
        base_dim: int = 32,
        depths: tuple = (1, 1, 2),
        mlp_ratio: int = 4,
        use_se: bool = False,
        se_reduction: int = 16,
        pool_size: int = 12,
        mlp_width: int = 512,
        mlp_depth: int = 4,
        dropout: float = 0.0,
        kernel: int = 3,
    ) -> None:
        nn.Module.__init__(self)
        self.n_modes = int(n_modes)
        self.base_dim = int(base_dim)
        self.depths = tuple(int(d) for d in depths)
        self.mlp_ratio = int(mlp_ratio)
        self.use_se = bool(use_se)
        self.se_reduction = int(se_reduction)
        self.pool_size = int(pool_size)
        self.mlp_width = int(mlp_width)
        self.mlp_depth = int(mlp_depth)
        self.dropout = float(dropout)
        self._length_embed = 0

        # Build the StarNet feature extractor.
        stages: list[nn.Module] = []
        prev_ch = 3
        dim = int(base_dim)
        for i, depth in enumerate(self.depths):
            word = []
            # Each stage begins with a stride-2 downsample conv.
            word.append(
                nn.Sequential(
                    ConvBN(prev_ch, dim, kernel, stride=2, padding=kernel // 2),
                    nn.ReLU(inplace=True),
                )
            )
            for _ in range(depth):
                word.append(StarBlock(dim, mlp_ratio=self.mlp_ratio))
            stages.append(nn.Sequential(*word))
            prev_ch = dim
            dim = dim * 2
        if self.use_se:
            stages.append(SEBlock(prev_ch, reduction=self.se_reduction))
        self.star_features = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self._flat_size = int(prev_ch * self.pool_size * self.pool_size)
        self._build_mlp()

    def _image_encoding(self, images: torch.Tensor) -> torch.Tensor:
        """Run the StarNet extractor and flatten the pooled encoding."""
        x = self.star_features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map intensity images to ``(B, n_modes)`` via the StarNet extractor."""
        x = self._image_encoding(images)
        return self.mlp(x)
