import torch
import torch.nn.functional as F

um = 1
mm = 1000*um
cm = 10000*um


def differentiable_interpolation_points(grid, points, *, mode="bilinear",
                                        align_corners=True):
    """
    Interpolates a 2D grid at multiple points in a fully differentiable way.

    Parameters:
      grid (torch.Tensor): 2D tensor of shape (H, W) representing the data.
      points (torch.Tensor or tuple/list): Tensor of shape (N, 2) containing N points 
          in pixel coordinates (x, y). If a single point is provided as shape (2,), it 
          will be converted to shape (1, 2).
      mode (str): Interpolation mode ('bilinear' or 'nearest').
      align_corners (bool): Passed to grid_sample to control coordinate normalization.
    
    Returns:
      torch.Tensor: A tensor of shape (N,) containing the interpolated values.
    """
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=grid.dtype, device=grid.device)
    if points.ndim == 1:
        points = points.unsqueeze(0)

    H, W = grid.shape
    grid_img = grid.unsqueeze(0).unsqueeze(0)                     # [1,1,H,W]

    norm_x = 2 * points[:, 0] / (W - 1) - 1                      # [-1,1]
    norm_y = 2 * points[:, 1] / (H - 1) - 1
    sample_grid = torch.stack((norm_x, norm_y), dim=1) \
                        .unsqueeze(0).unsqueeze(2)                # [1,N,1,2]

    return F.grid_sample(grid_img, sample_grid,
                         mode=mode, align_corners=align_corners)\
            .squeeze().flatten()                                  # (N,)


def create_interpolator(grid, region=None, *, mode="bilinear", align_corners=True):
    
    """
    Given a 2D tensor 'grid' sampled over the domain [xmin,xmax] x [ymin,ymax],
    returns a function that interpolates the grid at arbitrary domain points.

    The returned function accepts a tensor of interpolation points (shape: (N, 2)) 
    in domain coordinates and returns a tensor of interpolated values.

    Parameters:
    ----------

      grid (torch.Tensor): 2D tensor of shape (H, W) representing the data.
      region : list/tuple [xmin,xmax,ymin,ymax] specifying the domain in which
                 those samples live.  Default is [-1,1,-1,1].

      mode (str): Interpolation mode ('bilinear' or 'nearest').
      align_corners (bool): Whether to align corners when normalizing coordinates.

    Returns:
      function: A function f(points_domain) -> interpolated_values, where points_domain
                is a tensor of shape (N, 2) in the domain [-1,1]x[-1,1].
    """

    """
    Parameters
    ----------
    grid   : 2-D tensor (H,W)            – samples on a *regular* pixel grid.
    region : list/tuple [xmin,xmax,ymin,ymax] specifying the domain in which
             those samples live.  Default is [-1,1,-1,1].
    mode   : 'bilinear' | 'nearest'
    """
    if region is None:
        region = [-1.0, 1.0, -1.0, 1.0]
    if len(region) != 4:
        raise ValueError("`region` must have four elements: [xmin,xmax,ymin,ymax]")

    xmin, xmax, ymin, ymax = map(float, region)
    if xmax == xmin or ymax == ymin:
        raise ValueError("Region has zero size in at least one dimension.")

    H, W = grid.shape

    def interpolate_fn(x, y):
        # -- to tensor, preserve device/dtype --------------------------------
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=grid.dtype, device=grid.device)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=grid.dtype, device=grid.device)

        orig_shape = x.shape
        if y.shape != orig_shape:
            raise ValueError("`x` and `y` must have the same shape.")

        x_flat = x.contiguous().view(-1)
        y_flat = y.contiguous().view(-1)

        # -- domain → pixel ---------------------------------------------------
        pixel_x = (x_flat - xmin) * (W - 1) / (xmax - xmin)
        pixel_y = (y_flat - ymin) * (H - 1) / (ymax - ymin)
        points_pix = torch.stack((pixel_x, pixel_y), dim=1)        # (N,2)

        out = differentiable_interpolation_points(grid, points_pix,
                                                  mode=mode,
                                                  align_corners=align_corners)
        return out.view(orig_shape)

    return interpolate_fn


from pathlib import Path
from PIL import Image
import numpy as np


def flip_symmetric_array(arr, symmetry=None):
    """
    Return a symmetrised copy of *arr* based on the specified type of symmetry.

    Parameters:
        arr (np.ndarray): A 2D numpy array.
        symmetry (str): The type of symmetry to apply. Options:
                        "xy"       : Creates symmetry with respect to both vertical (y-axis)
                                       and horizontal (x-axis) axes using the upper-right quadrant.
                        "x" : Creates symmetry with respect to the horizontal axis using the upper half.
                        "y"   : Creates symmetry with respect to the vertical axis using the left half.
    
    Returns:
        np.ndarray: A symmetric array based on the selected symmetry type.
        
    Raises:
        ValueError: If an unsupported symmetry type is provided.
    """
    symmetry = symmetry.lower()
    if symmetry == "xy":
        rows, cols = arr.shape
        half_rows = rows // 2
        half_cols = cols // 2
        
        # Extract the upper-right quadrant.
        quadrant = arr[:half_rows, half_cols:]
        
        # Create the top half:
        # Left half is the horizontally flipped quadrant; right half is the quadrant itself.
        top_half = np.concatenate((quadrant[:, ::-1], quadrant), axis=1)
        
        # Create the bottom half by vertically flipping the top half.
        symmetric_array = np.concatenate((top_half, top_half[::-1, :]), axis=0)
        return symmetric_array

    elif symmetry == "x":
        rows, cols = arr.shape
        
        # Extract the upper half (if rows is odd, it takes floor(rows/2)).
        top_half = arr[:rows // 2, :]
        
        # Create the bottom half by vertically flipping the top half.
        symmetric_array = np.concatenate((top_half, top_half[::-1, :]), axis=0)
        return symmetric_array

    elif symmetry == "y":
        rows, cols = arr.shape
        
        # Extract the left half (if cols is odd, it takes floor(cols/2)).
        left_half = arr[:, :cols // 2]
        
        # Create the right half by flipping the left half horizontally.
        symmetric_array = np.concatenate((left_half, left_half[:, ::-1]), axis=1)
        return symmetric_array

    else:
        raise ValueError("Unsupported symmetry type. Choose 'xy', 'x', or 'y'.")


def load_image(path, image_size = None, symmetry=None):

    """Load *path* into a 2‑D float array.

    The function performs **three** independent preprocessing steps:

    1. Read the file with *Pillow* and convert to 8‑bit sRGB.
    2. Convert RGB to luminance (*Y*) using the ITU‑R BT.601 weights
       (0.299R+0.587G+0.114B).
    3. Flip vertically so that the origin lies bottom‑left (NumPy images usually
       have (0,0) at the top‑left, which is awkward for mathematical work).
    4. Optionally resize to *image_size* (Lanczos resampling).
    5. Optionally apply *flip_symmetric_array*.

    Parameters
    ----------
    path
        Path to any raster image supported by *Pillow* (PNG, JPEG, TIFF, …).
    image_size
        ``(width, height)`` tuple to which the image should be resized.  ``None``
        (default) keeps the original resolution.
    symmetry
        See :pyfunc:`flip_symmetric_array`.  ``None`` disables mirroring.
    dtype
        NumPy dtype of the returned array (default **float64**).

    Returns
    -------
    np.ndarray
        2‑D array of shape ``(height, width)`` with values in the range
        :math:`[0, 1]`.

    Examples
    --------
    >>> from pathlib import Path
    >>> img = load_image("input/photo.jpg"), image_size=(512, 512), symmetry=None)
    >>> print(img.min(), img.max(), img.shape)
    0.0 1.0 (512, 512)
    
    """
    img = Image.open(Path(path))
    img = img.convert("RGB")
    imgRGB = np.asarray(img) / 255.0

    t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
    t = np.array(np.flip(t, axis = 0))
    
    if symmetry== None:
        return t
    else:
        return flip_symmetric_array(t, symmetry)
