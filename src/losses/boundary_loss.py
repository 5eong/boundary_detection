"""
Boundary loss utilities and surface loss implementation.

This module contains utilities for boundary-aware loss functions,
including distance transforms and surface loss implementations.
"""

import argparse
from pathlib import Path
from operator import add
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import partial
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import torch.sparse
from tqdm import tqdm
from torch import einsum, Tensor
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt as eucl_distance


# Type variables
A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

# Colors for visualization
COLORS = [
    "c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
    'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato'
]

# Optimized tqdm for progress bars
tqdm_ = partial(
    tqdm, 
    dynamic_ncols=True,
    leave=False,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
)


def str2bool(v: str) -> bool:
    """Convert string to boolean value."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Utility functions
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """Map function over iterable."""
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """Multiprocessing map."""
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    """Multiprocessing starmap."""
    return Pool().starmap(fn, iter)


def uc_(fn: Callable) -> Callable:
    """Return uncurried version of function."""
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    """Uncurry function arguments."""
    return fn(*args)


def id_(x):
    """Identity function."""
    return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    """Flatten nested iterable."""
    return [e for l in to_flat for e in l]


def flatten__(to_flat):
    """Recursively flatten nested structure."""
    if type(to_flat) != list:
        return [to_flat]
    return [e for l in to_flat for e in flatten__(l)]


def depth(e: List) -> int:
    """Compute depth of nested lists."""
    if type(e) == list and e:
        return 1 + depth(e[0])
    return 0


# Tensor operations
def soft_size(a: Tensor) -> Tensor:
    """Compute soft size along spatial dimensions."""
    return torch.einsum("bk...->bk", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    """Compute batch soft size."""
    return torch.einsum("bk...->k", a)[..., None]


# Assert utilities
def uniq(a: Tensor) -> Set:
    """Get unique values from tensor."""
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    """Check if tensor values are subset of given iterable."""
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    """Check tensor equality."""
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis: int = 1) -> bool:
    """Check if tensor satisfies simplex constraint (sums to 1)."""
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis: int = 1) -> bool:
    """Check if tensor is one-hot encoded."""
    return simplex(t, axis) and sset(t, [0, 1])


# Metrics and operations
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    """
    Compute Dice coefficient with flexible summation.
    
    Args:
        sum_str: Einstein summation string
        label: Ground truth labels
        pred: Predictions
        smooth: Smoothing factor
        
    Returns:
        Dice coefficients
    """
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)
    return dices


# Dice coefficient variants
dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # For 3D dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    """Compute intersection of two binary tensors."""
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])
    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    """Compute union of two binary tensors."""
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])
    return res


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    """Sum of intersection."""
    return einsum("bk...->bk", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    """Sum of union."""
    return einsum("bk...->bk", union(a, b).type(torch.float32))


# Representation conversions
def probs2class(probs: Tensor) -> Tensor:
    """Convert probabilities to class indices."""
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)
    return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    """
    Convert class indices to one-hot encoding.
    
    Args:
        seg: Segmentation tensor with class indices
        K: Number of classes
        
    Returns:
        One-hot encoded tensor
    """
    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape
    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    return res


def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoding (NumPy version).
    
    This numpy implementation is used to avoid potential blocking issues
    with multiprocessing.
    """
    b, w, h = seg.shape
    res = np.zeros((b, K, w, h), dtype=np.int64)
    np.put_along_axis(res, seg[:, None, :, :], 1, axis=1)
    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    """Convert probabilities to one-hot encoding."""
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res


def one_hot2dist(
    seg: np.ndarray, 
    resolution: Tuple[float, float, float] = None,
    dtype=None
) -> np.ndarray:
    """
    Convert one-hot segmentation to distance transform.
    
    Args:
        seg: One-hot encoded segmentation
        resolution: Voxel resolution for distance calculation
        dtype: Output data type
        
    Returns:
        Distance transform array
    """
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = (
                eucl_distance(negmask, sampling=resolution) * negmask -
                (eucl_distance(posmask, sampling=resolution) - 1) * posmask
            )

    return res


# Image utilities
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    """
    Save segmentation images to disk.
    
    Args:
        segs: Segmentation tensors
        names: Filenames
        root: Root directory
        mode: Mode subdirectory
        iter: Iteration number
    """
    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(seg.shape) == 2:
            imsave(str(save_path), seg.detach().cpu().numpy().astype(np.uint8))
        elif len(seg.shape) == 3:
            np.save(str(save_path), seg.detach().cpu().numpy())
        else:
            raise ValueError("Unsupported tensor shape for saving")


def augment(
    *arrs: Union[np.ndarray, Image.Image], 
    rotate_angle: float = 45,
    flip: bool = True, 
    mirror: bool = True,
    rotate: bool = True, 
    scale: bool = False
) -> List[Image.Image]:
    """
    Apply data augmentation to arrays/images.
    
    Args:
        arrs: Input arrays or PIL images
        rotate_angle: Maximum rotation angle
        flip: Whether to apply vertical flip
        mirror: Whether to apply horizontal flip
        rotate: Whether to apply rotation
        scale: Whether to apply scaling
        
    Returns:
        List of augmented PIL images
    """
    imgs: List[Image.Image] = (
        map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)
    )

    if flip and random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    
    if mirror and random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    
    if rotate and random() > 0.5:
        angle: float = uniform(-rotate_angle, rotate_angle)
        imgs = map_(lambda e: e.rotate(angle), imgs)
    
    if scale and random() > 0.5:
        scale_factor: float = uniform(1, 1.2)
        w, h = imgs[0].size
        nw, nh = int(w * scale_factor), int(h * scale_factor)

        # Resize and crop
        imgs = map_(lambda i: i.resize((nw, nh)), imgs)
        bw, bh = randint(0, nw - w), randint(0, nh - h)
        imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
        assert all(i.size == (w, h) for i in imgs)

    return imgs


def augment_arr(
    *arrs_a: np.ndarray, 
    rotate_angle: float = 45,
    flip: bool = True, 
    mirror: bool = True,
    rotate: bool = True, 
    scale: bool = False,
    noise: bool = False, 
    noise_loc: float = 0.5, 
    noise_lambda: float = 0.1
) -> List[np.ndarray]:
    """
    Apply data augmentation to NumPy arrays.
    
    Args:
        arrs_a: Input arrays
        rotate_angle: Maximum rotation angle
        flip: Whether to apply vertical flip
        mirror: Whether to apply horizontal flip
        rotate: Whether to apply rotation
        scale: Whether to apply scaling
        noise: Whether to add noise
        noise_loc: Noise location parameter
        noise_lambda: Noise scale parameter
        
    Returns:
        List of augmented arrays
    """
    arrs = list(arrs_a)

    if flip and random() > 0.5:
        arrs = map_(np.flip, arrs)
    
    if mirror and random() > 0.5:
        arrs = map_(np.fliplr, arrs)
    
    if noise and random() > 0.5:
        mask: np.ndarray = np.random.laplace(noise_loc, noise_lambda, arrs[0].shape)
        arrs = map_(partial(add, mask), arrs)
        arrs = map_(lambda e: (e - e.min()) / (e.max() - e.min()), arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    """
    Center crop arrays to target shape.
    
    Args:
        shape: Target shape
        arrs: Input arrays
        
    Returns:
        Center-cropped arrays
    """
    def g_center(arr):
        if arr.shape == shape:
            return arr

        offsets: List[int] = [(arrs - s) // 2 for (arrs, s) in zip(arr.shape, shape)]

        if 0 in offsets:
            return arr[[slice(0, s) for s in shape]]

        res = arr[[slice(d, -d) for d in offsets]][[slice(0, s) for s in shape]]
        assert res.shape == shape, (res.shape, shape, offsets)
        return res

    return [g_center(arr) for arr in arrs]


def center_pad(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Center pad array to target shape.
    
    Args:
        arr: Input array
        target_shape: Target shape
        
    Returns:
        Center-padded array
    """
    assert len(arr.shape) == len(target_shape)

    diff: List[int] = [(nx - x) for (x, nx) in zip(arr.shape, target_shape)]
    pad_width: List[Tuple[int, int]] = [(w // 2, w - (w // 2)) for w in diff]

    res = np.pad(arr, pad_width)
    assert res.shape == target_shape, (res.shape, target_shape)
    return res


class SurfaceLoss:
    """
    Surface loss for boundary-aware segmentation.
    
    This loss function uses distance transforms to weight pixels based on
    their distance from object boundaries, encouraging accurate boundary
    prediction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize surface loss.
        
        Args:
            idc: List of class indices to include in loss computation
        """
        self.idc: List[int] = kwargs.get("idc", [])
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        """
        Compute surface loss.
        
        Args:
            probs: Predicted probabilities
            dist_maps: Distance transform maps
            
        Returns:
            Computed surface loss
        """
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)
        loss = multipled.mean()

        return loss