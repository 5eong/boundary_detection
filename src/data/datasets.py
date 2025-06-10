"""
Dataset classes for field delineation using satellite imagery.
"""

import os
import pathlib
import random
import copy
from typing import Dict, Any, Union, Optional, List

import cv2
import numpy as np
import rasterio
import torch
from torch import from_numpy, ones_like
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from skimage import exposure
from scipy.ndimage import binary_dilation
from tqdm import tqdm


class AI4Boundaries(Dataset):
    """
    AI4Boundaries dataset for agricultural field boundary detection.
    
    Supports both single-image and multi-temporal Sentinel-2 imagery.
    """
    
    def __init__(
        self, 
        directory: str, 
        source: Optional[str] = None, 
        augment: bool = False, 
        upsample_factor: int = 1, 
        cache_data: bool = False
    ):
        """
        Initialize AI4Boundaries dataset.
        
        Args:
            directory: Path to dataset directory
            source: Data source ('ESRI' or None for multi-temporal)
            augment: Whether to apply data augmentation
            upsample_factor: Factor for upsampling images
            cache_data: Whether to cache data in memory
        """
        self.directory = pathlib.Path(directory)
        self.source = source
        self.upsample_factor = upsample_factor
        self.augment = augment
        self.cache_data = cache_data
        self.data = []
        
        # Set file patterns based on source
        if source == 'ESRI':
            self.paths = list(self.directory.glob("*_S2_10m_256.tif"))
        else:
            self.paths = list(self.directory.glob("*_S2_10m_256_0[2-5].tif"))

        if self.cache_data:
            print("Pre-caching data...")
            for path in tqdm(self.paths):
                self.data.append(self._load_data(path))

    def _load_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load data for a single sample."""
        if self.source == 'ESRI':
            return self._load_esri_data(file_path)
        else:
            return self._load_multitemporal_data(file_path)

    def _load_esri_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load ESRI single-image data."""
        # Load raster
        with rasterio.open(file_path) as src:
            raster = src.read((3, 2, 1))
        raster = self._normalize_band(raster)

        # Load mask
        mask_path = self._get_mask_path(file_path, 'mask')
        with rasterio.open(mask_path) as src:
            band_1 = cv2.resize(src.read(1), (1280, 1280), interpolation=cv2.INTER_LINEAR)
            band_2 = cv2.resize(src.read(2), (1280, 1280), interpolation=cv2.INTER_NEAREST)
            band_3 = cv2.resize(src.read(3), (1280, 1280), interpolation=cv2.INTER_NEAREST)
            mask = cv2.merge((band_2, band_3, band_1))

        raster, mask = self._random_crop(raster, mask)
        return self._prepare_tensors(raster, mask, file_path, mask_path)

    def _load_multitemporal_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load multi-temporal data."""
        # Get preceding and following time periods
        preceding = self._get_temporal_path(file_path, -1)
        following = self._get_temporal_path(file_path, 1)

        # Stack temporal data
        stack = []
        for time_path in [preceding, file_path, following]:
            with rasterio.open(time_path) as src:
                raster = src.read((1, 2, 3))
                raster = self._normalize_band(raster)
                stack.append(raster)

        raster = np.concatenate(stack, axis=0)

        # Load mask
        mask_path = file_path.parent / f"{file_path.name[:-7]}.tif"
        with rasterio.open(mask_path) as src:
            mask = np.stack([src.read(1), src.read(2), src.read(3)], axis=0)

        return self._prepare_tensors(raster, mask, file_path, mask_path)

    def _get_temporal_path(self, file_path: pathlib.Path, offset: int) -> pathlib.Path:
        """Get path for temporal offset."""
        stem = file_path.stem
        new_index = int(stem[-1]) + offset
        return file_path.parent / f"{stem[:-1]}{new_index}{file_path.suffix}"

    def _get_mask_path(self, file_path: pathlib.Path, mask_dir: str) -> pathlib.Path:
        """Get corresponding mask path."""
        path_parts = str(file_path).split(os.sep)
        parent = str(file_path.parent).replace(path_parts[-2], mask_dir)
        return pathlib.Path(parent) / file_path.name

    def _prepare_tensors(
        self, 
        raster: np.ndarray, 
        mask: np.ndarray, 
        file_path: pathlib.Path, 
        mask_path: pathlib.Path
    ) -> Dict[str, Any]:
        """Convert arrays to tensors and apply augmentation."""
        raster = from_numpy(raster)
        mask = from_numpy(mask)
        binary = mask.detach().clone()
        binary[binary != 0] = 0

        if self.augment:
            raster, mask, binary = self._augment_images(raster, mask, binary)

        return {
            'image': raster, 
            'mask': mask, 
            'binary': binary, 
            'file_path': str(file_path), 
            'mask_path': str(mask_path)
        }

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band using percentile clipping."""
        band_min, band_max = np.percentile(band, [1, 99])
        normalized = (band - band_min) / (band_max - band_min)
        return np.clip(normalized, 0, 1)

    def _random_crop(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        crop_size: tuple = (256, 256)
    ) -> tuple:
        """Apply random crop to image and mask."""
        image = image.transpose(1, 2, 0)
        
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and mask must have the same dimensions")

        height, width = image.shape[:2]
        crop_height, crop_width = crop_size

        if height < crop_height or width < crop_width:
            raise ValueError("Crop size is larger than image dimensions")

        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)

        image = image[top:top + crop_height, left:left + crop_width]
        mask = mask[top:top + crop_height, left:left + crop_width]

        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def _augment_images(
        self, 
        raster: torch.Tensor, 
        mask: torch.Tensor, 
        binary: torch.Tensor, 
        rotation: str = 'ortho'
    ) -> tuple:
        """Apply data augmentation."""
        # Random flips
        if random.random() > 0.5:
            raster = TF.hflip(raster)
            mask = TF.hflip(mask)
            binary = TF.hflip(binary)

        if random.random() > 0.5:
            raster = TF.vflip(raster)
            mask = TF.vflip(mask)
            binary = TF.vflip(binary)

        # Rotation
        if rotation == 'ortho':
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                raster = TF.rotate(raster, angle)
                mask = TF.rotate(mask, angle)
                binary = TF.rotate(binary, angle)
        elif rotation == 'random':
            degrees = random.randint(0, 360)
            raster_height, raster_width = raster.shape[:2]
            center = (
                random.randint(0, raster_width - 1), 
                random.randint(0, raster_height - 1)
            )
            raster = TF.rotate(raster, angle=degrees, center=center, expand=False)
            mask = TF.rotate(mask, angle=degrees, center=center, expand=False)
            binary = TF.rotate(binary, angle=degrees, center=center, expand=False)

        return raster, mask, binary

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.cache_data:
            return self.data[index]
        else:
            return self._load_data(self.paths[index])


class Euro_0512(Dataset):
    """
    European satellite imagery dataset with multiple data sources.
    """
    
    def __init__(
        self, 
        directory: str, 
        channels: int = 9, 
        augment: bool = False, 
        upsample_factor: int = 1, 
        cache_data: bool = False
    ):
        """
        Initialize Euro_0512 dataset.
        
        Args:
            directory: Path to dataset directory
            channels: Number of input channels (3 or 9)
            augment: Whether to apply data augmentation
            upsample_factor: Factor for upsampling
            cache_data: Whether to cache data in memory
        """
        self.directory = pathlib.Path(directory)
        self.channels = channels
        self.upsample_factor = upsample_factor
        self.augment = augment
        self.cache_data = cache_data
        self.data = []
        
        self.paths = list(self.directory.glob("*_*.tif"))
        
        if self.cache_data:
            print("Pre-caching data...")
            for path in tqdm(self.paths):
                self.data.append(self._load_data(path))

    def _load_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load data for a single sample."""
        # Load mask
        with rasterio.open(file_path) as src:
            band_1 = cv2.merge((src.read(2), src.read(3), src.read(1)))
        mask = band_1
        
        # Load raster based on channel configuration
        if self.channels == 9:
            raster = self._load_multi_source_raster(file_path)
        else:
            raster = self._load_single_source_raster(file_path)
        
        raster, mask = self._random_crop(raster, mask)
        
        # Convert to tensors
        raster = from_numpy(raster)
        mask = from_numpy(mask)
        binary = ones_like(mask)

        if self.augment:
            raster, mask, binary = self._augment_images(raster, mask, binary)

        return {
            'image': raster.float(), 
            'mask': mask.float(), 
            'binary': binary.float(), 
            'file_path': str(file_path), 
            'mask_path': str(file_path)
        }

    def _load_multi_source_raster(self, file_path: pathlib.Path) -> np.ndarray:
        """Load raster from multiple sources (Bing, ESRI, Google)."""
        sources = ['Bing', 'ESRI', 'Google']
        stack = []
        
        for source in sources:
            source_path = self._get_source_path(file_path, source)
            with rasterio.open(source_path) as src:
                raster = src.read((1, 2, 3))
                raster = self._normalize_band(raster)
                stack.append(raster)
        
        return np.concatenate(stack, axis=0)

    def _load_single_source_raster(self, file_path: pathlib.Path) -> np.ndarray:
        """Load raster from a single randomly chosen source."""
        sources = ['Bing', 'ESRI', 'Google']
        source = random.choice(sources)
        source_path = self._get_source_path(file_path, source)
        
        with rasterio.open(source_path) as src:
            raster = src.read((3, 2, 1))
        
        return self._normalize_band(raster)

    def _get_source_path(self, file_path: pathlib.Path, source: str) -> pathlib.Path:
        """Get path for specific data source."""
        parent_str = str(file_path.parent).replace('\\shapefiles\\Masks', f'\\{source}\\Euro_0512')
        return pathlib.Path(parent_str) / file_path.name

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band using percentile clipping."""
        band_min, band_max = np.percentile(band, [1, 99])
        normalized = (band - band_min) / (band_max - band_min)
        return np.clip(normalized, 0, 1)

    def _random_crop(self, image: np.ndarray, mask: np.ndarray, crop_size: tuple = (256, 256)) -> tuple:
        """Apply random crop to image and mask."""
        image = image.transpose(1, 2, 0)
        
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and mask must have the same dimensions")

        height, width = image.shape[:2]
        crop_height, crop_width = crop_size

        if height < crop_height or width < crop_width:
            raise ValueError("Crop size is larger than image dimensions")

        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)

        image = image[top:top + crop_height, left:left + crop_width]
        mask = mask[top:top + crop_height, left:left + crop_width]

        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def _augment_images(self, raster: torch.Tensor, mask: torch.Tensor, binary: torch.Tensor) -> tuple:
        """Apply data augmentation."""
        # Random flips
        if random.random() > 0.5:
            raster = TF.hflip(raster)
            mask = TF.hflip(mask)
            binary = TF.hflip(binary)

        if random.random() > 0.5:
            raster = TF.vflip(raster)
            mask = TF.vflip(mask)
            binary = TF.vflip(binary)

        # Orthogonal rotation
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            raster = TF.rotate(raster, angle)
            mask = TF.rotate(mask, angle)
            binary = TF.rotate(binary, angle)

        return raster, mask, binary

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.cache_data:
            return self.data[index]
        else:
            return self._load_data(self.paths[index])


class MyanmarSatellite(Dataset):
    """
    Myanmar satellite imagery dataset with multi-source data.
    """
    
    def __init__(
        self, 
        directory: str, 
        channels: int = 9, 
        augment: bool = False, 
        upsample_factor: int = 1, 
        cache_data: bool = False
    ):
        """
        Initialize Myanmar satellite dataset.
        
        Args:
            directory: Path to dataset directory
            channels: Number of input channels (3 or 9)
            augment: Whether to apply data augmentation
            upsample_factor: Factor for upsampling
            cache_data: Whether to cache data in memory
        """
        self.directory = pathlib.Path(directory)
        self.channels = channels
        self.upsample_factor = upsample_factor
        self.augment = augment
        self.cache_data = cache_data
        self.data = []

        self.paths = list(self.directory.glob("[T]_Point_*.tif"))

        if self.cache_data:
            print("Pre-caching data...")
            for path in tqdm(self.paths):
                self.data.append(self._load_data(path))

    def _load_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load data for a single sample."""
        # Load mask
        with rasterio.open(file_path) as src:
            mask = np.stack([
                src.read(2).astype(np.float32), 
                src.read(3).astype(np.float32), 
                src.read(1).astype(np.float32)
            ], axis=0)
            field_mask = copy.deepcopy(mask)
            mask[mask > 0] = 1

        # Load raster
        if self.channels == 9:
            raster = self._load_multi_source_raster(file_path)
        else:
            raster = self._load_single_source_raster(file_path)

        # Convert to tensors
        raster_tensor = from_numpy(raster)
        mask_tensor = from_numpy(mask)
        field_tensor = from_numpy(field_mask)

        # Create binary mask
        binary = self._create_binary_mask(file_path, mask_tensor)

        if self.augment:
            raster_tensor, mask_tensor, binary = self._augment_images(
                raster_tensor, mask_tensor, binary
            )

        return {
            'image': raster_tensor.float(), 
            'mask': mask_tensor.float(), 
            'field': field_tensor.float(), 
            'binary': binary.float(), 
            'file_path': str(file_path), 
            'mask_path': str(file_path)
        }

    def _load_multi_source_raster(self, file_path: pathlib.Path) -> np.ndarray:
        """Load raster from multiple sources."""
        sources = ['Bing', 'ESRI', 'Google']
        stack = []
        
        for source in sources:
            source_path = self._get_source_path(file_path, source)
            with rasterio.open(source_path) as src:
                raster = src.read((1, 2, 3))
                raster = self._normalize_skimage(raster)
                stack.append(raster)
        
        return np.concatenate(stack, axis=0)

    def _load_single_source_raster(self, file_path: pathlib.Path) -> np.ndarray:
        """Load raster from a single randomly chosen source."""
        sources = ['Bing', 'ESRI', 'Google']
        source_path = self._get_source_path(file_path, random.choice(sources))
        
        with rasterio.open(source_path) as src:
            raster = src.read((3, 2, 1))
            
        return self._normalize_skimage(raster)

    def _get_source_path(self, file_path: pathlib.Path, source: str) -> pathlib.Path:
        """Get path for specific data source."""
        parent_str = str(file_path.parent).replace('\\Masks', f'\\{source}\\Images')
        return pathlib.Path(parent_str) / file_path.name

    def _normalize_skimage(
        self, 
        raster: np.ndarray, 
        lower_percentile: float = 2, 
        upper_percentile: float = 98
    ) -> np.ndarray:
        """Normalize using scikit-image exposure rescaling."""
        p2, p98 = np.percentile(raster, (lower_percentile, upper_percentile))
        return exposure.rescale_intensity(raster, in_range=(p2, p98), out_range=(0, 1))

    def _create_binary_mask(self, file_path: pathlib.Path, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Create binary mask based on file naming convention."""
        filename = os.path.basename(file_path)
        
        if any(substring in filename for substring in ["U_", "M_", "Z_"]):
            # For certain file types, use all ones
            binary = torch.ones_like(mask_tensor).to(torch.bool)
        else:
            # For others, dilate the mask
            binary = (mask_tensor != 0).detach().clone()
            binary_array = binary_dilation(binary[0, :, :], iterations=2)
            binary[0:, :, :] = torch.from_numpy(binary_array).to(torch.bool)
        
        return binary

    def _augment_images(self, raster: torch.Tensor, mask: torch.Tensor, binary: torch.Tensor) -> tuple:
        """Apply data augmentation."""
        # Random flips
        if random.random() > 0.5:
            raster = TF.hflip(raster)
            mask = TF.hflip(mask)
            binary = TF.hflip(binary)

        if random.random() > 0.5:
            raster = TF.vflip(raster)
            mask = TF.vflip(mask)
            binary = TF.vflip(binary)

        # Orthogonal rotation
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            raster = TF.rotate(raster, angle)
            mask = TF.rotate(mask, angle)
            binary = TF.rotate(binary, angle)

        return raster, mask, binary

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.cache_data:
            return self.data[index]
        else:
            return self._load_data(self.paths[index])


class SingleImage(Dataset):
    """
    Single image dataset for field delineation.
    """
    
    def __init__(
        self, 
        directory: str, 
        augment: bool = False, 
        upsample_factor: int = 1, 
        cache_data: bool = False
    ):
        """
        Initialize single image dataset.
        
        Args:
            directory: Path to dataset directory
            augment: Whether to apply data augmentation
            upsample_factor: Factor for upsampling
            cache_data: Whether to cache data in memory
        """
        self.directory = pathlib.Path(directory)
        self.upsample_factor = upsample_factor
        self.augment = augment
        self.cache_data = cache_data
        self.data = []

        self.paths = list(self.directory.glob("*_S2_10m_256_0[1-6].tif"))

        if self.cache_data:
            print("Pre-caching data...")
            for path in tqdm(self.paths):
                self.data.append(self._load_data(path))

    def _load_data(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Load data for a single sample."""
        # Load raster
        with rasterio.open(file_path) as src:
            raster = src.read((1, 2, 3))
        raster = self._normalize_band(raster)

        # Load mask
        mask_path = file_path.parent / f"{file_path.name[:-7]}.tif"
        with rasterio.open(mask_path) as src:
            mask = np.stack([src.read(1), src.read(2), src.read(3)], axis=0)

        # Convert to tensors
        raster = from_numpy(raster)
        mask = from_numpy(mask)
        binary = (mask != 0).detach().clone()

        if self.augment:
            raster, mask, binary = self._augment_images(raster, mask, binary)

        return {
            'image': raster.float(), 
            'mask': mask.float(), 
            'binary': binary.float(), 
            'file_path': str(file_path), 
            'mask_path': str(mask_path)
        }

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band using percentile clipping."""
        band_min, band_max = np.percentile(band, [1, 99])
        normalized = (band - band_min) / (band_max - band_min)
        return np.clip(normalized, 0, 1)

    def _augment_images(self, raster: torch.Tensor, mask: torch.Tensor, binary: torch.Tensor) -> tuple:
        """Apply data augmentation."""
        # Random flips
        if random.random() > 0.5:
            raster = TF.hflip(raster)
            mask = TF.hflip(mask)
            binary = TF.hflip(binary)

        if random.random() > 0.5:
            raster = TF.vflip(raster)
            mask = TF.vflip(mask)
            binary = TF.vflip(binary)

        # Orthogonal rotation
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            raster = TF.rotate(raster, angle)
            mask = TF.rotate(mask, angle)
            binary = TF.rotate(binary, angle)

        return raster, mask, binary

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.cache_data:
            return self.data[index]
        else:
            return self._load_data(self.paths[index])


class MyanmarSentinel(Dataset):
    """
    Myanmar Sentinel dataset for inference and prediction.
    """
    
    def __init__(
        self, 
        directory: str, 
        transform=None, 
        upsample_factor: int = 2
    ):
        """
        Initialize Myanmar Sentinel dataset.
        
        Args:
            directory: Path to dataset directory
            transform: Optional transforms to apply
            upsample_factor: Factor for upsampling
        """
        self.directory = pathlib.Path(directory)
        self.transform = transform
        self.upsample_factor = upsample_factor
        
        self.paths = list(self.directory.glob("Hinthada_Mar-Apr_2023*.tif"))
        
    def _upsample(self, image: np.ndarray, scale_factor: int, resample_mode) -> np.ndarray:
        """Upsample image using PIL."""
        pil_image = Image.fromarray(image)
        new_size = (
            int(pil_image.width * scale_factor), 
            int(pil_image.height * scale_factor)
        )
        return np.array(pil_image.resize(new_size, resample_mode))

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band values."""
        return (band - band.min()) / (band.max() - band.min())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_path = self.paths[index]

        # Read and resample temporal images
        raster_stack = []
        for time_period in ['Jan-Feb', 'Mar-Apr', 'May-Jun']:
            temp_path = (
                file_path.parent / 
                f"{file_path.name[:9]}{time_period}{file_path.name[16:]}"
            )
            with rasterio.open(temp_path) as src:
                raster = src.read((2, 3, 4))
                for band in range(raster.shape[0]):
                    raster[band] = self._upsample(
                        raster[band], self.upsample_factor, Image.LANCZOS
                    )
            raster_stack.append(raster)
            
        # Stack and normalize
        raster = np.vstack(raster_stack)
        for band in range(raster.shape[0]):
            raster[band] = self._normalize_band(raster[band])
        
        raster = raster.astype(np.float32)
        raster = from_numpy(raster)

        if self.transform:
            raster = self.transform(raster)

        return {'image': raster, 'file_path': str(file_path)}