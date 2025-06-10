"""
Data preprocessing utilities for satellite imagery.

This module contains functions for converting and preprocessing satellite data
from various formats (NetCDF, GeoTIFF, etc.) for use in machine learning pipelines.
"""

import os
import glob
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import netCDF4

try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("Warning: GDAL not available. GeoTIFF conversion functions will not work.")


def netcdf_to_geotiff(input_path: str, output_path: str) -> None:
    """
    Convert NetCDF file to GeoTIFF format.
    
    This function reads a NetCDF file and converts it to multiple GeoTIFF files,
    one for each time step in the data.
    
    Args:
        input_path: Path to input NetCDF file (without .nc extension)
        output_path: Base path for output GeoTIFF files
        
    Raises:
        ImportError: If GDAL is not available
        ValueError: If the NetCDF file structure is unexpected
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for GeoTIFF conversion")
    
    netcdf_path = f"{input_path}.nc"
    
    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")
    
    try:
        with netCDF4.Dataset(netcdf_path, 'r') as nc:
            spatial_ref_wkt = nc.variables['spatial_ref'].spatial_ref
            
            excluded_vars = set(nc.dimensions.keys()) | {'NDVI', 'spatial_ref'}
            band_names = [var for var in nc.variables if var not in excluded_vars]
            
            if not band_names:
                raise ValueError("No valid band variables found in NetCDF file")
            
            data_shape = nc[band_names[0]][:].shape
            
            if len(data_shape) < 3:
                raise ValueError("Expected at least 3D data (time, height, width)")
            
            for t_index in range(data_shape[0]):
                output_filename = f"{output_path}_{t_index + 1:02d}.tif"
                
                driver = gdal.GetDriverByName('GTiff')
                output_dataset = driver.Create(
                    output_filename, 
                    data_shape[2],
                    data_shape[1],
                    len(band_names),
                    gdal.GDT_Float32
                )
                
                srs = osr.SpatialReference()
                srs.ImportFromWkt(spatial_ref_wkt)
                output_dataset.SetProjection(srs.ExportToWkt())
                
                output_dataset.SetGeoTransform([0, 1, 0, data_shape[1], 0, -1])
                
                for band_index, band_name in enumerate(band_names):
                    band_data = nc[band_name][t_index, :, :]
                    output_dataset.GetRasterBand(band_index + 1).WriteArray(band_data)
                
                output_dataset = None
                
                print(f"Created GeoTIFF: {output_filename}")
                
    except Exception as e:
        raise RuntimeError(f"Failed to convert NetCDF to GeoTIFF: {e}")


def netcdf_to_npz(input_path: str, output_path: str) -> None:
    """
    Convert NetCDF file to compressed NumPy format (.npz).
    
    This function extracts data from NetCDF and saves it as a compressed
    NumPy array file, which can be faster to load for training.
    
    Args:
        input_path: Path to input NetCDF file (without .nc extension)
        output_path: Path for output .npz file
        
    Raises:
        FileNotFoundError: If the NetCDF file doesn't exist
        ValueError: If the NetCDF file structure is unexpected
    """
    netcdf_path = f"{input_path}.nc"
    
    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")
    
    try:
        with netCDF4.Dataset(netcdf_path, 'r') as nc:
            excluded_vars = set(nc.dimensions.keys()) | {'NDVI', 'spatial_ref'}
            band_names = [var for var in nc.variables if var not in excluded_vars]
            
            if not band_names:
                raise ValueError("No valid band variables found in NetCDF file")
            
            data_shape = nc[band_names[0]][:].shape
            
            if len(data_shape) < 3:
                raise ValueError("Expected at least 3D data (time, height, width)")
            
            data_arrays = []
            for t_index in range(data_shape[0]):
                time_step_data = []
                for band_name in band_names:
                    band_data = nc[band_name][t_index, :, :]
                    time_step_data.append(band_data)
                data_arrays.append(time_step_data)
            
            data_arrays = np.array(data_arrays, dtype=np.float32)
            
            np.savez_compressed(output_path, data=data_arrays, band_names=band_names)
            print(f"Created NPZ file: {output_path}.npz")
            
    except Exception as e:
        raise RuntimeError(f"Failed to convert NetCDF to NPZ: {e}")


def convert_to_geotiff(pattern: str, force_overwrite: bool = False) -> None:
    """
    Batch convert NetCDF files to GeoTIFF format.
    
    This function finds all NetCDF files matching the given pattern and
    converts them to GeoTIFF format.
    
    Args:
        pattern: Glob pattern to find NetCDF files
        force_overwrite: Whether to overwrite existing GeoTIFF files
        
    Raises:
        ImportError: If GDAL is not available
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for GeoTIFF conversion")
    
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        print(f"No files found matching pattern: {pattern}")
        return
    
    converted_count = 0
    error_count = 0
    
    for file_path in file_paths:
        if not file_path.endswith('.nc'):
            continue
            
        base_path = file_path[:-3]
        
        test_output = f"{base_path}_01.tif"
        if os.path.isfile(test_output) and not force_overwrite:
            print(f"Skipping {file_path} (output already exists)")
            continue
        
        try:
            netcdf_to_geotiff(base_path, base_path)
            converted_count += 1
            print(f"Successfully converted: {file_path}")
        except Exception as e:
            error_count += 1
            print(f"Failed to convert {file_path}: {e}")
    
    print(f"\nConversion complete. Success: {converted_count}, Errors: {error_count}")


def convert_to_npz(pattern: str, force_overwrite: bool = False) -> None:
    """
    Batch convert NetCDF files to NPZ format.
    
    This function finds all NetCDF files matching the given pattern and
    converts them to compressed NumPy format.
    
    Args:
        pattern: Glob pattern to find NetCDF files
        force_overwrite: Whether to overwrite existing NPZ files
    """
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        print(f"No files found matching pattern: {pattern}")
        return
    
    converted_count = 0
    error_count = 0
    
    for file_path in file_paths:
        if not file_path.endswith('.nc'):
            continue
            
        base_path = file_path[:-3]
        output_path = f"{base_path}.npz"
        
        if os.path.isfile(output_path) and not force_overwrite:
            print(f"Skipping {file_path} (output already exists)")
            continue
        
        try:
            netcdf_to_npz(base_path, base_path)
            converted_count += 1
            print(f"Successfully converted: {file_path}")
        except Exception as e:
            error_count += 1
            print(f"Failed to convert {file_path}: {e}")
    
    print(f"\nConversion complete. Success: {converted_count}, Errors: {error_count}")


def validate_netcdf_structure(file_path: str) -> dict:
    """
    Validate and inspect NetCDF file structure.
    
    Args:
        file_path: Path to NetCDF file
        
    Returns:
        Dictionary containing file structure information
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")
    
    info = {
        'dimensions': {},
        'variables': {},
        'attributes': {},
        'spatial_ref': None,
        'band_variables': [],
        'time_steps': 0,
        'spatial_shape': None
    }
    
    try:
        with netCDF4.Dataset(file_path, 'r') as nc:
            info['dimensions'] = {name: len(dim) for name, dim in nc.dimensions.items()}
            
            info['attributes'] = {attr: getattr(nc, attr) for attr in nc.ncattrs()}
            
            for var_name, var in nc.variables.items():
                info['variables'][var_name] = {
                    'dimensions': var.dimensions,
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'attributes': {attr: getattr(var, attr) for attr in var.ncattrs()}
                }
            
            if 'spatial_ref' in nc.variables:
                try:
                    info['spatial_ref'] = nc.variables['spatial_ref'].spatial_ref
                except AttributeError:
                    info['spatial_ref'] = "Available but couldn't extract"
            
            excluded_vars = set(nc.dimensions.keys()) | {'NDVI', 'spatial_ref'}
            info['band_variables'] = [var for var in nc.variables if var not in excluded_vars]
            
            if info['band_variables']:
                first_band = nc[info['band_variables'][0]]
                if len(first_band.shape) >= 3:
                    info['time_steps'] = first_band.shape[0]
                    info['spatial_shape'] = first_band.shape[1:]
                elif len(first_band.shape) == 2:
                    info['time_steps'] = 1
                    info['spatial_shape'] = first_band.shape
                    
    except Exception as e:
        raise RuntimeError(f"Failed to validate NetCDF structure: {e}")
    
    return info


def create_data_inventory(
    root_directory: str, 
    file_extensions: List[str] = None,
    output_file: Optional[str] = None
) -> dict:
    """
    Create an inventory of data files in a directory structure.
    
    Args:
        root_directory: Root directory to scan
        file_extensions: List of file extensions to include (e.g., ['.nc', '.tif'])
        output_file: Optional path to save inventory as JSON
        
    Returns:
        Dictionary containing file inventory
    """
    if file_extensions is None:
        file_extensions = ['.nc', '.tif', '.npz']
    
    inventory = {
        'root_directory': root_directory,
        'scan_date': str(np.datetime64('now')),
        'file_extensions': file_extensions,
        'files': [],
        'summary': {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_extension': {}
        }
    }
    
    root_path = pathlib.Path(root_directory)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_directory}")
    
    for ext in file_extensions:
        inventory['summary']['by_extension'][ext] = {'count': 0, 'size_bytes': 0}
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            try:
                file_size = file_path.stat().st_size
                relative_path = file_path.relative_to(root_path)
                
                file_info = {
                    'path': str(relative_path),
                    'full_path': str(file_path),
                    'size_bytes': file_size,
                    'extension': file_path.suffix.lower(),
                    'parent_directory': str(relative_path.parent)
                }
                
                inventory['files'].append(file_info)
                inventory['summary']['total_files'] += 1
                inventory['summary']['total_size_bytes'] += file_size
                
                ext = file_path.suffix.lower()
                if ext in inventory['summary']['by_extension']:
                    inventory['summary']['by_extension'][ext]['count'] += 1
                    inventory['summary']['by_extension'][ext]['size_bytes'] += file_size
                    
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access file {file_path}: {e}")
    
    if output_file:
        import json
        try:
            with open(output_file, 'w') as f:
                json.dump(inventory, f, indent=2)
            print(f"Inventory saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Could not save inventory to {output_file}: {e}")
    
    return inventory


def print_data_summary(inventory: dict) -> None:
    """
    Print a human-readable summary of data inventory.
    
    Args:
        inventory: Inventory dictionary from create_data_inventory
    """
    summary = inventory['summary']
    
    print(f"\nData Inventory Summary")
    print(f"======================")
    print(f"Root Directory: {inventory['root_directory']}")
    print(f"Scan Date: {inventory['scan_date']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Size: {summary['total_size_bytes'] / (1024**3):.2f} GB")
    
    print(f"\nBy File Extension:")
    print(f"-----------------")
    for ext, stats in summary['by_extension'].items():
        if stats['count'] > 0:
            size_mb = stats['size_bytes'] / (1024**2)
            print(f"{ext}: {stats['count']} files, {size_mb:.1f} MB")
    
    if inventory['files']:
        directories = set()
        for file_info in inventory['files']:
            directories.add(file_info['parent_directory'])
        
        print(f"\nDirectories with data:")
        print(f"---------------------")
        for directory in sorted(directories):
            file_count = sum(1 for f in inventory['files'] 
                           if f['parent_directory'] == directory)
            print(f"{directory}: {file_count} files")


def normalize_satellite_bands(
    data: np.ndarray, 
    method: str = 'percentile',
    percentiles: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Normalize satellite imagery bands.
    
    Args:
        data: Input data array
        method: Normalization method ('percentile', 'minmax', 'zscore')
        percentiles: Percentile values for clipping (only used with 'percentile')
        
    Returns:
        Normalized data array
    """
    if method == 'percentile':
        p_low, p_high = np.percentile(data, percentiles)
        data_clipped = np.clip(data, p_low, p_high)
        return (data_clipped - p_low) / (p_high - p_low)
    
    elif method == 'minmax':
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_data_integrity(file_path: str) -> dict:
    """
    Check data file integrity and return statistics.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        'file_exists': False,
        'file_size': 0,
        'readable': False,
        'format_valid': False,
        'has_nan_values': False,
        'has_infinite_values': False,
        'data_range': None,
        'errors': []
    }
    
    try:
        if os.path.exists(file_path):
            results['file_exists'] = True
            results['file_size'] = os.path.getsize(file_path)
        else:
            results['errors'].append("File does not exist")
            return results
        
        file_ext = pathlib.Path(file_path).suffix.lower()
        
        if file_ext == '.nc':
            with netCDF4.Dataset(file_path, 'r') as nc:
                results['readable'] = True
                results['format_valid'] = True
                
                var_names = list(nc.variables.keys())
                if var_names:
                    first_var = nc[var_names[0]][:]
                    results['has_nan_values'] = np.isnan(first_var).any()
                    results['has_infinite_values'] = np.isinf(first_var).any()
                    results['data_range'] = (float(first_var.min()), float(first_var.max()))
                    
        elif file_ext == '.npz':
            data = np.load(file_path)
            results['readable'] = True
            results['format_valid'] = True
            
            array_names = list(data.keys())
            if array_names:
                first_array = data[array_names[0]]
                results['has_nan_values'] = np.isnan(first_array).any()
                results['has_infinite_values'] = np.isinf(first_array).any()
                results['data_range'] = (float(first_array.min()), float(first_array.max()))
        
        elif file_ext in ['.tif', '.tiff']:
            if GDAL_AVAILABLE:
                dataset = gdal.Open(file_path)
                if dataset:
                    results['readable'] = True
                    results['format_valid'] = True
                    
                    # Read first band
                    band = dataset.GetRasterBand(1)
                    array = band.ReadAsArray()
                    if array is not None:
                        results['has_nan_values'] = np.isnan(array).any()
                        results['has_infinite_values'] = np.isinf(array).any()
                        results['data_range'] = (float(array.min()), float(array.max()))
                else:
                    results['errors'].append("Could not open with GDAL")
            else:
                results['errors'].append("GDAL not available for TIFF checking")
        
        else:
            results['errors'].append(f"Unsupported file format: {file_ext}")
            
    except Exception as e:
        results['errors'].append(f"Error during integrity check: {str(e)}")
    
    return results