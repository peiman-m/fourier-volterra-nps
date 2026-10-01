import gc
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# Import multiprocessing with safety check
try:
    from multiprocessing import Pool
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    print("[WARNING] Multiprocessing not available, using sequential processing")

try:
    import cdsapi
except ImportError:
    cdsapi = None


class ERA5DataProcessor:
    """ERA5 reanalysis dataset loader with attribute-based data splitting and normalization.

    This class provides a comprehensive interface for loading, processing, and splitting ERA5
    reanalysis data for machine learning applications. It supports both automatic downloading
    from the Copernicus Climate Data Store (CDS) and loading from local files.

    IMPORTANT - Longitude Format Standardization:
    This dataset automatically converts all longitude coordinates to the [-180, 180] format
    during initialization, regardless of the original data format. If your data uses [0, 360]
    format, it will be automatically converted. All subsequent operations (subset ranges,
    normalization, filtering) expect and use the [-180, 180] format. Ensure that any
    longitude ranges you specify in subset_coordinate_ranges follow this standard format.

    Key Features:
    - Automatic data downloading via CDS API
    - Flexible temporal and spatial filtering
    - Attribute-based train/validation/test splits (geographic/temporal coherence)
    - Data normalization using train subset statistics
    - Support for multiple coordinate systems and time formats
    - Automatic surface elevation computation from geopotential data
    - Processed dataset subset caching for faster repeated access

    The dataset supports splitting based on ranges of attributes like latitude, longitude,
    and time, allowing for geographically or temporally coherent splits rather than
    traditional random splits. This is particularly useful for evaluating model
    generalization across space and time.

    Attributes:
        dataset (xarray.Dataset): The processed ERA5 dataset for the requested subset.
        data_variables (list[str]): List of loaded data variables.
        subset (str): Current subset ('train', 'validation', or 'test').
        numerical_time_unit (str): Time unit for numerical_time coordinate.
        datetime_reference_time (str): Reference time for numerical_time coordinate.

    Raises:
        ValueError: If invalid parameters are provided (subset,
                   numerical_time_unit, coordinate ranges, etc.).
        FileNotFoundError: If required data files are missing and download=False.
        ImportError: If cdsapi package is missing when download=True.
        RuntimeError: If download fails or CDS API credentials are invalid.

    Examples:
        Basic usage with automatic download:

        >>> # Download and load temperature and geopotential data
        >>> dataset = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020, 2021],
        ...     data_variables=["2m_temperature", "geopotential"],
        ...     download=True
        ... )
        >>> print(dataset.dataset)
        <xarray.Dataset>
        Dimensions:  (time: 17544, latitude: 721, longitude: 1440)
        Coordinates:
          * latitude       (latitude) float32 90.0 89.75 89.5 ... -89.5 -89.75 -90.0
          * longitude      (longitude) float32 0.0 0.25 0.5 0.75 ... 359.25 359.5 359.75
          * time           (time) datetime64[ns] 2020-01-01 ... 2021-12-31T23:00:00
            numerical_time (time) float64 0.0 1.0 2.0 3.0 ... 17541.0 17542.0 17543.0
            surface_elevation (latitude, longitude) float32 ...
        Data variables:
            2m_temperature (time, latitude, longitude) float32 ...
            z              (time, latitude, longitude) float32 ...

        Geographic splitting for spatial generalization:

        >>> # Define train/validation/test splits by latitude ranges
        >>> subset_ranges = {
        ...     'train': {'latitude': (60, 90)},      # Arctic
        ...     'validation': {'latitude': (30, 60)}, # Mid-latitudes
        ...     'test': {'latitude': (-30, 30)}       # Tropics
        ... }
        >>>
        >>> # Load training data
        >>> train_data = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020, 2021],
        ...     subset='train',
        ...     subset_coordinate_ranges=subset_ranges,
        ...     normalize=True
        ... )
        >>>
        >>> # Load validation data (normalized using train statistics)
        >>> val_data = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020, 2021],
        ...     subset='validation',
        ...     subset_coordinate_ranges=subset_ranges,
        ...     normalize=True  # Uses cached train statistics
        ... )

        Temporal splitting for time series forecasting:

        >>> # Define temporal splits
        >>> temporal_ranges = {
        ...     'train': {'time': ('2020-01-01', '2020-08-31')},
        ...     'validation': {'time': ('2020-09-01', '2020-10-31')},
        ...     'test': {'time': ('2020-11-01', '2020-12-31')}
        ... }
        >>>
        >>> # Load test data for forecasting evaluation
        >>> test_data = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020],
        ...     subset='test',
        ...     subset_coordinate_ranges=temporal_ranges
        ... )

        Regional analysis with specific variables and temporal filtering:

        >>> # Load European region data with specific temporal sampling
        >>> europe_data = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2019, 2020, 2021],
        ...     months=[6, 7, 8],  # Summer months only
        ...     hours=[0, 6, 12, 18],  # 6-hourly sampling
        ...     data_variables=["2m_temperature", "total_precipitation", "10m_u_component_of_wind"],
        ...     subset_coordinate_ranges={
        ...         'train': {
        ...             'latitude': (35, 70),
        ...             'longitude': (-10, 30)
        ...         }
        ...     },
        ...     download=True
        ... )

        Working with numerical time coordinates:

        >>> # Use days as time unit with custom reference time
        >>> dataset = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020],
        ...     numerical_time_unit='days',
        ...     datetime_reference_time='2020-06-01T00:00:00',
        ...     subset_coordinate_ranges={
        ...         'train': {'numerical_time': (0, 90)},    # First 90 days from June 1st
        ...         'test': {'numerical_time': (91, 180)}    # Next 90 days
        ...     }
        ... )
        >>>
        >>> # Access both time formats
        >>> print(dataset.dataset.time.values[:5])        # Absolute datetimes
        >>> print(dataset.dataset.numerical_time.values[:5])  # Days since reference

        Advanced usage with multiple coordinate filters:

        >>> # Complex filtering combining spatial, temporal, and elevation constraints
        >>> mountain_summer = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2015, 2016, 2017, 2018, 2019, 2020],
        ...     months=[6, 7, 8],
        ...     data_variables=["2m_temperature", "geopotential", "total_precipitation"],
        ...     subset_coordinate_ranges={
        ...         'train': {
        ...             'latitude': (45, 47),        # Alps region
        ...             'longitude': (6, 10),
        ...             'time': ('2015-01-01', '2018-12-31'),
        ...             'surface_elevation': (1000, 4000)  # Mountain elevations only
        ...         },
        ...         'test': {
        ...             'latitude': (45, 47),
        ...             'longitude': (6, 10),
        ...             'time': ('2019-01-01', '2020-12-31'),
        ...             'surface_elevation': (1000, 4000)
        ...         }
        ...     },
        ...     normalize=True,
        ...     cache_stats=True
        ... )

        Using dataset caching to avoid redundant processing:

        >>> # First time - processes and caches the dataset subset
        >>> dataset = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020, 2021],
        ...     subset='train',
        ...     subset_coordinate_ranges=subset_ranges,
        ...     cache_dataset=True,  # Enable dataset caching
        ...     dataset_cache_dirpath="/path/to/cache"  # Optional custom cache dir
        ... )
        >>>
        >>> # Subsequent loads - uses cached version (much faster)
        >>> dataset_cached = ERA5DataProcessor(
        ...     dirpath="/path/to/era5/data",
        ...     years=[2020, 2021],
        ...     subset='train', 
        ...     subset_coordinate_ranges=subset_ranges,
        ...     cache_dataset=True
        ... )
        >>>
        >>> # Clear caches when needed
        >>> dataset.clear_cache("dataset")  # Clear only dataset caches
        >>> dataset.clear_cache("all")      # Clear both dataset and stats caches

    Usage for Training:
        This processor can be used standalone for data exploration and analysis.
        For training with PyTorch DataLoader, use ERA5Dataset which provides
        efficient random sampling and batching:

        >>> # For exploration and analysis:
        >>> processor = ERA5DataProcessor(dirpath='./data', years=[2020], subset='train')
        >>> print(processor.dataset)  # Access xarray dataset directly
        >>>
        >>> # For training (recommended):
        >>> processor = ERA5DataProcessor(dirpath='./data', years=[2020], subset='train')
        >>> dataset = ERA5Dataset(
        ...     dataset=processor,
        ...     grid_size_range=(16, 16),  # or ((12, 20), (12, 20)) for variable size
        ...     max_p_nan=0.1,
        ...     min_pc=0.1,
        ...     max_pc=0.5,
        ... )
        >>> loader = DataLoader(dataset, batch_size=8, num_workers=4)

    Notes:
        - Data files are cached locally and organized by month and variables
        - Processed dataset subsets can be cached to disk for faster repeated access
        - Normalization statistics are computed only from the train subset
        - Surface elevation is automatically derived from geopotential data
        - Time coordinates support both absolute datetime and numerical formats
        - Coordinate ranges are inclusive on both ends
        - Missing data files trigger automatic download if download=True
        - CDS API credentials must be configured for downloading
        - ERA5Dataset is an IterableDataset that generates samples on-the-fly
        - ERA5Dataset uses numpy mmap files (not xarray) at runtime, so
          num_workers>0 is safe — each worker lazily opens its own mmap handle
    """

    def __init__(
        self,
        *,
        dirpath: str,
        years: list[int],
        months: list[int] | None = None,
        days: list[int] | None = None,
        hours: list[int] | None = None,
        download: bool = False,
        num_processes: int = 1,
        subset: str = "train",
        subset_coordinate_ranges: (
            dict[str, dict[str, tuple[float, float] | tuple[str, str]]] | None
        ) = None,
        data_variables: list[str] = ["2m_temperature", "geopotential"],
        numerical_time_unit: str = "hours",
        datetime_reference_time: str | None = None,
        normalize: bool = True,
        normalize_exclude_coords: list[str] | None = None,
        cache_stats: bool = True,
        stats_cache_dirpath: str | None = None,
        cache_dataset: bool = False,
        dataset_cache_dirpath: str | None = None,
        drop_geo_z_variable: bool = True,
        chunks: dict | str | None = None,
    ):
        """Initialize ERA5DataProcessor.

        Args:
            dirpath (str): Directory path for storing/loading ERA5 data files.
            years (list[int]): List of years to include for data coverage.
            days (list[int], optional): List of days of month to include (1-31).
                If None, includes all days.
                NOTE: Day values must be in range 1-31 (1-based indexing, not 0-based).
            months (list[int], optional): List of months to include (1-12).
                If None, includes all months.
                NOTE: Month values must be in range 1-12 (1-based indexing, not 0-based).
            hours (list[int], optional): List of hours to include (0-23).
                If None, includes all hours.
            download (bool): If True, automatically download missing files from CDS.
                Requires cdsapi package and valid CDS credentials. Default: False.
            num_processes (int): Number of parallel processes for downloading.
                Default: 1.
            subset (str): Which data subset to return ('train', 'validation', 'test').
                Default: 'train'.
            subset_coordinate_ranges (dict, optional): Dictionary defining coordinate
                ranges for each subset. Format:
                {
                    'train': {'coord_name': (lo, hi), ...},
                    'validation': {'coord_name': (lo, hi), ...},
                    'test': {'coord_name': (lo, hi), ...}
                }
                If None, all subsets use the entire dataset.

                CAUTION: Longitude ranges should be specified in [-180,180] format.
                Use ranges like (-180, 180). Cross-dateline ranges are supported.
            data_variables (list[str]): List of ERA5 variables to load.
                Default: ['2m_temperature', 'geopotential'].
            numerical_time_unit (str): Time unit for numerical_time coordinate.
                Options: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'.
                Default: 'hours'.
            datetime_reference_time (str, optional): ISO 8601 datetime string for
                numerical_time reference. If None, uses "{min(years)}-01-01T00:00:00".

                IMPORTANT NOTE: This reference time ONLY affects the numerical_time coordinate
                values. The original 'time' coordinate remains as absolute datetime values.
                When filtering by time ranges:
                - For 'time' coordinate: Use absolute datetime strings (e.g., "2020-01-01")
                - For 'numerical_time' coordinate: Use either datetime strings (which will be
                  converted relative to reference_time) or numerical values (in the specified
                  time unit relative to reference_time)
            normalize (bool): If True, normalize coordinates and variables using
                train subset statistics. Default: False.
            cache_stats (bool): If True and normalize=True, cache normalization
                statistics to disk for faster loading. Default: True.
            stats_cache_dirpath (str, optional): Directory path for storing/loading
                normalization statistics cache files. If None, uses the same directory
                as data files (dirpath). Default: None.
            cache_dataset (bool): If True, cache processed dataset subsets to disk
                for faster loading. Default: False.
            dataset_cache_dirpath (str, optional): Directory path for storing/loading
                processed dataset cache files. If None, uses the same directory
                as data files (dirpath). Default: None.
            drop_geo_z_variable (bool): If True, drop the 'z' (geopotential) variable
                after creating surface_elevation coordinates. Default: True.
            chunks (dict | str | None): Chunk sizes for dask arrays when loading data.
        """
        # --- Validate basics
        if subset not in ("train", "validation", "test"):
            raise ValueError("subset must be one of 'train', 'validation', 'test'")

        valid_time_units = (
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
        )
        if numerical_time_unit not in valid_time_units:
            raise ValueError(f"numerical_time_unit must be one of {valid_time_units}")

        if (
            not years
            or not isinstance(years, list)
            or not all(isinstance(y, int) for y in years)
        ):
            raise ValueError("years must be a non-empty list of integers")

        # Validate time components
        time_validations = [
            (months, "months", 1, 12, "1..12"),
            (days, "days", 1, 31, "1..31"),
            (hours, "hours", 0, 23, "0..23")
        ]
        
        for values, name, min_val, max_val, range_desc in time_validations:
            if values is not None:
                if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
                    raise ValueError(f"{name} must be a list of integers")
                bad = [v for v in values if not (min_val <= v <= max_val)]
                if bad:
                    raise ValueError(f"{name} must be in {range_desc}. Invalid: {bad}")

        # Validate longitudes in user-provided split ranges
        if subset_coordinate_ranges:
            for name, coord_ranges in subset_coordinate_ranges.items():
                if "longitude" in coord_ranges:
                    lo, hi = coord_ranges["longitude"]
                    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                        # Longitude ranges should be in [-180,180] format
                        if (lo < -180 or hi > 180) and not (
                            lo > hi
                        ):  # Allow cross-dateline
                            raise ValueError(
                                f"[{name}] longitude range must lie in [-180,180] "
                                "or be a valid cross-dateline range."
                            )

        # --- Store config (remove duplicates and sort)
        def clean_and_sort(values):
            return sorted(list(set(values))) if values else None
        
        self.data_variables_requested = sorted(list(set(data_variables)))
        self.years_requested = sorted(list(set(years)))
        self.months_requested = clean_and_sort(months)
        self.days_requested = clean_and_sort(days)
        self.hours_requested = clean_and_sort(hours)

        # Store configuration
        self.enable_download = download
        self.download_processes = num_processes if MULTIPROCESSING_AVAILABLE else 1
        self.chunks = chunks
        
        # Subset and normalization settings
        self.subset = subset
        self.subset_coordinate_ranges = subset_coordinate_ranges
        self.normalize = normalize
        # Coordinates to leave un-normalized even when normalize=True (e.g.
        # ["numerical_time"] to feed raw time while lat/lon stay normalized).
        self.normalize_exclude_coords = (
            list(normalize_exclude_coords) if normalize_exclude_coords else []
        )
        self.cache_stats = cache_stats
        self.cache_dataset = cache_dataset
        
        # Time processing settings
        self.numerical_time_unit = numerical_time_unit
        self.datetime_reference_time = (
            f"{min(self.years_requested)}-01-01T00:00:00"
            if datetime_reference_time is None
            else datetime_reference_time
        )
        
        # Dataset processing settings
        self.drop_geo_z_variable = drop_geo_z_variable

        # Set up directories
        self.data_directory = Path(dirpath)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        self.stats_cache_directory = (
            Path(stats_cache_dirpath) if stats_cache_dirpath is not None
            else self.data_directory
        )
        self.stats_cache_directory.mkdir(parents=True, exist_ok=True)
        
        self.dataset_cache_directory = (
            Path(dataset_cache_dirpath) if dataset_cache_dirpath is not None
            else self.data_directory
        )
        self.dataset_cache_directory.mkdir(parents=True, exist_ok=True)

        # Check and handle missing files
        available_files, missing = self._check_data_files_exist()

        if missing:
            if self.enable_download:
                self._handle_missing_files_download(missing)
                available_files, missing2 = self._check_data_files_exist()
                if missing2:
                    self._raise_missing_files_error(missing2)
            else:
                self._raise_missing_files_error(missing)

        # Try to load cached dataset first
        cached_data = None
        if self.cache_dataset:
            cached_data = self._load_cached_dataset()

        if cached_data is not None:
            self.data = cached_data
        else:
            # Load, process and subset dataset
            ds = self._load_dataset_files(available_files)
            ds = self._process_dataset(ds)
            gc.collect()
            
            self.data = self._apply_subset_filter(ds)
            gc.collect()
            
            # Save to cache if enabled
            if self.cache_dataset:
                self._save_cached_dataset(self.data)

        # Apply normalization if requested
        if self.normalize:
            self._apply_normalization()

        # Count NaNs in data variables. These are xarray scalars (untyped);
        # annotate as Any so the `.values` accessor below typechecks.
        var_nans: Any = sum(
            np.isnan(self.data[var]).sum().compute() for var in self.data.data_vars
        )

        # Count NaNs in coordinates
        coord_nans: Any = 0
        for coord_name in self.data.coords:
            coord_data = self.data.coords[coord_name]
            if np.issubdtype(coord_data.dtype, np.floating):
                coord_nans += np.isnan(coord_data).sum().compute()

        # Calculate dataset statistics
        total_elements = np.prod(list(self.data.sizes.values()))
        total_nans = var_nans.values + coord_nans.values
        nan_percentage = 100 * total_nans / total_elements
        
        print(f"[INFO] ERA5DataProcessor '{self.subset}' subset successfully initialized")
        print(f"[INFO] Dataset shape: {dict(self.data.sizes)}")
        print(f"[INFO] Data variables: {list(self.data.data_vars)}")
        print(f"[INFO] NaN statistics:")
        print(f"  • Variables: {var_nans.values:,} NaNs")
        print(f"  • Coordinates: {coord_nans.values:,} NaNs")
        print(f"  • Total: {total_nans:,} / {total_elements:,} ({nan_percentage:.3f}%)")
        if self.normalize:
            print(f"[INFO] Data normalized using train subset statistics")

    # ---------------------- Processing helpers ----------------------

    def _load_dataset_files(self, available_files: list[str]) -> xr.Dataset:
        """Load and combine dataset files with fallback strategies."""
        filepaths = [str(self.data_directory / fn) for fn in sorted(available_files)]
        chunking_mode = "with chunks" if self.chunks else "without chunks"
        
        print(f"[INFO] Loading {len(filepaths)} ERA5 files ({chunking_mode})")
        if len(filepaths) > 1:
            print(f"[INFO] Time range: {available_files[0][:7]} to {available_files[-1][:7]}")

        safe_chunks = self.chunks or None
        
        try:
            print("[INFO] Attempting combined loading with open_mfdataset")
            ds = xr.open_mfdataset(
                filepaths,
                combine="by_coords",
                parallel=False,
                chunks=safe_chunks,
                engine="netcdf4",
                decode_times=True,
                use_cftime=False,
            )
            print("[INFO] Successfully loaded dataset files")
            return ds
        except Exception as e:
            print(f"[WARNING] Combined loading failed: {str(e)[:100]}...")
            print("[INFO] Falling back to manual concatenation with chunks")
            try:
                parts = [xr.open_dataset(fp, chunks=safe_chunks, engine="netcdf4") for fp in filepaths]
                ds = xr.concat(parts, dim="time", join="override", combine_attrs="override")
                print("[INFO] Successfully loaded dataset files using manual concatenation")
                return ds
            except Exception as e2:
                print(f"[WARNING] Chunked concatenation failed: {str(e2)[:100]}...")
                print("[INFO] Final fallback: loading without chunks")
                parts = [xr.open_dataset(fp, engine="netcdf4") for fp in filepaths]
                ds = xr.concat(parts, dim="time", join="override", combine_attrs="override")
                print("[INFO] Successfully loaded dataset files without chunks")
                return ds

    def _process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Process the dataset according to requirements."""
        # Drop singleton 'time' dimension if it exists, keeping 'valid_time'
        if "time" in ds.dims and ds.sizes["time"] == 1 and "valid_time" in ds.dims:
            ds = ds.squeeze("time", drop=True)

        # Normalize coordinate names
        ren = {}
        if "lat" in ds.coords:
            ren["lat"] = "latitude"
        if "lon" in ds.coords:
            ren["lon"] = "longitude"
        if "valid_time" in ds.coords:
            ren["valid_time"] = "time"
        if ren:
            ds = ds.rename(ren)

        # Always convert longitudes to [-180,180] format
        if "longitude" in ds.coords:
            lon = ds.coords["longitude"]

            # Check if longitudes are already in [-180, 180] range
            lon_min, lon_max = float(lon.min()), float(lon.max())

            if lon_min >= -180 and lon_max <= 180:
                # Already in correct format, no conversion needed
                pass
            elif lon_min >= 0 and lon_max <= 360:
                # Data is in [0,360] format, convert to [-180,180]
                lon_wrapped = xr.where(lon > 180, lon - 360, lon)
                ds = ds.assign_coords(longitude=lon_wrapped)
                print(
                    "[INFO] Converted longitude coordinates from [0,360] to [-180,180] format"
                )
            else:
                raise ValueError(
                    f"Longitude values ({lon_min:.2f}, {lon_max:.2f}) are outside "
                    "expected ranges. Expected either [0,360] or [-180,180] format."
                )

            ds = ds.sortby("longitude")

        # Derive surface_elevation from geopotential 'z' if present
        if "z" in ds.data_vars:
            # ERA5 geopotential is in m²/s², divide by 9.80665 to get meters
            g = 9.80665
            surface_elevation = ds["z"] / g

            # Surface elevation is topography and should be time-invariant
            # If z has time dimension, take the mean across time
            # (it should be constant anyway)
            if "time" in surface_elevation.dims:
                surface_elevation = surface_elevation.mean(dim="time")

            # Create surface_elevation coordinate with only spatial dimensions
            ds = ds.assign_coords(
                surface_elevation=(["latitude", "longitude"], surface_elevation.values)
            )
            ds.coords["surface_elevation"].attrs.update(
                units="m",
                long_name="Surface elevation derived from geopotential",
                standard_name="surface_altitude",
            )

            if self.drop_geo_z_variable:
                ds = ds.drop_vars("z")

        # Add numerical_time coordinate based on time coordinate
        if "time" in ds.coords:
            numerical_time = self._convert_time_to_numerical(ds.coords["time"])
            ds = ds.assign_coords(numerical_time=("time", numerical_time))
            ds.coords["numerical_time"].attrs.update(
                units=self.numerical_time_unit,
                long_name=f"Numerical time in {self.numerical_time_unit}",
                reference_time=self.datetime_reference_time,
            )

        # Map requested variables to dataset names for later normalization
        self._dataset_var_names = self._map_requested_to_dataset_vars(
            list(ds.data_vars)
        )

        return ds

    # ---------------------- Utilities ----------------------

    def _map_requested_to_dataset_vars(self, present_vars: list[str]) -> dict[str, str]:
        """Map user-facing ERA5 variable names to dataset internal names.

        Returns a dict like {"2m_temperature": "t2m", "geopotential": "z"} if present.
        """
        # Common ERA5 single-levels mapping (extend as needed)
        known = {
            "2m_temperature": "t2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "total_precipitation": "tp",
            "geopotential": "z",
        }
        out: dict[str, str] = {}
        for req in self.data_variables_requested:
            cand = known.get(req, req)
            if cand in present_vars:
                out[req] = cand
            elif req in present_vars:
                out[req] = req
            # else: silently ignore missing; download step should ensure presence
        return out

    def _convert_time_to_numerical(self, time_coord) -> list[float]:
        """Convert datetimes to numerical time since reference_time."""
        vals = pd.to_datetime(
            time_coord.values if hasattr(time_coord, "values") else time_coord
        )
        ref = pd.to_datetime(self.datetime_reference_time)
        # Ensure both tz-naive or both tz-aware
        if vals.tz is not None and ref.tz is None:
            ref = ref.tz_localize(vals.tz)
        elif vals.tz is None and ref.tz is not None:
            vals = vals.tz_localize(ref.tz)

        delta = vals - ref

        unit_map = {
            "seconds": np.timedelta64(1, "s"),
            "minutes": np.timedelta64(1, "m"),
            "hours": np.timedelta64(1, "h"),
            "days": np.timedelta64(1, "D"),
            "weeks": np.timedelta64(1, "W"),
        }
        if self.numerical_time_unit in unit_map:
            denom = unit_map[self.numerical_time_unit]
            arr = (delta.to_numpy() / denom).astype(float)
        elif self.numerical_time_unit == "months":
            arr = (delta.to_numpy() / np.timedelta64(1, "D")) / 30.44
        elif self.numerical_time_unit == "years":
            arr = (delta.to_numpy() / np.timedelta64(1, "D")) / 365.25
        else:
            raise ValueError(f"Unsupported time unit: {self.numerical_time_unit}")

        return arr.tolist()

    # Prefer sel/slice for numeric coords (fast, keeps indexes)
    def _slice_on_numeric_coord(
        self, ds: xr.Dataset, name: str, lo: float, hi: float
    ) -> xr.Dataset:
        if name not in ds.coords:
            raise ValueError(f"Coordinate '{name}' not in dataset: {list(ds.coords)}")
        c = ds.coords[name]
        if not np.issubdtype(c.dtype, np.number):
            raise ValueError(f"Coordinate '{name}' is not numeric; got dtype {c.dtype}")
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        mask = (c >= a) & (c <= b)
        return ds.where(mask, drop=True)

    def _select_cyclic_longitudes(
        self, ds: xr.Dataset, lo: float, hi: float
    ) -> xr.Dataset:
        """Select a longitude window, possibly crossing the dateline."""
        if "longitude" not in ds.coords:
            raise ValueError("longitude coordinate not found")

        if lo <= hi:
            return self._slice_on_numeric_coord(ds, "longitude", lo, hi)

        # Cross-dateline selection: union of two ranges (always [-180,180] format)
        left = self._slice_on_numeric_coord(ds, "longitude", lo, 180.0)
        right = self._slice_on_numeric_coord(ds, "longitude", -180.0, hi)
        # concat along longitude then sort
        merged = xr.concat([left, right], dim="longitude")
        return merged.sortby("longitude")

    # ---------------------- Subsetting ----------------------

    def _apply_subset_filter(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply subset filtering based on subset_coordinate_ranges."""

        if not self.subset_coordinate_ranges:
            warnings.warn(
                "No subset_coordinate_ranges provided; "
                "using full dataset for all subsets.",
                UserWarning,
            )
            return ds

        if self.subset not in self.subset_coordinate_ranges:
            warnings.warn(
                f"Subset '{self.subset}' not found in subset_coordinate_ranges; "
                "returning full dataset.",
                UserWarning,
            )
            return ds

        ranges = self.subset_coordinate_ranges[self.subset]
        out = ds

        # Apply each range filter
        for coord_name, (lo, hi) in ranges.items():

            if coord_name not in out.coords:
                raise ValueError(
                    f"Coordinate '{coord_name}' not found. "
                    f"Available: {list(out.coords)}"
                )

            if coord_name == "time":
                if not (isinstance(lo, str) and isinstance(hi, str)):
                    raise ValueError(
                        f"Time filtering expects datetime strings for 'time' coordinate; "
                        f"got types {type(lo).__name__}, {type(hi).__name__}"
                    )
                t0, t1 = pd.to_datetime(lo), pd.to_datetime(hi)
                out = out.sel(time=slice(t0, t1))

            elif coord_name == "numerical_time":
                if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                    raise ValueError(
                        f"numerical_time filtering expects numeric values; "
                        f"got types {type(lo).__name__}, {type(hi).__name__}"
                    )
                if "numerical_time" not in out.coords:
                    raise ValueError(
                        f"numerical_time coordinate not found in dataset. "
                        f"Available coordinates: {list(out.coords)}"
                    )
                if abs(lo) > 0 or abs(hi) > 0:
                    warnings.warn(
                        "Filtering by numerical_time is relative to reference "
                        f"'{self.datetime_reference_time}' "
                        f"in {self.numerical_time_unit}. Ensure ranges align.",
                        UserWarning,
                    )
                out = out.sel(numerical_time=slice(min(lo, hi), max(lo, hi)))

            elif coord_name == "longitude":
                out = self._select_cyclic_longitudes(out, float(lo), float(hi))

            else:
                # numeric coord (latitude, surface_elevation, etc.)
                if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                    raise ValueError(
                        f"Coordinate '{coord_name}' requires numeric range; "
                        f"got types {type(lo)}, {type(hi)}"
                    )
                out = self._slice_on_numeric_coord(
                    out, coord_name, float(lo), float(hi)
                )

        return out

    def _check_data_files_exist(self) -> tuple[list[str], list[dict]]:
        """Check which data files exist and identify missing ones.

        Returns:
            available_files: List of filenames that exist in the data directory
            missing_file_info: List of dictionaries with info about missing files
        """
        available_files = []
        missing_file_info = []
        vars_id = "_".join(sorted(self.data_variables_requested))

        months_to_check = self.months_requested or list(range(1, 13))

        for year in self.years_requested:
            for month in months_to_check:
                filename = f"{month:02d}-{year}_{vars_id}.nc"
                file_path = self.data_directory / filename

                if file_path.exists() and file_path.is_file():
                    # Verify file is not empty and readable
                    if file_path.stat().st_size > 0:
                        available_files.append(filename)
                    else:
                        missing_file_info.append(
                            {
                                "filename": filename,
                                "year": year,
                                "month": month,
                                "reason": "empty_file",
                                "path": file_path,
                            }
                        )
                else:
                    missing_file_info.append(
                        {
                            "filename": filename,
                            "year": year,
                            "month": month,
                            "reason": "file_not_found",
                            "path": file_path,
                        }
                    )

        return available_files, missing_file_info

    def _raise_missing_files_error(self, missing: list[dict]) -> None:
        """Raise a detailed error about missing files."""
        total_missing = len(missing)
        not_found = [m for m in missing if m["reason"] == "file_not_found"]
        empties = [m for m in missing if m["reason"] == "empty_file"]

        # Create detailed error message
        error_parts = [f"[ERROR] {total_missing} data files are missing or invalid."]

        if not_found:
            sample_not_found = [m["filename"] for m in not_found[:3]]
            suffix = "..." if len(not_found) > 3 else ""
            error_parts.append(
                f"Files not found ({len(not_found)}): {sample_not_found}{suffix}"
            )

        if empties:
            sample_empty = [m["filename"] for m in empties[:3]]
            suffix = "..." if len(empties) > 3 else ""
            error_parts.append(f"Empty files ({len(empties)}): {sample_empty}{suffix}")

        error_parts += [
            "\nTo download missing files, set `download=True`.",
            f"Data directory: {self.data_directory}",
            f"Requested variables: {self.data_variables_requested}",
            f"Years: {self.years_requested}",
            f"Months: {self.months_requested or 'all'}",
        ]

        raise FileNotFoundError("\n".join(error_parts))

    def _handle_missing_files_download(self, missing: list[dict]) -> None:
        """Handle downloading of missing data files."""
        if cdsapi is None:
            raise ImportError(
                "The 'cdsapi' package is required for downloading ERA5 data. "
                "Install it with: pip install cdsapi"
            )

        # Clean zero-byte files
        for info in [m for m in missing if m["reason"] == "empty_file"]:
            try:
                info["path"].unlink()
                print(f"[INFO] Cleaned up empty file: {info['filename']}")
            except OSError as e:
                print(f"[WARNING] Failed to remove empty file {info['filename']}: {e}")

        # Set default temporal coverage if not specified
        default_days = list(range(1, 32))  # Full month coverage
        default_hours = list(range(24))  # Full day coverage

        print(f"[INFO] Starting download of {len(missing)} missing ERA5 files")
        print(f"[INFO] Variables to download: {self.data_variables_requested}")
        print(f"[INFO] Using {self.download_processes} parallel download processes")

        # Prepare download jobs
        download_jobs = [
            (
                info["year"],
                info["month"],
                self.data_directory,
                self.data_variables_requested,
                self.days_requested or default_days,
                self.hours_requested or default_hours,
            )
            for info in missing
        ]

        # Execute downloads with parallel processing when requested
        if self.download_processes > 1 and MULTIPROCESSING_AVAILABLE:
            print(f"[INFO] Using {self.download_processes} parallel download processes")
            try:
                # Attempt parallel downloads
                with Pool(processes=self.download_processes) as pool:
                    results = []
                    for job in download_jobs:
                        result = pool.apply_async(self._download_single_file, job)
                        results.append((result, job))

                    # Wait for all downloads to complete
                    for result, job in results:
                        try:
                            result.get()  # This will raise any exception from the worker
                        except Exception as e:
                            print(f"[ERROR] Parallel download failed for {job[0]}-{job[1]:02d}: {e}")

            except Exception as e:
                print(f"[WARNING] Parallel downloading failed ({e}), falling back to sequential downloads")
                # Fall back to sequential downloads
                for job in download_jobs:
                    try:
                        self._download_single_file(*job)
                    except Exception as e:
                        print(f"[ERROR] Download failed for {job[0]}-{job[1]:02d}: {e}")
                        continue
        else:
            # Single process download
            print("[INFO] Using sequential download process")
            for job in download_jobs:
                try:
                    self._download_single_file(*job)
                except Exception as e:
                    print(f"[ERROR] Download failed for {job[0]}-{job[1]:02d}: {e}")
                    continue

    @staticmethod
    def _download_single_file(
        year: int,
        month: int,
        data_directory: Path,
        variables: list[str],
        days: list[int],
        hours: list[int],
    ) -> None:
        """Download a single monthly ERA5 file from Copernicus Climate Data Store."""
        vars_id = "_".join(sorted(variables))
        out_file_path = data_directory / f"{month:02d}-{year}_{vars_id}.nc"

        # Skip if file already exists and is valid
        if out_file_path.exists() and out_file_path.stat().st_size > 0:
            print(
                f"[INFO] Skipping download - file already exists: {out_file_path.name}"
            )
            return

        # Initialize CDS API client
        assert cdsapi is not None  # availability is checked before download starts
        try:
            cds_client = cdsapi.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to init CDS client: {e}") from e

        # Prepare download request
        download_request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{day:02d}" for day in days],
            "time": [f"{hour:02d}:00" for hour in hours],
            "format": "netcdf",
        }

        # Attempt download with error handling
        try:
            print(f"[INFO] Starting CDS download: {out_file_path.name}")
            print(f"[INFO] Variables: {variables}")
            print(
                f"[INFO] Time period: {year}-{month:02d}, Days: {len(days)}, Hours: {len(hours)}"
            )

            cds_client.retrieve(
                "reanalysis-era5-single-levels", download_request, str(out_file_path)
            )

            if out_file_path.exists() and out_file_path.stat().st_size > 0:
                print(
                    f"[INFO] Successfully downloaded {out_file_path.name} "
                    f"({out_file_path.stat().st_size:,} bytes)"
                )
            else:
                raise RuntimeError("Download completed but file is missing or empty")

        except Exception as e:
            # Clean up partially downloaded file
            if out_file_path.exists():
                try:
                    out_file_path.unlink()
                except OSError:
                    pass

            raise RuntimeError(
                f"Download failed for {out_file_path.name} ({year}-{month:02d}): {e}"
            ) from e

    def _get_stats_cache_path(self) -> Path:
        """Generate a unique cache path for normalization statistics."""

        # Create a dictionary of all relevant parameters
        config = {
            "years": sorted(self.years_requested),
            "days": sorted(self.days_requested) if self.days_requested else None,
            "months": sorted(self.months_requested) if self.months_requested else None,
            "hours": sorted(self.hours_requested) if self.hours_requested else None,
            "data_variables": sorted(self.data_variables_requested),
            "numerical_time_unit": self.numerical_time_unit,
            "datetime_reference_time": self.datetime_reference_time,
            "subset_coordinate_ranges": self.subset_coordinate_ranges,
        }

        # Create a hash based on dataset configuration parameters
        cfg_str = json.dumps(config, sort_keys=True, default=str)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]

        stats_filename = f"era5_stats_{cfg_hash}.pkl"
        return self.stats_cache_directory / stats_filename

    def _compute_train_statistics(self) -> dict:
        """Compute normalization statistics from train subset."""
        if self.subset == "train":
            # If already train subset, use current dataset
            train_data = self.data
        else:
            # Create temporary train subset
            if (
                self.subset_coordinate_ranges is None
                or "train" not in self.subset_coordinate_ranges
            ):
                raise ValueError(
                    "Cannot compute train statistics: 'train' subset not defined "
                    "in subset_coordinate_ranges. Either set subset='train' or "
                    "define 'train' ranges in subset_coordinate_ranges."
                )

            # Check files (and optionally download) - similar to __init__
            available_files, missing = self._check_data_files_exist()

            if missing:
                if self.enable_download:
                    self._handle_missing_files_download(missing)
                    available_files, missing2 = self._check_data_files_exist()
                    if missing2:
                        self._raise_missing_files_error(missing2)
                else:
                    self._raise_missing_files_error(missing)

            # Load and process dataset
            combined_dataset = self._load_dataset_files(available_files)
            processed_dataset = self._process_dataset(combined_dataset)

            # Apply train subset filter
            original_subset = self.subset
            self.subset = "train"
            train_data = self._apply_subset_filter(processed_dataset)
            self.subset = original_subset  # Restore original subset
            
            # Clean up intermediate dataset
            del processed_dataset
            gc.collect()

        # Compute statistics for coordinates and data variables
        stats = {"coords_mean": {}, "coords_std": {}, "vars_mean": {}, "vars_std": {}}

        # Compute coordinate statistics (for spatial coordinates)
        coord_names = ["latitude", "longitude", "numerical_time"]
        if "surface_elevation" in train_data.coords:
            coord_names.append("surface_elevation")

        for name in coord_names:
            if name in train_data.coords:
                # compute() if dask-backed
                m = (
                    train_data.coords[name].mean().compute().item()
                    if hasattr(train_data.coords[name].data, "compute")
                    else float(train_data.coords[name].mean())
                )
                s = (
                    train_data.coords[name].std().compute().item()
                    if hasattr(train_data.coords[name].data, "compute")
                    else float(train_data.coords[name].std())
                )
                stats["coords_mean"][name] = float(m)
                stats["coords_std"][name] = float(s if s != 0 else 1.0)

        # Determine variables to normalize: use all float data_vars present
        vars_to_norm = [
            v
            for v in train_data.data_vars
            if np.issubdtype(train_data[v].dtype, np.floating)
        ]
        for v in vars_to_norm:
            mean_v = train_data[v].mean(skipna=True)
            std_v = train_data[v].std(skipna=True)
            if hasattr(mean_v.data, "compute"):  # dask
                mean_v = mean_v.compute().item()
                std_v = std_v.compute().item()
            else:
                mean_v = float(mean_v)
                std_v = float(std_v)
            stats["vars_mean"][v] = float(mean_v)
            stats["vars_std"][v] = float(std_v if std_v != 0 else 1.0)

        return stats

    def _save_statistics(self, stats: dict) -> None:
        path = self._get_stats_cache_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(stats, f)
            print(f"[INFO] Cached normalization statistics to: {path}")
        except Exception as e:
            raise RuntimeError(f"Could not save stats cache: {e}") from e

    def _load_statistics(self) -> dict | None:
        path = self._get_stats_cache_path()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    stats = pickle.load(f)
                print(f"[INFO] Loaded cached normalization statistics from: {path}")
                return stats
            except Exception as e:
                raise RuntimeError(f"Could not load stats cache: {e}") from e
        return None

    def _get_dataset_cache_path(self) -> Path:
        """Generate a unique cache path for processed dataset subset."""
        
        # Create a dictionary of all relevant parameters that affect the final dataset
        config = {
            "years": sorted(self.years_requested),
            "days": sorted(self.days_requested) if self.days_requested else None,
            "months": sorted(self.months_requested) if self.months_requested else None,
            "hours": sorted(self.hours_requested) if self.hours_requested else None,
            "data_variables": sorted(self.data_variables_requested),
            "numerical_time_unit": self.numerical_time_unit,
            "datetime_reference_time": self.datetime_reference_time,
            "subset": self.subset,
            "subset_coordinate_ranges": self.subset_coordinate_ranges,
            "drop_geo_z_variable": self.drop_geo_z_variable,
            "normalize": self.normalize,
            "normalize_exclude_coords": sorted(self.normalize_exclude_coords),
        }
        
        # Create a hash based on dataset configuration parameters
        cfg_str = json.dumps(config, sort_keys=True, default=str)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:12]
        
        cache_filename = f"era5_dataset_{self.subset}_{cfg_hash}.nc"
        return self.dataset_cache_directory / cache_filename

    def _get_numpy_cache_hash(self) -> str:
        """Generate a hash for numpy export files.

        Uses the same config keys as _get_dataset_cache_path (which already
        includes subset, normalize, and all parameters that affect the final
        state of self.data).
        """
        config = {
            "years": sorted(self.years_requested),
            "days": sorted(self.days_requested) if self.days_requested else None,
            "months": sorted(self.months_requested) if self.months_requested else None,
            "hours": sorted(self.hours_requested) if self.hours_requested else None,
            "data_variables": sorted(self.data_variables_requested),
            "numerical_time_unit": self.numerical_time_unit,
            "datetime_reference_time": self.datetime_reference_time,
            "subset": self.subset,
            "subset_coordinate_ranges": self.subset_coordinate_ranges,
            "drop_geo_z_variable": self.drop_geo_z_variable,
            "normalize": self.normalize,
            "normalize_exclude_coords": sorted(self.normalize_exclude_coords),
        }
        cfg_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(cfg_str.encode()).hexdigest()[:12]

    def export_numpy(self) -> dict[str, str]:
        """Export coordinates and data variables as .npy files for mmap access.

        Must be called after __init__ completes (post-normalization).
        Returns a dict mapping array names to file paths.
        Uses atomic writes (temp file + os.replace) for safety.
        Skips files that already exist (cache-friendly).

        Data variables are exported one at a time to avoid OOM when
        dask-backed arrays are materialized via .values/.compute().
        """
        cache_hash = self._get_numpy_cache_hash()
        cache_dir = self.dataset_cache_directory / "numpy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        def _save_array(name: str, arr: np.ndarray) -> str:
            """Save a single array with atomic write. Returns the file path."""
            fpath = cache_dir / f"era5_{name}_{cache_hash}.npy"
            str_path = str(fpath)
            if fpath.exists():
                return str_path
            # Atomic write: write to temp file in same dir, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(cache_dir), suffix=".npy"
            )
            try:
                os.close(fd)
                np.save(tmp_path, arr)
                os.replace(tmp_path, str_path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return str_path

        # --- Export coordinates (small, can do together) ---
        coord_names = ["numerical_time", "latitude", "longitude"]
        if "surface_elevation" in self.data.coords:
            coord_names.append("surface_elevation")

        for name in coord_names:
            coord = self.data.coords[name]
            arr = coord.values if not hasattr(coord.data, "compute") else coord.compute().values
            paths[name] = _save_array(f"coord_{name}", arr)

        # --- Export data variables one at a time (OOM-safe for dask) ---
        for var_name in self.data.data_vars:
            da = self.data[var_name]
            if hasattr(da.data, "compute"):
                arr = da.compute().values
            else:
                arr = da.values
            paths[str(var_name)] = _save_array(f"var_{var_name}", arr)
            del arr
            gc.collect()

        print(f"[INFO] Numpy cache ready ({len(paths)} arrays) at: {cache_dir}")
        return paths

    def _save_cached_dataset(self, dataset: xr.Dataset) -> None:
        """Save processed dataset subset to cache."""
        cache_path = self._get_dataset_cache_path()
        try:
            # Use netcdf format for efficient xarray storage
            dataset.to_netcdf(cache_path, engine="netcdf4")
            file_size = cache_path.stat().st_size
            print(f"[INFO] Cached processed dataset to: {cache_path.name} ({file_size:,} bytes)")
        except Exception as e:
            print(f"[WARNING] Failed to save dataset cache: {e}")
            # Remove partial file if it exists
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass

    def _load_cached_dataset(self) -> xr.Dataset | None:
        """Load processed dataset subset from cache."""
        cache_path = self._get_dataset_cache_path()
        if cache_path.exists():
            try:
                # Load with same chunking as original setup
                cached_ds = xr.open_dataset(
                    cache_path, 
                    chunks=self.chunks, 
                    engine="netcdf4",
                    decode_timedelta=True
                )
                
                # Fix numerical_time dtype - NetCDF converts it back to timedelta64[ns]
                if "numerical_time" in cached_ds.coords:
                    numerical_time_values = self._convert_time_to_numerical(cached_ds.coords["time"])
                    cached_ds = cached_ds.assign_coords(numerical_time=("time", numerical_time_values))
                    cached_ds.coords["numerical_time"].attrs.update(
                        units=self.numerical_time_unit,
                        long_name=f"Numerical time in {self.numerical_time_unit}",
                        reference_time=self.datetime_reference_time,
                    )
                
                file_size = cache_path.stat().st_size
                print(f"[INFO] Loaded cached dataset from: {cache_path.name} ({file_size:,} bytes)")
                return cached_ds
            except Exception as e:
                print(f"[WARNING] Failed to load dataset cache: {e}")
                # Remove corrupted cache file
                try:
                    cache_path.unlink()
                    print(f"[INFO] Removed corrupted cache file: {cache_path.name}")
                except OSError:
                    pass
        return None

    def _apply_normalization(self) -> None:
        """Apply normalization to dataset using train statistics."""
        # Try to load cached statistics first
        stats = None
        if self.cache_stats:
            stats = self._load_statistics()

        # Compute statistics if not cached or caching disabled
        if stats is None:
            print("[INFO] Computing normalization statistics from training subset")
            stats = self._compute_train_statistics()

            # Save statistics if caching enabled
            if self.cache_stats:
                self._save_statistics(stats)

        # Apply normalization to coordinates
        norm_ds = self.data.copy()

        for name, mean in stats["coords_mean"].items():
            if name in self.normalize_exclude_coords:
                continue  # leave this coordinate un-normalized (e.g. raw time)
            if name in norm_ds.coords:
                std = stats["coords_std"][name]
                if std > 0:  # Avoid division by zero
                    norm_ds = norm_ds.assign_coords(
                        {name: (norm_ds.coords[name] - mean) / std}
                    )

        # Apply normalization to data variables
        for name, mean in stats["vars_mean"].items():
            if name in norm_ds.data_vars:
                std = stats["vars_std"][name]
                if std > 0:  # Avoid division by zero
                    norm_ds[name] = (norm_ds[name] - mean) / std

        # Store statistics as attributes for reference
        norm_ds.attrs["normalization_stats"] = stats

        # Update dataset with normalized data
        self.data = norm_ds

    def clear_cache(self, cache_type: str = "all") -> None:
        """Clear cached files.
        
        Args:
            cache_type (str): Type of cache to clear. Options:
                - "all": Clear both dataset and statistics caches
                - "dataset": Clear only dataset caches  
                - "stats": Clear only statistics caches
        """
        cleared_files = []
        
        if cache_type in ("all", "dataset"):
            # Clear dataset caches - remove all era5_dataset_*.nc files
            dataset_pattern = "era5_dataset_*.nc"
            for cache_file in self.dataset_cache_directory.glob(dataset_pattern):
                try:
                    cache_file.unlink()
                    cleared_files.append(str(cache_file))
                except OSError as e:
                    print(f"[WARNING] Failed to remove {cache_file}: {e}")
        
        if cache_type in ("all", "stats"):  
            # Clear statistics caches - remove all era5_stats_*.pkl files
            stats_pattern = "era5_stats_*.pkl"
            for cache_file in self.stats_cache_directory.glob(stats_pattern):
                try:
                    cache_file.unlink()
                    cleared_files.append(str(cache_file))
                except OSError as e:
                    print(f"[WARNING] Failed to remove {cache_file}: {e}")
        
        if cleared_files:
            print(f"[INFO] Cleared {len(cleared_files)} cache file(s)")
            for f in cleared_files[:5]:  # Show first 5
                print(f"  • {Path(f).name}")
            if len(cleared_files) > 5:
                print(f"  • ... and {len(cleared_files) - 5} more")
        else:
            print(f"[INFO] No {cache_type} cache files found to clear")