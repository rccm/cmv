#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_zarr_processing.py
Process ERA5 and CMV datasets: resample ERA5 onto CMV grid and
save outputs as Zarr stores using Dask for parallel computation.

@Guangyu Zhao 12/12/2024
"""
import os
import glob
import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import mpi4py.MPI as MPI

def process_era_to_zarr(era_path: str, cmv_path: str, output_dir: Path) -> None:
    """
    Resample ERA5 data onto CMV grid and write as a compressed Zarr store.

    Parameters
    ----------
    era_path : str
        Path to the input ERA5 NetCDF file.
    cmv_path : str
        Path to the CMV NetCDF file defining the target grid.
    output_dir : Path
        Directory where the Zarr store will be saved.
    """
    # Derive year and month from filename
    parts = Path(era_path).stem.split('_')
    year, mon = parts[1], parts[2]
    zarr_path = output_dir / f"resampled_era5_{year}_{mon}.zarr"
    if zarr_path.exists():
        print(f"Skipping existing: {zarr_path}")
        return

    # Start Dask client
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    # Load CMV grid and ERA5 data, with chunking
    ds_cmv = xr.open_dataset(cmv_path, chunks={'time': 'auto'})
    ds_era = xr.open_dataset(era_path, chunks={'valid_time': 1, 'latitude': -1, 'longitude': -1})

    # Normalize longitudes to [0, 360)
    lons = ds_cmv.Longitude.where(ds_cmv.Longitude >= 0, ds_cmv.Longitude + 360)
    lats = ds_cmv.Latitude
    times = ds_cmv.Time

    # Nearest-neighbor selection onto CMV grid
    ds_resampled = ds_era.sel(
        longitude=lons, latitude=lats, valid_time=times, method='nearest'
    )[['z', 'u', 'v']]

    # Clear attributes to reduce storage
    ds_resampled.attrs = {}
    for var in ds_resampled.data_vars:
        ds_resampled[var].attrs = {}

    # Rechunk along time dimension for Zarr writing
    ds_rechunked = ds_resampled.chunk({'Time': 10000})

    # Prepare compressor
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = {v: {'compressor': compressor} for v in ds_rechunked.data_vars}

    # Write to Zarr
    print(f"Writing {zarr_path}")
    with ProgressBar():
        ds_rechunked.to_zarr(store=str(zarr_path), mode='w', encoding=encoding)
    print(f"Finished {zarr_path}")

    client.close()

def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    print(f"MPI ranks: {size}, current rank: {rank}")

    # File lists
    era_files = sorted(glob.glob("/data/gdi/f/gzhao1/era/eradaily*_2021_02_v.nc"))
    cmv_pattern = "/data/gdi/c/gzhao1/arctic/data/misrcmv/l3/M*{mon}{year}*.nc"
    output_dir = Path("/data/gdi/c/gzhao1/cmv/era5/resampled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribute work among ranks
    n_files = len(era_files)
    per_rank = (n_files + size - 1) // size
    start = rank * per_rank
    end = min(start + per_rank, n_files)

    for era_path in era_files[start:end]:
        parts = Path(era_path).stem.split('_')
        year, mon = parts[1], parts[2]
        mon_abbr = calendar.month_abbr[int(mon)].upper()
        cmv_files = glob.glob(cmv_pattern.format(mon=mon_abbr, year=year))
        if not cmv_files:
            print(f"No CMV file for {year}-{mon}")
            continue
        process_era_to_zarr(era_path, cmv_files[0], output_dir)

if __name__ == '__main__':
    main()
