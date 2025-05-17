#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trending_zonal.py
Compute zonal trends of CMV and ERA5 histograms (east, north, speed, direction)
using the Mann-Kendall test. Processes NetCDF files in parallel via MPI.

@Guangyu Zhao last update: 12/22/2024
"""
import glob
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
import pymannkendall as mk
from mpi4py import MPI


def mktest1d(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Mann-Kendall trend test along time axis for each latitude bin.

    Parameters
    ----------
    arr : np.ndarray
        2D array with shape (time, lat).

    Returns
    -------
    slope : np.ndarray
        Trend slopes (lat,).
    p_value : np.ndarray
        Corresponding p-values (lat,).
    """
    n_time, n_lat = arr.shape
    slopes = np.full(n_lat, np.nan)
    p_values = np.full(n_lat, np.nan)

    for j in range(n_lat):
        series = arr[:, j]
        valid = ~np.isnan(series)
        if valid.sum() <= 2:
            continue
        result = mk.original_test(series[valid])
        slopes[j] = result.slope * 120  # convert to per decade
        p_values[j] = result.p

    return slopes, p_values


def compute_zonal_trends(input_path: Path) -> xr.Dataset:
    """
    Compute zonal mean, anomalies, and Mann-Kendall trends for CMV bins.

    Parameters
    ----------
    input_path : Path
        NetCDF file containing 'cmvbin' variable with dimensions (time, windir, lat).

    Returns
    -------
    xr.Dataset
        Dataset with trend slopes, p-values, and long-term means for each wind direction.
    """
    ds = xr.open_dataset(input_path)
    # Compute longitude mean
    ds_zonal = ds.mean(dim='lon')

    # Remove seasonal cycle
    clim = ds_zonal.groupby('time.month').mean('time')
    anomalies = ds_zonal.groupby('time.month') - clim

    # Extract anomalies for each component
    cme = anomalies.cmvbin[:, 0, :].values
    cmn = anomalies.cmvbin[:, 1, :].values
    cms = anomalies.cmvbin[:, 2, :].values
    cmd = anomalies.cmvbin[:, 3, :].values

    # Long-term means
    mean_ds = ds_zonal.mean(dim='time')
    mean_east = mean_ds.cmvbin[0, :].values
    mean_north = mean_ds.cmvbin[1, :].values
    mean_speed = mean_ds.cmvbin[2, :].values
    mean_dir = mean_ds.cmvbin[3, :].values

    # Compute trends and p-values
    slope_east, p_east = mktest1d(cme)
    slope_north, p_north = mktest1d(cmn)
    slope_speed, p_speed = mktest1d(cms)
    slope_dir, p_dir     = mktest1d(cmd)

    # Assemble output Dataset
    ds_out = xr.Dataset(
        {
            'TrendEast': (['lat'], slope_east),
            'PValueEast': (['lat'], p_east),
            'TrendNorth': (['lat'], slope_north),
            'PValueNorth': (['lat'], p_north),
            'TrendSpeed': (['lat'], slope_speed),
            'PValueSpeed': (['lat'], p_speed),
            'TrendDirection': (['lat'], slope_dir),
            'PValueDirection': (['lat'], p_dir),
            'MeanEast': (['lat'], mean_east),
            'MeanNorth': (['lat'], mean_north),
            'MeanSpeed': (['lat'], mean_speed),
            'MeanDirection': (['lat'], mean_dir),
        },
        coords={'lat': ds.lat}
    )

    ds.close()
    return ds_out
  
def main():
    parser = argparse.ArgumentParser(description='Compute zonal trends on CMV histograms.')
    parser.add_argument('--input_pattern', required=True,
                        help='Glob pattern for input NetCDF files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save trend NetCDF outputs')
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute files among MPI ranks
    n = len(files)
    per = (n + size - 1) // size
    start = rank * per
    end = min(start + per, n)

    for f in files[start:end]:
        inp = Path(f)
        ds_out = compute_zonal_trends(inp)
        out_name = inp.stem + '_zonal_trend.nc'
        ds_out.to_netcdf(out_dir / out_name)
        ds_out.close()

if __name__ == '__main__':
    main()
