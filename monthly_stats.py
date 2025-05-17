#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_altitude.py

Aggregate Cloud Top Altitude (CTA) monthly on a regular latitude-longitude grid
for CMV or ERA5 datasets using the Cython 'add' extension. Distributes work
across MPI ranks for parallel processing.

@Guangyu Zhao last update: 12/21/2024
"""
import glob
import argparse
from pathlib import Path
import sys
import numpy as np
import xarray as xr
from time import strptime
import mpi4py.MPI as MPI
from addmonsd import add


def process_altitude(
    input_file: Path,
    era_dir: Path,
    dataset: str,
    resolution: float = 0.25,
    hq: int = 50
) -> None:
    """
    Process one monthly CMV/ERA5 file to compute CTA histograms.

    Parameters
    ----------
    input_file : Path
        Input MISR CMV file path (netCDF).
    era_dir : Path
        Directory containing resampled ERA5 files.
    dataset : str
        'CMV' or 'ERA5' specifying source of motion vectors.
    resolution : float, optional
        Grid resolution in degrees (default 0.25).
    hq : int, optional
        High-quality threshold (default 50).
    """
    # Map resolution to label
    res_map = {0.25: 'h', 1.0: '1', 2.5: '8'}
    resname = res_map.get(resolution)
    if resname is None:
        raise ValueError(f"Unsupported resolution: {resolution}")

    # Extract month and year from filename
    parts = input_file.stem.split('_')
    mon_abbr, year = parts[3], parts[4]
    month = str(strptime(mon_abbr, '%b').tm_mon)

    # Load input dataset
    ds = xr.open_dataset(input_file)
    dims = ds.Latitude.shape[0]
    cta = ds.CloudTopAltitude.values
    lat = ds.Latitude.values
    lon = ds.Longitude.values
    mode = ds.InstrumentHeading.values
    qflag = ds.QualityIndicator.values
    vr = 1000  # vertical resolution in meters

    # Select motion vectors
    if dataset.upper() == 'ERA5':
        era_file = era_dir / f"UVP_Era5CMV_{year}_{month}_v.nc"
        if not era_file.exists():
            raise FileNotFoundError(f"ERA5 file not found: {era_file}")
        era_ds = xr.open_dataset(era_file)
        cme = era_ds.ur.values
        cmn = era_ds.vr.values
        era_ds.close()
    elif dataset.upper() == 'CMV':
        cme = ds.CloudMotionEast.values
        cmn = ds.CloudMotionNorth.values
    else:
        raise ValueError("dataset must be 'CMV' or 'ERA5'.")

    # Compute histograms via Cython 'add'
    cmv, cnt, qf = add(
        dims, cta, cme, cmn, lat, lon,
        qflag, mode, hq, int(1/resolution), vr
    )

    # Build regular grid coordinates
    nlat = int(180 / resolution)
    nlon = int(360 / resolution)
    lats = np.linspace(90 - resolution/2, -90 + resolution/2, nlat)
    lons = np.linspace(-180 + resolution/2, 180 - resolution/2, nlon)
    bins = np.arange(vr/2, 20000 + vr, vr)

    # Create output Dataset
    ds_out = xr.Dataset(
        {
            'cmv': (['windir', 'bins', 'lat', 'lon'], cmv),
            'cnt': (['windir', 'bins', 'lat', 'lon'], cnt),
            'qf':  (['windir', 'bins', 'lat', 'lon'], qf),
        },
        coords={
            'windir': ['East', 'North', 'Speed', 'Direct'],
            'bins': bins,
            'lat': lats,
            'lon': lons,
        }
    )
    ds_out['cmv'].attrs = {'description': 'Accumulated CTH monthly', 'units': 'm'}
    ds_out['cnt'].attrs = {'description': 'Monthly sample counts',  'units': None}
    ds_out['qf'].attrs  = {'description': 'Monthly quality flags',   'units': None}

    # Write to NetCDF
    out_name = f"{dataset}_HQ{hq}_SD_{resname}KM_{year}{month}.nc"
    out_path = era_dir / out_name
    ds_out.to_netcdf(
        path=out_path,
        mode='w',
        encoding={'cnt': {'_FillValue': 0}, 'cmv': {'_FillValue': 0}}
    )
    ds.close()
    ds_out.close()


def main():
    parser = argparse.ArgumentParser(description='Process CTA monthly histograms.')
    parser.add_argument('--input_pattern', required=True,
                        help='Glob pattern for MISR CMV files')
    parser.add_argument('--era_dir', required=True,
                        help='Directory for resampled ERA5 files and output')
    parser.add_argument('--dataset', choices=['CMV', 'ERA5'], required=True,
                        help='Use CMV or ERA5 motion vectors')
    parser.add_argument('--resolution', type=float, default=0.25,
                        help='Latitude-longitude grid resolution')
    parser.add_argument('--hq', type=int, default=50,
                        help='High-quality threshold')
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    chunk = (len(files) + size - 1) // size
    start = rank * chunk
    end = min(start + chunk, len(files))

    era_dir = Path(args.era_dir)
    for f in files[start:end]:
        process_altitude(
            input_file=Path(f),
            era_dir=era_dir,
            dataset=args.dataset,
            resolution=args.resolution,
            hq=args.hq
        )

if __name__ == '__main__':
    main()
