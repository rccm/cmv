#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figur 1 for the Nature manuscript

Script to analyze and plot zonal trends in wind speed and components
using MISR and ERA5 datasets, applying FDR correction and visualizing
altitude profiles.

Author: Guangyu Zhao 02/13/2025
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import proplot as pplt
from statsmodels.stats.multitest import multipletests


def fdr_correction_multi_alpha_tighten(
    slope: np.ndarray,
    p_values: np.ndarray,
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Storey's method to estimate the proportion of true nulls and
    adjust the FDR threshold, then perform Benjamini-Hochberg correction.

    Returns:
        sig_slope: np.ndarray
            Slope values significant after FDR tightening, others set to NaN.
        uncorrected: np.ndarray
            Original slope with values where p > alpha set to NaN.
    """
    flat_p = p_values.flatten()
    valid = ~np.isnan(flat_p)
    pvals = flat_p[valid]

    def estimate_pi0(p, lambda_val=0.5):
        pi0_est = np.sum(p > lambda_val) / ((1 - lambda_val) * len(p))
        return min(pi0_est, 1.0)

    pi0 = estimate_pi0(pvals)
    adjusted_alpha = min(alpha / (1 - pi0), 1.0)

    rejects, _, _, _ = multipletests(pvals, alpha=adjusted_alpha, method='fdr_bh')
    mask_rejects = np.full_like(flat_p, False, dtype=bool)
    mask_rejects[valid] = rejects
    mask_grid = mask_rejects.reshape(p_values.shape)

    sig_slope = np.where(mask_grid, slope, np.nan)
    uncorrected = slope.copy()
    uncorrected[p_values > alpha] = np.nan
    return sig_slope, uncorrected


def zonal_plot_trend(
    slope: np.ndarray,
    mean_field: np.ndarray,
    tropo_height: np.ndarray,
    ax: plt.Axes,
    vmin: float,
    vmax: float,
    levels: np.ndarray,
    cmap_name: str,
    show_colorbar: bool = True
) -> None:
    """
    Plot zonal trend image, overlay tropopause height, and contour the mean field.
    """
    cmap = plt.get_cmap(cmap_name, len(levels))
    im = ax.imshow(
        slope, origin='lower', extent=[0, slope.shape[1], 0, slope.shape[0]],
        cmap=cmap, interpolation='none', aspect='auto', vmin=vmin, vmax=vmax
    )
    ax.plot(np.arange(len(tropo_height)), tropo_height, color='gray', linewidth=1)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Trend [units]',
                     ticks=np.linspace(vmin, vmax, num=6))

    contours = ax.contour(
        mean_field, origin='lower', levels=levels,
        colors='black', linewidths=0.8
    )
    ax.clabel(contours, inline=True, fontsize=6)

    ax.set_xticks(np.linspace(0, slope.shape[1], 7))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.invert_xaxis()
    ax.set_ylim(0, slope.shape[0])
    ax.grid(True, linewidth=0.5)
    ax.set_ylabel('Altitude [km]')


def main():
    tropo_file = 'data/tphlat_mean.nc'
    cmv_file = 'data/stderrall_U.nc'
    era5_file = 'output/erazonaltaltvswind_sd_hkm_hq50.nc'
    test_file = 'output/zonalcombine_trend_diff_ztest.nc'

    # Load tropopause height
    ds_tropo = xr.open_dataset(tropo_file)
    raw_tropo = ds_tropo.tphlat.mean('month').values
    tropo_height = 0.5 * (raw_tropo[1:] + raw_tropo[:-1])

    # Load trend data and create mask
    ds = xr.open_dataset(era5_file)
    ds_cmv = xr.open_dataset(cmv_file)
    mask = np.where(ds_cmv.samples.values < 5000, np.nan, 1)

    slope = ds.alts.values * mask
    p_values = ds.altsp.values * mask
    mean_field = ds.meanspeed.values * mask

    # Mask non-significant trends by Z-test
    ds_test = xr.open_dataset(test_file)
    p_diff = ds_test.PValue_Diff_Speed.values
    slope[p_diff > 0.05] = np.nan

    # Apply FDR tightening
    sig_slope, _ = fdr_correction_multi_alpha_tighten(slope, p_values)

    # Plot results
    fig, ax = pplt.subplots(refwidth=4, refaspect=1)
    zonal_plot_trend(
        sig_slope, mean_field, tropo_height, ax,
        vmin=-5, vmax=5, levels=np.arange(0, 30, 5), cmap_name='coldhot'
    )

    plt.savefig('figures/allwinds_trends.jpg', dpi=600,
                bbox_inches='tight')

if __name__ == "__main__":
    main()
