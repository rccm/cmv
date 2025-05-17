#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
u0_umax_analysis.py
The script is used to generate figure 2 for the Nature manuscript

Calculate linear trends and correlations between U0 and Umax indices
for CMV, ERA5_MS, and ERA5_All datasets. Produces time series anomaly
plots and seasonal climatology insets for Northern and Southern Hemispheres.

@Guangyu Zhao (02/18/2025)
"""
import numpy as np
import xarray as xr
import pandas as pd
import proplot as pplt
from scipy import stats
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import calendar


def deseasonalize(data: np.ndarray) -> np.ndarray:
    """
    Remove monthly climatology to compute anomalies.
    Expects data with dimensions (time, level, hemisphere).
    """
    times = pd.date_range('2000-01-31', '2020-12-31', freq='M')
    ds = xr.Dataset({
        'field': (['time', 'level', 'HS'], data)
    }, coords={'time': times, 'level': np.arange(data.shape[1]), 'HS': ['NH', 'SH']})
    clim = ds.groupby('time.month').mean('time')
    return (ds.groupby('time.month') - clim).field.values


def monthly_average(data: np.ndarray) -> np.ndarray:
    """
    Compute monthly climatology (2000–2020) for each level and hemisphere.
    """
    times = pd.date_range('2000-01-01', '2020-12-31', freq='M')
    ds = xr.Dataset({
        'field': (['time', 'level', 'HS'], data)
    }, coords={'time': times, 'level': np.arange(data.shape[1]), 'HS': ['NH', 'SH']})
    return ds.groupby('time.month').mean('time').field.values


def linear_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear regression along time axis for each (level, hemisphere).
    Returns slope, p-value, intercept, standard error.
    """
    n_time, n_lev, n_hs = arr.shape
    slopes = np.full((n_lev, n_hs), np.nan)
    pvals = np.full((n_lev, n_hs), np.nan)
    intercs = np.full((n_lev, n_hs), np.nan)
    errs = np.full((n_lev, n_hs), np.nan)

    for lev in range(n_lev):
        for hs in range(n_hs):
            series = arr[:, lev, hs]
            valid = ~np.isnan(series)
            if valid.sum() < n_time - 2:
                continue
            x = np.arange(n_time)[valid]
            y = series[valid]
            res = stats.linregress(x, y)
            slopes[lev, hs] = res.slope
            pvals[lev, hs] = res.pvalue
            intercs[lev, hs] = res.intercept
            errs[lev, hs] = res.stderr
    return slopes, pvals, intercs, errs


def plot_anomalies(
    level: int,
    hs_idx: int,
    data_c: np.ndarray,
    data_e_ms: np.ndarray,
    data_e_all: np.ndarray,
    label: str,
    slopes: np.ndarray,
    pvals: np.ndarray,
    intercs: np.ndarray,
    errs: np.ndarray,
    ax: pplt.Axes
) -> None:
    """
    Scatter anomalies and overlay linear trends for a given level and hemisphere.
    """
    times = pd.date_range('2000-01-31', '2020-12-31', freq='M')
    colors = ['RoyalBlue', 'black', 'red']

    # Plot CMV
    ax.scatter(times, data_c[:, level, hs_idx], marker='o', edgecolor=colors[0],
               label=f'CMV {slopes[level,hs_idx]*120:.2f}±{errs[level,hs_idx]*120:.2f}°/decade; P={pvals[level,hs_idx]:.2f}', alpha=0.7)
    ax.plot(times, slopes[level,hs_idx] * np.arange(len(times)) + intercs[level,hs_idx],
            color=colors[0], linewidth=0.8)

    # Plot ERA5_MS
    ax.scatter(times, data_e_ms[:, level, hs_idx], marker='*', edgecolor=colors[1],
               label=f'ERA5_MS {slopes[level,hs_idx]*120:.2f}±{errs[level,hs_idx]*120:.2f}°/decade; P={pvals[level,hs_idx]:.2f}', alpha=0.7)
    ax.plot(times, slopes[level,hs_idx] * np.arange(len(times)) + intercs[level,hs_idx],
            color=colors[1], linewidth=0.8)

    # Plot ERA5_All
    ax.scatter(times, data_e_all[:, level, hs_idx], marker='^', edgecolor=colors[2],
               label=f'ERA5_All {slopes[level,hs_idx]*120:.2f}±{errs[level,hs_idx]*120:.2f}°/decade; P={pvals[level,hs_idx]:.2f}', alpha=0.7)
    ax.plot(times, slopes[level,hs_idx] * np.arange(len(times)) + intercs[level,hs_idx],
            color=colors[2], linewidth=0.8)

    ax.set_ylabel(label)


def plot_climatology(
    level: int,
    hs_idx: int,
    clim_c: np.ndarray,
    clim_e_ms: np.ndarray,
    clim_e_all: np.ndarray,
    ax: pplt.Axes
) -> None:
    """
    Plot seasonal climatology for a given level and hemisphere.
    """
    months = np.arange(1, 13)
    colors = ['RoyalBlue', 'black', 'red']
    ax.plot(months, clim_c[:, level, hs_idx], '-o', color=colors[0], label='CMV')
    ax.plot(months, clim_e_ms[:, level, hs_idx], '-*', color=colors[1], label='ERA5_MS')
    ax.plot(months, clim_e_all[:, level, hs_idx], '-^', color=colors[2], label='ERA5_All')
    ax.set_xticks(months)
    ax.set_title(f'Level {level}, HS {hs_idx}')


def main():
    smooth = 6
    hq = '50'
    data_dir = 'data'

    # Load data arrays
    cmv = np.load(f'{data_dir}/u0umax_cmv_s{smooth}_bymon_hq{hq}.npz')
    c_u0, c_max = cmv['arr_0'], cmv['arr_1']
    era_ms = np.load(f'{data_dir}/u0umax_era5_s{smooth}_bymon_hq{hq}.npz')
    e_u0_ms, e_max_ms = era_ms['arr_0'], era_ms['arr_1']
    era_all = np.load(f'{data_dir}/u0umax_eall_s{smooth}_bymon_hq{hq}.npz')
    e_u0_all, e_max_all = era_all['arr_0'], era_all['arr_1']

    # Preprocess
    c_u0 = deseasonalize(c_u0)
    y_c_u0 = monthly_average(c_u0)
    c_max = deseasonalize(c_max)
    y_c_max = monthly_average(c_max)

    e_u0_ms = deseasonalize(e_u0_ms)
    y_e_u0_ms = monthly_average(e_u0_ms)
    e_max_ms = deseasonalize(e_max_ms)
    y_e_max_ms = monthly_average(e_max_ms)

    e_u0_all = deseasonalize(e_u0_all)
    y_e_u0_all = monthly_average(e_u0_all)
    e_max_all = deseasonalize(e_max_all)
    y_e_max_all = monthly_average(e_max_all)

    # Trend fitting
    s_u0, p_u0, i_u0, t_u0 = linear_fit(c_u0)
    s_e0, p_e0, i_e0, t_e0 = linear_fit(e_u0_ms)
    s_a0, p_a0, i_a0, t_a0 = linear_fit(e_u0_all)
    s_umax_c, p_umax_c, i_umax_c, t_umax_c = linear_fit(c_max)
    s_umax_e, p_umax_e, i_umax_e, t_umax_e = linear_fit(e_max_ms)
    s_umax_a, p_umax_a, i_umax_a, t_umax_a = linear_fit(e_max_all)

    # Correlation Umax CMV vs ERA5_MS
    for hs in range(2):
        mask = ~np.isnan(c_max[:, :, hs]) & ~np.isnan(e_max_ms[:, :, hs])
        flat_c = c_max[:, :, hs][mask]
        flat_e = e_max_ms[:, :, hs][mask]
        corr = np.corrcoef(flat_c, flat_e)[0, 1]
        print(f'HS={hs}, Umax correlation: {corr:.3f}')

    # Plotting
    fig = pplt.figure(refwidth=10, refaspect=2, sharex=True, sharey=True)
    axs = fig.subplots(2, 2)

    labels = ['U0 Anomalies', 'U0 Anomalies', 'Umax Anomalies', 'Umax Anomalies']
    datasets = [(c_u0, e_u0_ms, e_u0_all, s_u0, p_u0, i_u0, t_u0),
                (c_u0, e_u0_ms, e_u0_all, s_u0, p_u0, i_u0, t_u0),
                (c_max, e_max_ms, e_max_all, s_umax_c, p_umax_c, i_umax_c, t_umax_c),
                (c_max, e_max_ms, e_max_all, s_umax_c, p_umax_c, i_umax_c, t_umax_c)]

    for ax, lvl, hs, label, data in zip(
        axs.flat, [0, 0, 1, 1], [0, 1, 0, 1], labels, datasets
    ):
        plot_anomalies(lvl, hs, *data, label, ax)
        ax.inset_axes = inset_axes(ax, width='30%', height='30%', loc='upper right')
        plot_climatology(lvl, hs,
                         y_c_u0 if lvl == 0 else y_c_max,
                         y_e_u0_ms if lvl == 0 else y_e_max_ms,
                         y_e_u0_all if lvl == 0 else y_e_max_all,
                         ax.inset_axes)

    axs.format(
        xlocator='year', xformatter='%Y', xrotation=45,
        toplabels=('NH', 'SH'), leftlabels=('U0', 'Umax'),
        abc='a', abcloc='lr', abcsize='20', ticklabelsize=10
    )
    fig.savefig('figures/fig_2.svg')

if __name__ == '__main__':
    main()
