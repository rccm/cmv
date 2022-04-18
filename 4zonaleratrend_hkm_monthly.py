#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gzhao1
"""
# %%
import xarray as xr
import numpy as np
import calendar
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
import pymannkendall as mk
import copy
def mktest(arr):
    [x,y,z] = arr.shape
    p_val = np.empty((y,z))+np.nan
    slope = np.empty((y,z))+np.nan
    for i in range(y):
        for j in range(z):
            if np.count_nonzero(np.isnan(arr[:,i,j])) > 200:
                continue
            p_val[i,j] = mk.original_test(arr[:,i,j]).p
            slope[i,j] = mk.original_test(arr[:,i,j]).slope
    return slope,p_val
def mktest1d(arr):
    [y,x] = arr.shape
    p_val = np.empty((x))+np.nan
    slope = np.empty((x))+np.nan
    for i in range(x):
        if np.count_nonzero(np.isnan(arr[:,i])) > 230:
           continue
        p_val[i] = mk.original_test(arr[:,i]).p
        slope[i] = mk.original_test(arr[:,i]).slope
    return slope, p_val
def processtrend(inputfile, outputfilename):
    ds = xr.open_dataset(inputfile)
    ds  =  ds.mean(('lon'))
    climatology = ds.groupby("time.month").mean("time")
    anomalies = ds.groupby("time.month") - climatology
    cme = anomalies.cmvbin[:,0,:].values
    cmn = anomalies.cmvbin[:,1,:].values
    cms = anomalies.cmvbin[:,2,:].values
    cmd = anomalies.cmvbin[:,3,:].values
    print(cme.shape)
    dsm = ds.mean(('time'))
    print(dsm.keys())
    meancme = dsm.cmvbin[0,:].values
    meancmn = dsm.cmvbin[1,:].values
    meancms = dsm.cmvbin[2,:].values
    meancmd = dsm.cmvbin[3,:].values
    es,ep = mktest1d(cme)
    print('cme is done')
    #slopep = np.copy(slope)
    #slopep[p_val > 0.05] =np.nan
    #slopep =slopep*120
    es = es*120
    ns,np = mktest1d(cmn)
    #nslopep = np.copy(nslope)
    #nslopep[np_val > 0.05] =np.nan
    #nslopep = slopep*120
    ns = ns*120
    ss,sp = mktest1d(cms)
    #nslopep = np.copy(nslope)
    #nslopep[np_val > 0.05] =np.nan
    #nslopep = slopep*120
    ss = ss*120

    rs,rp = mktest1d(cmd)
    #nslopep = np.copy(nslope)
    #nslopep[np_val > 0.05] =np.nan
    #nslopep = slopep*120
    rs = rs*120


    dsz = xr.Dataset(
        {
            "ZonalEastTrend_S": (["lat"], es),
            "ZonalEastTrend_P": (["lat"], ep),
            "ZonalNorthTrend_S": (["lat"], ns),
            "ZonalNorthTrend_P": (["lat"], np),
            "ZonalSpeedTrend_S": (["lat"], ss),
            "ZonalSpeedTrend_P": (["lat"], sp),
            "ZonalDirectTrend_S": (["lat"], rs),
            "ZonalDirectTrend_P": (["lat"], rp),
            "ZonalMeanNorth": (["lat"], meancmn),
            "ZonalMeanEast": (["lat"], meancme),
            "ZonalMeanSpeed": (["lat"], meancms),
            "ZonalMeanDirect": (["lat"], meancmd),
        },
        coords={"lat": ds.lat},
    )
    outputfile = outputfilename
    dsz.to_netcdf(
        path= outputfile,
        mode="w",
    )
if __name__ == "__main__":
    import glob
    import mpi4py.MPI as MPI
    cmm = MPI.COMM_WORLD
    rank = cmm.Get_rank()
    n_ranks = cmm.Get_size()
    print("there are {:d} rank".format(n_ranks))
    files = glob.glob("/data/gdi/c/gzhao1/cmv/era5/repack/EraMonthly_Exsfc_hkm_*.nc")
    print("I'm rankd {:d}".format(rank))
    print(len(files))
    numbers = len(files)
    numbers_per_rank = numbers // n_ranks
    if numbers % n_ranks > 0:
        numbers_per_rank += 1
    for r in range(n_ranks):
        if rank == r:
            my_first = rank * numbers_per_rank
            my_last = my_first + numbers_per_rank
            for i in range(my_first, my_last):
                if i < numbers:
                    file = files[i]
                    outfilename = "".join('../output/TrendingZonal_Era_Monthly_Exsfc_hkm_'+file.split('_')[3].split('.')[0]+'_bin.nc')
                    print(outfilename)
                    processtrend(file,outfilename)


# %%
