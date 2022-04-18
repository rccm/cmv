#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gzhao1
"""
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
import sys
resolution = 0.25
lats = np.arange(90-resolution/2, -90-resolution/2, -resolution)
lons = np.arange(-180+resolution/2, 180+resolution/2, resolution)
#timelist = pd.date_range('2000-01-01','2021-01-01', , dtype='datetime64[ns]',freq='M').strftime("%Y-%m").tolist()
cmvall = np.zeros((3,2,252,int(180/resolution))) + np.nan
timelist = pd.date_range('2000-01-01','2021-01-01',freq='M',name="time")

hq=50
for i in range(2):
     cfile='../era5/repack/Mission_CMV_HQ'+str(hq)+'_SD_hKM_'+str(i)+'.nc'
     dsc = xr.open_dataset(cfile)
     dsc = dsc.mean('lon')
     #ds.cmvbin[ds.time.dt.year == 2000] = np.nan
     cmvall[0,i,:,:] = dsc.cmvbin[:,0,:]   

     efile='../era5/repack/Mission_ERA5_HQ'+str(hq)+'_SD_hKM_'+str(i)+'.nc'
     dse = xr.open_dataset(efile)
     dse = dse.mean('lon')
     #ds.cmvbin[ds.time.dt.year == 2000] = np.nan
     cmvall[1,i,:,:] = dse.cmvbin[:,0,:]   

     afile='../era5/repack/EraMonthly_Exsfc_hkm_'+str(i)+'.nc'
     dsa = xr.open_dataset(afile)
     dsa = dsa.mean('lon')
     #ds.cmvbin[ds.time.dt.year == 2000] = np.nan
     cmvall[2,i,:,:] = dsa.cmvbin[:,0,:]   
     print(f'level {i} is done')

    
dsm = xr.Dataset(
    {
        "cmvall": (["dataname","bin","time", "lat"], cmvall),
 #       "QFLAG": (["time","band","lat", "lon"], cthqflag),
    },
    coords={"dataname":['CMV','ERA5','EALL'], \
          "bin": np.arange(2), "time": dsc.time, "lat": dsc.lat}
)
outputfile = "".join('../output/U0max_CMVERA_Exsfc_HQ'+str(hq)+'_hkm.nc')
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in dsm.data_vars}
dsm.to_netcdf(
        path= outputfile,
        mode="w",
        encoding=encoding,
    )
 
