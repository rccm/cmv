#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:07:18 2021
CVM trending
@author: gzhao1
"""
import xarray as xr
import numpy as np
import calendar
import os
import pandas as pd
import calendar
def degrade(a, factor):
    newshape = [a.shape[0]//factor,a.shape[1]//factor]
    sh = newshape[0],a.shape[0]//newshape[0],newshape[1],a.shape[1]//newshape[1]
    a=a.reshape(sh)
    return np.nanmean(np.nanmean(a,-1),1)  
i=0
resolution = 0.25
if resolution == 0.25:
        resname = 'h'
elif resolution == 1:
        resname = '1'
elif resolution == 2.5:
        resname = '8'

for binno in range(22):
    cmvbin = np.zeros((21*12,4,int(180/resolution),int(360/resolution)))+np.nan
    lats = np.arange(90-resolution/2, -90-resolution/2, -resolution)
    lons = np.arange(0, 360-resolution/2, resolution)
#timelist = pd.date_range('2000-01-01','2021-01-01', , dtype='datetime64[ns]',freq='M').strftime("%Y-%m").tolist()
    timelist = pd.date_range('2000-01-01','2021-01-01',freq='M',name="time")
    i = 0
    for year in range(2000,2021):
       for mon in range(1,13):
           ncfile = "".join('/data/gdi/c/gzhao1/cmv/era5/monthly/MonthlyERA5_Resampled_'+str(year)+'_'+str(mon).zfill(2)+'_v.nc')
           month = (calendar.month_name[mon][0:3].upper())
           cmvfile = "".join('/data/gdi/c/gzhao1/cmv/era5/output/CMV'+'_HQ'+str(50)+'_SD_'+resname+'KM_'+month+'_'+str(year)+'.nc')
           if os.path.exists(cmvfile):
                dsc = xr.open_dataset(cmvfile)
                mask = dsc.cnt[0,binno,:,:].values
                mask[mask<1] = np.nan   
                mask[~np.isnan(mask)] = 1               
           else:
               print(f'{cmvfile} doe snot exist')
               i += 1
               continue
           if os.path.exists(ncfile):
               print(ncfile)
               DS = xr.open_dataset(ncfile)
               if resolution == 0.25:
                    cmvbin[i,0,:,:] = DS.eu[binno,:,:].values*mask
                    cmvbin[i,1,:,:] = DS.ev[binno,:,:].values*mask
                    cmvbin[i,2,:,:] = DS.es[binno,:,:].values*mask
                    cmvbin[i,3,:,:] = DS.ed[binno,:,:].values*mask
               elif resolution == 1:
                    cmvbin[i,0,:,:] = degrade(np.array(DS.eu[binno,:,:].values),4)*mask
                    cmvbin[i,1,:,:] = degrade(np.array(DS.ev[binno,:,:].values),4)*mask
                    cmvbin[i,2,:,:] = degrade(np.array(DS.es[binno,:,:].values),4)*mask
                    cmvbin[i,3,:,:] = degrade(np.array(DS.ed[binno,:,:].values),4)*mask
           else:
                continue    
#          cthqflag[i,:,:,:,:] = DS.qf.values
           i += 1
    ds = xr.Dataset(
        {
            "".join("cmvbin"): (["time","band","lat","lon"], cmvbin),
     #       "QFLAG": (["time","band","lat", "lon"], cthqflag),
     #       "CNT": (["time","band","lat", "lon"], cnt)
        },
        coords={"time": timelist,"band": ['east','north','speed','direction'], "lat": lats,"lon":lons}
    )
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    outputfile = "".join('/data/gdi/c/gzhao1/cmv/era5/repack/EraMonthly_Exsfc_'+resname+'km_'+str(binno) + '.nc')
    print(outputfile)
    ds.to_netcdf(
        path= outputfile,
        mode="w",
        encoding=encoding,
    )

