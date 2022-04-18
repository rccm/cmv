#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:07:18 2021
CVM trending
@author: gzhao1
"""
import re
import xarray as xr
import numpy as np
import calendar
import os
import pandas as pd
 
def main(dataname,resolution=0.25,hq=80): 
    if resolution == 0.25:
            resname = 'h'
    elif resolution == 1:
            resname = '1'
    elif resolution == 2.5:
            resname = '8'
    for binno in range(22):
        cmvbin = np.zeros((21*12,4,int(180/resolution),int(360/resolution)))+np.nan
        cntbin = np.zeros((21*12,4,int(180/resolution),int(360/resolution)))+np.nan
        lats = np.arange(90-resolution/2, -90-resolution/2, -resolution)
        lons = np.arange(-180+resolution/2, 180+resolution/2, resolution)
    #timelist = pd.date_range('2000-01-01','2021-01-01', , dtype='datetime64[ns]',freq='M').strftime("%Y-%m").tolist()
        timelist = pd.date_range('2000-01-01','2021-01-01',freq='M',name="time")
        i = 0
        for year in range(2000,2021):
            for mon in range(12):
                month = (calendar.month_name[mon+1][0:3].upper())
                ncfile = "".join('../../output/'+dataname+'_HQ'+str(hq)+'_SD_'+resname+'KM_'+month+'_'+str(year)+'.nc')
                if os.path.exists(ncfile):
                    print(ncfile)
                    DS = xr.open_dataset(ncfile)
                    cmvbin[i,:,:,:] = DS.cmv[:,binno,:,:].values
                    cntbin[i,:,:,:] = DS.cnt[:,binno,:,:].values
        #          cthqflag[i,:,:,:,:] = DS.qf.values
                i += 1
        ds = xr.Dataset(
            {
                "".join("cmvbin"): (["time","band","lat","lon"], cmvbin),
                "".join("cntbin"): (["time","band","lat","lon"], cntbin),
        #       "QFLAG": (["time","band","lat", "lon"], cthqflag),
        #       "CNT": (["time","band","lat", "lon"], cnt)
            },
            coords={"time": timelist,"band": ['east','north','speed','direct'], "lat": lats,"lon":lons}
        )  
        outputfile = "".join('../../repack/Mission_'+dataname+'_HQ'+str(hq)+'_SD_'+resname+'KM_'+str(binno) + '.nc')
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        print(outputfile)
        ds.to_netcdf(
            path= outputfile,
            mode="w",
            encoding= encoding,
        )

main('CMV',hq=80,resolution=1)
main('ERA5',hq=80,resolution=1)