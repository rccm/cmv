#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:07:18 2021
CVM trending
@author: gzhao1
"""
# %%
import xarray as xr
import numpy as np
import calendar
import os
import pandas as pd

# %%
def main(dataname,hq):
   resolution = 0.25
   lats = np.arange(90-resolution/2, -90-resolution/2, -resolution)
   lons = np.arange(-180+resolution/2, 180+resolution/2, resolution)
   #timelist = pd.date_range('2000-01-01','2021-01-01', , dtype='datetime64[ns]',freq='M').strftime("%Y-%m").tolist()
   print(lats.shape)
   cmv_all = np.zeros((240,22,180*4,360*4))+np.nan
   #cnt_all = np.zeros((240,22,180*4,360*4))+np.nan

   timelist = pd.date_range('2001-01-01','2021-01-01',freq='M',name="time")
   i=0
   for year in range(2001,2021):
         for mon in range(12):
               month = (calendar.month_name[mon+1][0:3].upper())
               ncfile = "".join('../era5/output/'+dataname+'_HQ'+str(hq)+'_SD_hKM_'+month+'_'+str(year)+'.nc')
               if os.path.exists(ncfile):
                  print(ncfile)
                  DS = xr.open_dataset(ncfile)
                  cmv_all[i] = DS.cmv[0,:,:,:].values
                  #print(np.nanmax(temp.flatten()))
                  # temp = DS.cnt[0,:,:,:].values
                  # # print(temp[(temp >0) & (temp<1)])
                  # temp[temp==0] = np.nan
                  # cnt_all[i] = temp               
               #    temp = DS.cmv.values 
               #    mask = ~np.isnan(temp)
               #    cmv_mean[mask] = cmv_mean[mask] + temp[mask]
               #    cmv_mean_cnt[mask] = cmv_mean_cnt[mask] + 1
               #    cmv_cnt[mask] = cmv_cnt[mask]+DS.cnt.values[mask]
   #           cthqflag[i,:,:,:,:] = DS.qf.values
               i += 1
         print(year)
   dsf = xr.Dataset(
      {
         "missionall": (["time", "bins", "lat","lon"], cmv_all),
   #      "missioncnt": (["time","bins", "lat","lon"], cnt_all),
   #       "QFLAG": (["time","band","lat", "lon"], cthqflag),
   #       "CNT": (["time","band","lat", "lon"], cnt)
      },
      coords={"time": timelist, "bins": np.arange(500, 22000,1000),"lat": lats, "lon": lons}
   )

   missionall  = dsf["missionall"].groupby('time.year').mean('time').mean('lon').values
  # missioncnt   =  dsf["missioncnt"].groupby('time.year').sum('time').sum('lon').values

   years = xr.Dataset(
      {
         "missionall": (["year", "bins", "lat"], missionall),
   #      "missioncnt": (["year","bins", "lat"], missioncnt),
   #       "QFLAG": (["time","band","lat", "lon"], cthqflag),
   #       "CNT": (["time","band","lat", "lon"], cnt)
      },
      coords={"year": np.arange(2001,2021),"bins": np.arange(500, 22000,1000),"lat": lats, "lon": lons}
   )
   comp = dict(zlib=True, complevel=5)
   outputfile = 'Misson_'+dataname+'_HQ'+hq+'_Year2001_hkm.nc'
   encoding = {var: comp for var in years.data_vars}
   years.to_netcdf(
         path= outputfile,
         mode="w",
         encoding=encoding,

   )
main('CMV','80')
main('CMV','50')
main('ERA5','80')
main('ERA5','50')