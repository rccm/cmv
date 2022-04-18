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
def mktest1d(arr,numnan):
    [y,x] = arr.shape
    p_val = np.empty((x))+np.nan
    slope = np.empty((x))+np.nan
    for i in range(x):
        if np.count_nonzero(np.isnan(arr[:,i])) > numnan:
           continue
        p_val[i] = mk.original_test(arr[:,i]).p
        slope[i] = mk.original_test(arr[:,i]).slope
    return slope, p_val

def processtrend(inputfile, outputfilename,resolution):
    import numpy as np
    ds = xr.open_dataset(inputfile)
    """ 2000 data are removed """
    ds.cmvbin[ds.time.dt.year == 2000] = np.nan
    ds  =  ds.mean(('lon'))
    climatology = ds.groupby("time.season").mean("time")
    ano = ds.groupby("time.season") - climatology
    midx=ano.groupby('time.season').groups

    # if ano.groupby('season').groups, then use 'MAM'....
   
    mes = np.empty((4,resolution))
    mep = np.empty((4,resolution))
   
    mns = np.empty((4,resolution))
    mnp = np.empty((4,resolution))
    
    mss = np.empty((4,resolution))
    msp = np.empty((4,resolution))
   
    mds = np.empty((4,resolution))
    mdp = np.empty((4,resolution))
 

    meancme = np.empty((4,resolution))
    meancmn = np.empty((4,resolution))  
    meancms = np.empty((4,resolution))  
    meancmd = np.empty((4,resolution))             

    
    s = -1
    for i in ['DJF', 'MAM', 'JJA', 'SON']:
#        monano=ano.isel(time=midx[i])
        monmean = ds.sel(time=ds.time.dt.season==i)
        seasonclimatology = monmean.groupby("time.month").mean("time")
        # monano =  monmean.groupby(monmean.time.dt.year).mean("time")
        monano = monmean.groupby("time.month") - seasonclimatology
        cme = monano.cmvbin[:,0,:].values
        cmn = monano.cmvbin[:,1,:].values
        cms = monano.cmvbin[:,2,:].values   
        cmd = monano.cmvbin[:,3,:].values
        dsm = ds.isel(time=midx[i])
        dsm = dsm.mean('time')
        s +=1
        print(monano.cmvbin.shape)
        meancme[s,:] = dsm.cmvbin[0,:].values
        meancmn[s,:] = dsm.cmvbin[1,:].values
        meancms[s,:] = dsm.cmvbin[2,:].values
        meancmd[s,:] = dsm.cmvbin[3,:].values
        es,ep = mktest1d(cme,55)
        es = es*30
        mes[s,:] = es
        mep[s,:] = ep
        ns,np = mktest1d(cmn,55) 
        ns = ns*30
        mns[s,:] = ns
        mnp[s,:] = np
        ss,sp = mktest1d(cms,55) 
        ss = ss*30
        mss[s,:] = ss
        msp[s,:] = sp
        dds,ddp = mktest1d(cmd,55) 
        ns = ns*30
        mds[s,:] = dds
        mdp[s,:] = ddp
    #slopep = np.copy(slope)
    #slopep[p_val > 0.05] =np.nan
    #slopep =slopep*120

    #nslopep = np.copy(nslope)
    #nslopep[np_val > 0.05] =np.nan
    #nslopep = slopep*120
    dsz = xr.Dataset(
        {
            "ZonalEastTrend_S": (["season","lat"], mes),
            "ZonalEastTrend_P": (["season","lat"], mep),
            "ZonalNorthTrend_S": (["season","lat"], mns),
            "ZonalNorthTrend_P": (["season","lat"], mnp),
            "ZonalSpeedTrend_S": (["season","lat"], mss),
            "ZonalSpeedTrend_P": (["season","lat"], msp),
            "ZonalDirectTrend_S": (["season","lat"], mds),
            "ZonalDirectTrend_P": (["season","lat"], mdp),

            "ZonalMeanNorth": (["season","lat"], meancmn),
            "ZonalMeanEast": (["season","lat"], meancme),
            "ZonalMeanSpeed": (["season","lat"], meancms),
            "ZonalMeanDirect": (["season","lat"], meancmd),
        },
        coords={"month":['DJF', 'MAM', 'JJA', 'SON'],"lat": ds.lat},
    )
    outputfile = outputfilename
    dsz.to_netcdf(
        path= outputfile,
        mode="w",
    )
def main(data_name,resolution):
    import glob
    import mpi4py.MPI as MPI
    cmm = MPI.COMM_WORLD
    rank = cmm.Get_rank()
    n_ranks = cmm.Get_size()
    print('start processing...')
    print("there are {:d} rank".format(n_ranks))
    #da_name = 'EALL'
    #resolution = 720
    workdir = '/data/gdi/c/gzhao1/cmv/era5/repack/'
    
    if resolution == 180:
        files = glob.glob("".join(workdir+'Mission_'+data_name+'_HQ50_SD_1KM_*.nc'))
        res='1km'
    elif resolution == 720:
        if data_name == 'EALL' :
            files = glob.glob("".join(workdir+'EraMonthly_hkm_*.nc')) 
            res='hkm'
            print('hafasfa')
        else:
            files = glob.glob("".join(workdir+'Mission_'+data_name+'_HQ50_SD_hKM_*.nc'))
            res='hkm'
    elif resolution == 72:
        files = glob.glob("".join(workdir+'Mission_'+data_name+'_HQ50_SD_8KM_*.nc'))    
        res='8km'
    else:
        print('Cannot find the source files ...') 
        sys.exit(0)
    print("I'm rankd {:d}".format(rank))
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
                     print(file.split('_')[2])
                     if data_name == 'EALL':
                        outfilename = "".join('/data/gdi/c/gzhao1/cmv/output/'+data_name+'SeasonTrendingTurbo_HQ50_'+res+'_2001_'+file.split('_')[2].split('.')[0]+'_bin.nc')
                     else:
                        outfilename = "".join('/data/gdi/c/gzhao1/cmv/output/'+data_name+'SeasonTrendingTurbo_HQ50_'+res+'_2001_'+file.split('_')[5].split('.')[0]+'_bin.nc')
                     print(outfilename)
                     processtrend(file,outfilename,resolution)
if __name__ == "__main__":
   main('CMV',720)
   main('ERA5',720)
