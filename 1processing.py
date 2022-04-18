#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 30 19:16:23 2021
%%cython
@author: gzhao1
"""
import xarray as xr
import numpy as np
import glob
from addmonsd import add
from time import strptime
import os,sys

def processalt(filename,erapath,dataname, resolution=0.25, hq=50):
    
    if resolution == 0.25:
        resname = 'h'
    elif resolution == 1:
        resname = '1'
    elif resolution == 2.5:
        resname = '8'

    mon  =  filename.split('_')[3]
    year =  filename.split('_')[4]
    mon = str(strptime(mon,'%b').tm_mon)
    erafile =  "".join(erapath+'UVP_Era5CMV_'+year+'_'+mon+'_v.nc')
    print(erafile)
    if not os.path.exists(erafile):
        print("cannot find the erafile {0}".format(erafile))
        sys.exit(0)
    dsera = xr.open_dataset(erafile)
    DS=xr.open_dataset(filename)
    #lon, lat = np.meshgrid(np.arange(-180+resolution/2, 180+resolution/2, resolution),np.arange(90-resolution/2, -90-resolution/2, -resolution))
    cth = np.zeros(shape=(int(180/resolution),int(360/resolution)))
    cnt = np.zeros(shape=(int(180/resolution),int(360/resolution)))
    cta = DS.CloudTopAltitude.values
    lat = DS.Latitude.values
    lon = DS.Longitude.values
    mode = DS.InstrumentHeading.values
    cme = DS.CloudMotionEast.values
    cmn = DS.CloudMotionNorth.values
    cmur = dsera.ur.values
    cmvr = dsera.vr.values
    qflag = DS.QualityIndicator
    dims = DS.Latitude.shape[0]
# vertical resolution
    vr = 1000
    print(cta[cta>0].size)
    if dataname == 'ERA5':
        cmv, cnt, qf = add(dims,cta,cmur,cmvr,lat, lon, qflag,mode,hq,int(1/resolution),vr)
    elif dataname == 'CMV': 
        cmv, cnt, qf = add(dims,cta,cme,cmn,lat, lon, qflag,mode,hq,int(1/resolution),vr)
    else: 
        print("wrong datasets. exitting...")
        sys.exit(0)
    #for i in tqdm.tqdm(range(DS.Latitude.shape[0])):
    #    x = int((90-DS.Latitude[i].values)/resolution)
    #    y = int(((180.0+DS.Longitude[i].values)/resolution))
    #    value = DS.CloudTopAltitude[i].values
    #    if value > 0 and value < 300000 and x >=0 and y>= 0:
    #       if y >= 360/resolution:
    #           y = int(360/resolution) - 1
    #       if x >= 180/resolution:
    #           x = int(180/resolution) - 1
    #       cth[x,y] = cth[x,y] + value
    #       cnt[x,y] = cnt[x,y] + 1
           #break
     # create xarray dataset
    lats = np.arange(90-resolution/2, -90-resolution/2, -resolution)
    lons = np.arange(-180+resolution/2, 180+resolution/2, resolution)
    ds = xr.Dataset(
        {
            "cmv": (["windir","bins","lat", "lon"], cmv),
            "cnt": (["windir","bins","lat", "lon"], cnt),
            "qf": (["windir", "bins","lat", "lon"], qf),
        },
        coords={"windir": ['East', 'North','Speed','Direct'],"bins": np.arange(int(vr/2),int(20000/vr+2)*vr,vr),"lat": lats, "lon": lons},
    )
    ds["cmv"].attrs[
        "description"
    ] = "Accumulated CTH monthly ."
    ds["cmv"].attrs["unit"] = "m"

    ds["cnt"].attrs[
        "description"
    ] = "Number of samples monthly."
    ds["qf"].attrs[
                "description"

    ] = "Quality Fag monthly."
    ds["qf"].attrs["unit"] = "None"
    ds["cnt"].attrs["unit"] = "None"
    outputfile = "".join('../../output/'+dataname+'_HQ'+str(hq)+'_SD_'+resname+'KM_'+filename[-20:-12] + '.nc')
    ds.to_netcdf(
        path= outputfile,
        mode="w",
        encoding={
            "cnt": {"_FillValue": 0},
            "cmv": {"_FillValue": 0},
            },
    )
if __name__ == "__main__":
    import mpi4py.MPI as MPI
    erapath ='/data/gdi/c/gzhao1/cmv/era5/resampled/'
    cmm = MPI.COMM_WORLD
    rank = cmm.Get_rank()
    n_ranks = cmm.Get_size()
    print("there are {:d} rank".format(n_ranks))
    files = glob.glob("/data/gdi/c/gzhao1/arctic/data/misrcmv/l3/M*.nc")
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
                    print(file)
                    processalt(file, erapath,'CMV',hq=50,resolution=1)
                    processalt(file, erapath,'ERA5',hq=50,resolution=1)



