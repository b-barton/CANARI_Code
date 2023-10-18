#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import glob
import netCDF4 as nc
import datetime as dt
import sys
import gsw as sw
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cftime
import coast
import xarray as xr
import dask
import haversine as ha


# In[ ]:


root = '/gws/nopw/j04/canari/users/benbar/Data/Sat_Sea_Ice/'
fn_dat_sic = root + 'METOFFICE-GLO-SST-L4-REP-OBS-SST_*.nc'

out_file = './Processed/'


# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1
now = dt.datetime.now()

flist_sic = sorted(glob.glob(fn_dat_sic))

v_map = {}
v_map['tmask'] = 'mask'
v_map['lat'] = 'lat'
v_map['lon'] = 'lon'
v_map['time'] = 'time'
v_map['siconc'] = 'sea_ice_fraction'

with nc.Dataset(flist_sic[0], 'r') as nc_fid:
    siconc = nc_fid.variables[v_map['siconc']][:]
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]
    tmask = nc_fid.variables[v_map['tmask']][:]


lon_bnds, lat_bnds = (-180, 180), (60, 90)
y1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])
y2 = np.max(np.nonzero((lat <= lat_bnds[1]))[0])
x1 = np.min(np.nonzero((lon >= lon_bnds[0]))[0])
x2 = np.max(np.nonzero((lon <= lon_bnds[1]))[0])
    
lat = lat[y1:y2:sub]
lon = lon[x1:x2:sub]
tmask = (tmask[0, y1:y2:sub, x1:x2:sub] == 0)[np.newaxis, :, :]

def dask_load_i(fn_sic, tmask, y1, y2, x1, x2, sub):
    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fn_sic]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['siconc']][:, y1:y2:sub, x1:x2:sub], shape=siconc[:, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    # daily to monthly
    lazy_arrays = [x.mean(axis=0)[np.newaxis, :, :]
                        for x in lazy_arrays]
    ds_sic = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_sic = ds_sic.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = dask.array.from_array(tmask)
    tmask = tmask.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = tmask.repeat(ds_sic.shape[0], axis=0)
    print(ds_sic.shape, tmask.shape)
    ds_sic = dask.array.ma.masked_array(ds_sic, mask=tmask)
    
    print(ds_sic) # t, z, y, x

    return ds_sic


# In[ ]:


dx, dy = ha.area(lon, lat)
area = np.zeros((lat.shape[0], lon.shape[0]))
area[:lat.shape[0], :lon.shape[0]] = dx * dy
area[-1, :] = area[-2, :]
area[:, -1] = area[:, -2]

mask_south = np.zeros((lat.shape[0], lon.shape[0], 3), dtype=bool)
mask_south[:, :, 0] = lat[:, np.newaxis].repeat(lon.shape[0], axis=1) < 70
mask_south[:, :, 1] = lat[:, np.newaxis].repeat(lon.shape[0], axis=1) < 75
mask_south[:, :, 2] = lat[:, np.newaxis].repeat(lon.shape[0], axis=1) < 80

    
ref = dt.datetime(1981, 1, 1)
date = np.zeros((len(flist_sic)), dtype=object)
c = 0
for i in range(len(flist_sic)):
    with nc.Dataset(flist_sic[i], 'r') as nc_fid:
        time = nc_fid.variables[v_map['time']][:]
    date[i] = ref + dt.timedelta(seconds=int(time[0]))

    
area = dask.array.from_array(area[np.newaxis, :, :], chunks={0:1, 1:100, 2:100})
mask_south = dask.array.from_array(mask_south[np.newaxis, :, :, :], chunks={0:1, 0:100, 1:100, 2:-1})
area = area.repeat(12, axis=0)
mask_south = mask_south.repeat(12, axis=0)

sic_time = np.ma.zeros((len(flist_sic), 3))

for i in range(len(flist_sic) // 12):
    #i1 = int(i)
    #i2 = int((i + 1))
    i1 = int(i * 12)
    i2 = int((i + 1) * 12)
    
    ds_sic = dask_load_i(flist_sic[i1:i2], tmask, y1, y2, x1, x2, sub)

    sic_part = (
        dask.array.ma.masked_where(mask_south, 
        (ds_sic * area)
        [:, :, :, np.newaxis].repeat(3, axis=3)) # t, y, x, mask
        .sum(axis=(1, 2)) # t, mask
        )

    print(date[i1])
    print(sic_part)
    sic_time[i1:i2, :] = sic_part.compute()

print(sic_time.shape)
                                       


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
sic_time = sic_time.filled(-1e20)
np.savez(out_file + 'sat_ice_time.npz', sic_time=sic_time, date=date)

