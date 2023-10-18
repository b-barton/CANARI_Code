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


# In[ ]:


var = 'siflfwbot' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
i_o = 'SI' # SI or O for sea ice or ocean
freq = 'mon' # mon or day
time_s = 'highres-future' # 'highres-future' or 'hist-1950'

def make_path(var, i_o, freq, time_s):
    if 'future' in time_s:
        ddir = 'MOHC'
    else:
        ddir = 'NERC'
    root = '/badc/cmip6/data/CMIP6/HighResMIP/' + ddir + '/HadGEM3-GC31-HH/' + time_s + '/r1i1p1f1/'
    return root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_' + time_s + '_r1i1p1f1_gn_*.nc'

fn_nemo_dat_s1 = make_path(var, i_o, freq, 'hist-1950')
fn_nemo_dat_s2 = make_path(var, i_o, freq, time_s)
fn_nemo_dat_t1 = make_path('thetao', 'O', freq, 'hist-1950')

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1
now = dt.datetime.now()

flist_s = sorted(glob.glob(fn_nemo_dat_s1))
flist_s.extend(sorted(glob.glob(fn_nemo_dat_s2)))
flist_t = sorted(glob.glob(fn_nemo_dat_t1))


v_map = {}
v_map['e1t'] = 'e1t'
v_map['e2t'] = 'e2t'
v_map['e3t_0'] = 'e3t_0'
v_map['tmask'] = 'tmask'
v_map['lat'] = 'latitude'
v_map['lon'] = 'longitude'
v_map['depth'] = 'lev'
v_map['time'] = 'time'
v_map['temp'] = 'thetao'
v_map['sal'] = 'so' 
v_map['melt'] = 'siflfwbot'

with nc.Dataset(flist_s[0], 'r') as nc_fid:
    melt = nc_fid.variables[v_map['melt']][:]

with nc.Dataset(flist_t[0], 'r') as nc_fid:
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]

with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, ...]
    e3t = nc_fid.variables[v_map['e3t_0']][0, ...] # t, z, y, x
    tmask = nc_fid.variables[v_map['tmask']][:, :, 1:-1, 1:-1]


lon_bnds, lat_bnds = (-180, 180), (60, 90)
y1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])
y2 = np.max(np.nonzero((lat <= lat_bnds[1]))[0])
x1 = np.min(np.nonzero((lon >= lon_bnds[0]))[1])
x2 = np.max(np.nonzero((lon <= lon_bnds[1]))[1])

lat = lat[y1:y2:sub, x1:x2:sub]
lon = lon[y1:y2:sub, x1:x2:sub]
tmask = (tmask[:, 0, y1:y2:sub, x1:x2:sub] == 0)
e1t = e1t[y1:y2:sub, x1:x2:sub]
e2t = e2t[y1:y2:sub, x1:x2:sub]
e3t = e3t[:, y1:y2:sub, x1:x2:sub]

def dask_load_i(fn_sic, tmask, y1, y2, x1, x2, sub):
    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fn_sic]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['melt']][:, y1:y2:sub, x1:x2:sub], shape=melt[:, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_s = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_s = ds_s.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = dask.array.from_array(tmask)
    tmask = tmask.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = tmask.repeat(ds_s.shape[0], axis=0)
    print(ds_s.shape, tmask.shape)
    ds_s = dask.array.ma.masked_array(ds_s, mask=tmask)
    print(ds_s) # t, z, y, x
    
    return ds_s


# In[ ]:


area = e1t * e2t
    
mask_south = np.zeros((lat.shape[0], lat.shape[1], 3), dtype=bool)
mask_south[:, :, 0] = lat < 70 # mask where less
mask_south[:, :, 1] = lat < 75
mask_south[:, :, 2] = lat < 80

    
ref = 'days since 1950-01-01'
date = np.zeros((len(flist_s) * 12), dtype=object)
c = 0
for i in range(len(flist_s)):
    with nc.Dataset(flist_s[i], 'r') as nc_fid:
        time = nc_fid.variables[v_map['time']][:]
    for k in range(12):
        date[c] = cftime.num2date(time, ref, calendar='360_day')[k]
        c = c + 1
    
area = dask.array.from_array(area[np.newaxis, :, :], chunks={0:1, 1:100, 2:100})
mask_south = dask.array.from_array(mask_south[np.newaxis, :, :, :], chunks={0:1, 0:100, 1:100, 2:-1})
area = area.repeat(12, axis=0)
mask_south = mask_south.repeat(12, axis=0)

melt_time = np.ma.zeros((len(flist_s) * 12, 3))
for i in range(len(flist_s) // 1):
    i1 = int(i)
    i2 = int((i + 1))
    f1 = int(i * 12)
    f2 = int((i + 1) * 12)
    
    ds_s = dask_load_i(flist_s[i1:i2], tmask, y1, y2, x1, x2, sub)
    
    # percentage to fraction
    melt_part = (
        dask.array.ma.masked_where(mask_south, 
        ((ds_s / 100) * area)
        [:, :, :, np.newaxis].repeat(3, axis=3)) # t, y, x, mask
        .sum(axis=(1, 2)) # t, mask
        )
    print(date[f1])
    print(melt_part)
    melt_time[f1:f2, :] = melt_part.compute()
    melt_time[f1:f2, :] = melt_part.compute()
print(melt_time.shape)
                                       


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
melt_time = melt_time.filled(-1e20)
np.savez(out_file + 'melt_time.npz', melt_time=melt_time, date=date)

