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


var = 'evspsbl' # thetao, so, uo, vo, evap, siage, sivol, prec, siu, siv, 
i_o = 'A' # SI or O for sea ice or ocean
freq = 'mon' # mon or day
time_s = 'highres-future' # 'highres-future' or 'hist-1950'

def make_path(var, i_o, freq, time_s):
    if 'future' in time_s:
        ddir = 'MOHC'
    else:
        ddir = 'NERC'
    root = '/badc/cmip6/data/CMIP6/HighResMIP/' + ddir + '/HadGEM3-GC31-HH/' + time_s + '/r1i1p1f1/'
    return root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_' + time_s + '_r1i1p1f1_gn_*.nc'

fn_nemo_dat_evap1 = make_path(var, i_o, freq, 'hist-1950')
fn_nemo_dat_prec1 = make_path('pr', i_o, freq, 'hist-1950')

fn_nemo_dat_evap2 = make_path(var, i_o, freq, time_s)
fn_nemo_dat_prec2 = make_path('pr', i_o, freq, time_s)

domain_root = '/badc/cmip6/data/CMIP6/HighResMIP/MOHC/HadGEM3-GC31-HH/hist-1950/r1i1p1f1/fx/'
fn_nemo_dom = domain_root + 'areacella/gn/latest/areacella_fx_HadGEM3-GC31-HH_hist-1950_r1i1p1f1_gn.nc'
fn_mask = domain_root + 'sftlf/gn/latest/sftlf_fx_HadGEM3-GC31-HH_hist-1950_r1i1p1f1_gn.nc'
#fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
#fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
#fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1
now = dt.datetime.now()

flist_evap = sorted(glob.glob(fn_nemo_dat_evap1))
flist_prec = sorted(glob.glob(fn_nemo_dat_prec1))

flist_evap.extend(sorted(glob.glob(fn_nemo_dat_evap2)))
flist_prec.extend(sorted(glob.glob(fn_nemo_dat_prec2)))
               

v_map = {}
v_map['e1t'] = 'e1t'
v_map['e2t'] = 'e2t'
v_map['e3t_0'] = 'e3t_0'
v_map['tmask'] = 'tmask'
v_map['area'] = 'areacella'
v_map['land'] = 'sftlf'

v_map['lat'] = 'lat'
v_map['lon'] = 'lon'
v_map['depth'] = 'lev'
v_map['time'] = 'time'
v_map['temp'] = 'thetao'
v_map['sal'] = 'so' 
v_map['evap'] = 'evspsbl'
v_map['prec'] = 'pr'

    
with nc.Dataset(flist_evap[0], 'r') as nc_fid:
    evap = nc_fid.variables[v_map['evap']][:]
with nc.Dataset(flist_prec[0], 'r') as nc_fid:
    prec = nc_fid.variables[v_map['prec']][:]
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]

with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    area = nc_fid.variables[v_map['area']][:] # y, x

with nc.Dataset(fn_mask, 'r') as nc_fid:
    land_frac = nc_fid.variables[v_map['land']][:] # y, x
mask = land_frac > 0.5
mask = mask[np.newaxis, :, :]
    
lon, lat = np.meshgrid(lon, lat)  

lon_bnds, lat_bnds = (0, 360), (60, 90)
y1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])
y2 = np.max(np.nonzero((lat <= lat_bnds[1]))[0])
x1 = np.min(np.nonzero((lon >= lon_bnds[0]))[1])
x2 = np.max(np.nonzero((lon <= lon_bnds[1]))[1])

lat = lat[y1:y2:sub, x1:x2:sub]
lon = lon[y1:y2:sub, x1:x2:sub]
area = area[y1:y2:sub, x1:x2:sub]
mask = mask[:, y1:y2:sub, x1:x2:sub]

def dask_load_i(fn_evap, fn_prec, tmask, y1, y2, x1, x2, sub):
    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fn_evap]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['evap']][:, y1:y2:sub, x1:x2:sub], shape=evap[:, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_evap = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_evap = ds_evap.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = dask.array.from_array(tmask)
    tmask = tmask.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    tmask = tmask.repeat(ds_evap.shape[0], axis=0)
    ds_evap = dask.array.ma.masked_array(ds_evap, mask=tmask)
    print(ds_evap) # t, z, y, x

    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fn_prec]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['prec']][:, y1:y2:sub, x1:x2:sub], shape=evap[:, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_prec = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_prec = ds_prec.rechunk(chunks={0:1, 1:100, 2:100})#, balance=True)
    ds_prec = dask.array.ma.masked_array(ds_prec, mask=tmask)
    return ds_evap, ds_prec


# In[ ]:


mask_south = np.zeros((lat.shape[0], lat.shape[1], 3), dtype=bool)
mask_south[:, :, 0] = lat < 70 # mask where less
mask_south[:, :, 1] = lat < 75
mask_south[:, :, 2] = lat < 80

    
ref = 'days since 1950-01-01'
date = np.zeros((len(flist_evap) * 12), dtype=object)
c = 0
for i in range(len(flist_evap)):
    with nc.Dataset(flist_evap[i], 'r') as nc_fid:
        time = nc_fid.variables[v_map['time']][:]
    for k in range(12):
        date[c] = cftime.num2date(time, ref, calendar='360_day')[k]
        c = c + 1
    
area = dask.array.from_array(area[np.newaxis, :, :], chunks={0:1, 1:100, 2:100})
mask_south = dask.array.from_array(mask_south[np.newaxis, :, :, :], chunks={0:1, 0:100, 1:100, 2:-1})
area = area.repeat(12, axis=0)
mask_south = mask_south.repeat(12, axis=0)

evap_time = np.ma.zeros((len(flist_evap) * 12, 3))
prec_time = np.ma.zeros((len(flist_prec) * 12, 3))
for i in range(len(flist_evap) // 1):
    i1 = int(i)
    i2 = int((i + 1))
    f1 = int(i * 12)
    f2 = int((i + 1) * 12)
    
    ds_evap, ds_prec = dask_load_i(flist_evap[i1:i2], flist_prec[i1:i2], mask, y1, y2, x1, x2, sub)
    
    # percentage to fraction
    evap_part = (
        dask.array.ma.masked_where(mask_south, 
        (ds_evap * area)
        [:, :, :, np.newaxis].repeat(3, axis=3)) # t, y, x, mask
        .sum(axis=(1, 2)) # t, mask
        )
    prec_part = (
        dask.array.ma.masked_where(mask_south, 
        (ds_prec * area)
        [:, :, :, np.newaxis].repeat(3, axis=3)) # t, y, x, mask
        .sum(axis=(1, 2)) # t, mask
        )
    print(date[f1])
    print(evap_part)
    evap_time[f1:f2, :] = evap_part.compute()
    prec_time[f1:f2, :] = prec_part.compute()
print(evap_time.shape)
                                       


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
evap_time = evap_time.filled(-1e20)
prec_time = prec_time.filled(-1e20)
np.savez(out_file + 'pr_ev_time.npz', evap_time=evap_time, prec_time=prec_time, date=date)

