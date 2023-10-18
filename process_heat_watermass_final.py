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
from functools import partial
import dask


# In[ ]:


var = 'thetao' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
i_o = 'O' # SI or O for sea ice or ocean
freq = 'mon' # mon or day
time_s = 'highres-future' # 'highres-future' or 'hist-1950'

def make_path(var, i_o, freq, time_s):
    if 'future' in time_s:
        ddir = 'MOHC'
    else:
        ddir = 'NERC'
    root = '/badc/cmip6/data/CMIP6/HighResMIP/' + ddir + '/HadGEM3-GC31-HH/' + time_s + '/r1i1p1f1/'
    return root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_' + time_s + '_r1i1p1f1_gn_*.nc'

fn_nemo_dat_t1 = make_path('thetao', i_o, freq, 'hist-1950')
fn_nemo_dat_s1 = make_path('so', i_o, freq, 'hist-1950')
fn_nemo_dat_t2 = make_path('thetao', i_o, freq, time_s)
fn_nemo_dat_s2 = make_path('so', i_o, freq, time_s)


domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


# Define start and end date for decade mean

st_date = dt.datetime(1950, 1, 1)
en_date = dt.datetime(1960, 1, 1)


# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1

# Flag for Atlantic Water of Pacific Water
do_aw = 1

now = dt.datetime.now()

flist_t = sorted(glob.glob(fn_nemo_dat_t1))
flist_s = sorted(glob.glob(fn_nemo_dat_s1))

flist_t.extend(sorted(glob.glob(fn_nemo_dat_t2)))
flist_s.extend(sorted(glob.glob(fn_nemo_dat_s2)))

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
    

with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, ...]
    e3t = nc_fid.variables[v_map['e3t_0']][0, ...] # t, z, y, x
    tmask = nc_fid.variables[v_map['tmask']][:, :, 1:-1, 1:-1]
    
with nc.Dataset(flist_t[0], 'r') as nc_fid:
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]
    depth = nc_fid.variables[v_map['depth']][:]
    temp = nc_fid.variables[v_map['temp']][:] # t, z, y, x


lon_bnds, lat_bnds = (-180, 180), (60, 90)
y1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])
y2 = np.max(np.nonzero((lat <= lat_bnds[1]))[0])
x1 = np.min(np.nonzero((lon >= lon_bnds[0]))[0])
x2 = np.max(np.nonzero((lon <= lon_bnds[1]))[0])

lat = lat[y1:y2:sub, x1:x2:sub]
lon = lon[y1:y2:sub, x1:x2:sub]
tmask = (temp[:, :, y1:y2:sub, x1:x2:sub] == 1e20) | (tmask[:, :, y1:y2:sub, x1:x2:sub] == 0)
e1t = e1t[y1:y2:sub, x1:x2:sub]
e2t = e2t[y1:y2:sub, x1:x2:sub]
e3t = e3t[:, y1:y2:sub, x1:x2:sub]

def dask_load_ts(fnt, fns, tmask, y1, y2, x1, x2, sub):
    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fnt]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['temp']][:, :, y1:y2:sub, x1:x2:sub], shape=temp[:, :, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_t = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_t = ds_t.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    tmask = dask.array.from_array(tmask)
    tmask = tmask.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    tmask = tmask.repeat(ds_t.shape[0], axis=0)
    ds_t = dask.array.ma.masked_array(ds_t, mask=tmask)
    print(ds_t) # t, z, y, x

    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in fns]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['sal']][:, :, y1:y2:sub, x1:x2:sub], shape=temp[:, :, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_s = dask.array.concatenate(lazy_arrays[:], axis=0)
    ds_s = ds_s.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    ds_s = dask.array.ma.masked_array(ds_s, mask=tmask)
    return ds_t, ds_s



# In[ ]:


def calc_rho(sp, tp, depth, lon, lat, zero=False):
    pres = sw.p_from_z(depth * -1, lat)
    sa = sw.SA_from_SP(sp, pres, lon, lat)
    ct = sw.CT_from_pt(sa, tp)
    if zero:
        pres = 0
    rho = sw.rho(sa, ct, pres)
    return rho, ct

def calc_heat(rho, ct=[]):
    if len(ct) == 0:
        rho, ct = rho
    cap = 3991.86795711963 # 3985 # J kg-1 K-1
    #cap = sw.cp0 # use with conservative temp
    return ct * rho * cap

def hc(heat, volume):
    return heat * volume
    
e1t = np.tile(e1t, (e3t.shape[0], 1, 1))
e2t = np.tile(e2t, (e3t.shape[0], 1, 1))
print(e3t.shape, e1t.shape)
volume = e1t * e2t * e3t

depth_g = np.tile(depth, (lon.shape[1], lon.shape[0], 1)).T
mask_south = np.zeros((lat.shape[0], lat.shape[1]), dtype=bool)
mask_water = np.zeros((lat.shape[0], lat.shape[1]), dtype=bool)
mask_south[:, :] = lat < 70 # mask where less


ref = 'days since 1950-01-01'
date = np.zeros((len(flist_t)), dtype=object)
for i in range(len(flist_t)):
    with nc.Dataset(flist_t[i], 'r') as nc_fid:
        time = nc_fid.variables[v_map['time']][:]
    date[i] = cftime.num2date(time, ref, calendar='360_day')[0]

date_from = np.zeros((len(flist_t)), dtype=object)
date_to = np.zeros((len(flist_t)), dtype=object)
for i in range(len(flist_t)):
    part = flist_t[i].split('_')[-1].split('.')[0].split('-')
    date_from[i] = dt.datetime.strptime(part[0], '%Y%m')
    date_to[i] = dt.datetime.strptime(part[1], '%Y%m')

date_use = np.nonzero((date_from >= st_date) & (date_to < en_date))[0]
date_from = date_from[date_use]
str_year = str(st_date.year) + '-' + str(en_date.year)
flist_t = flist_t[date_use[0]:date_use[-1]]
flist_s = flist_s[date_use[0]:date_use[-1]]
    
lat_g = dask.array.from_array(lat[np.newaxis, np.newaxis, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
lon_g = dask.array.from_array(lon[np.newaxis, np.newaxis, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
depth_g = dask.array.from_array(depth_g[np.newaxis, :, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
volume = dask.array.from_array(volume[np.newaxis, :, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
mask_south = dask.array.from_array(mask_south[np.newaxis, np.newaxis, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
mask_water = dask.array.from_array(mask_water[np.newaxis, np.newaxis, :, :], chunks={0:1, 1:-1, 2:100, 3:100})
lat_g = lat_g.repeat(depth_g.shape[1], axis=1).repeat(1, axis=0)
lon_g = lon_g.repeat(depth_g.shape[1], axis=1).repeat(1, axis=0)
depth_g = depth_g.repeat(1, axis=0)
volume = volume.repeat(1, axis=0)
mask_south = mask_south.repeat(depth_g.shape[1], axis=1).repeat(1, axis=0)
mask_water = mask_water.repeat(depth_g.shape[1], axis=1).repeat(1, axis=0)

heat_time = np.ma.zeros((len(flist_t)))
for i in range(len(flist_t) // 1):
    i1 = int(i * 1)
    i2 = int((i + 1) * 1)
    
    ds_t, ds_s = dask_load_ts(flist_t[i1:i2], flist_s[i1:i2], tmask, y1, y2, x1, x2, sub)
    
    sig, _ = calc_rho(ds_s, ds_t, depth_g, lon_g, lat_g, zero=True)
    sig = sig - 1000
    if do_aw:
        mask_water[:, :, :, :] = np.invert((sig > 27) & (sig <= 29)) # AW Zhong et al 2019
    else:
        mask_water[:, :, :, :] = np.invert((sig >= 23.2) & (sig <= 27)) # PW # PWW 26 > sig > 27 Zhong et al 2019, PSW 23.2 > sig > 25.2 MacKinnon et al 2021
    mask_water = (mask_water | mask_south)
    
    heat_part = (
        dask.array.ma.masked_where(mask_water,
        hc(
        calc_heat(
        calc_rho(
        ds_s, ds_t, depth_g, 
        lon_g, lat_g)), 
        volume))        # t, z, y, x
        .sum(axis=(1, 2, 3)) # t
        )
    print(date[i1])
    print(heat_part)
    heat_time[i1:i2] = heat_part.compute()
print(heat_time.shape)
    
                          


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
print(heat_time)
heat_time = heat_time.filled(-1e20)
if do_aw:
    np.savez(out_file + 'heat_content_aw_' + str_year + '.npz', heat_time=heat_time, date=date)
else:
    np.savez(out_file + 'heat_content_pw_' + str_year + '.npz', heat_time=heat_time, date=date)

