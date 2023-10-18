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


# In[ ]:


var = 'friver' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
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

fn_nemo_dat_s1 = make_path('friver', i_o, freq, 'hist-1950')
fn_nemo_dat_s2 = make_path('friver', i_o, freq, time_s)


domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


# Define start and end date for decade mean
if 0:
    st_date = dt.datetime(1990, 1, 1)
    en_date = dt.datetime(2000, 1, 1)
else:
    st_date = dt.datetime(2040, 1, 1)
    en_date = dt.datetime(2050, 1, 1)


# In[ ]:


now = dt.datetime.now()

flist_s = sorted(glob.glob(fn_nemo_dat_s1))
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
v_map['riv'] = 'friver'


with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, ...]
    e3t = nc_fid.variables[v_map['e3t_0']][0, ...] # t, z, y, x
    tmask = nc_fid.variables[v_map['tmask']][0, :, 1:-1, 1:-1]
    
with nc.Dataset(flist_s[0], 'r') as nc_fid:
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]
    riv = nc_fid.variables[v_map['riv']][0, ...]

riv = np.ma.masked_where((riv==1e20) | (tmask[0, :, :]==1), riv)
mask = riv.mask


# Subset data

# In[ ]:


yi1 = 2800

lat = lat[yi1:, :]
lon = lon[yi1:, :]
mask = mask[yi1:, :]


# Time slice. Notes dates are 360 day years so use cftime

# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1


e1t = e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
e2t = e2t[yi1+1:-1:sub, 1:-1:sub]
e3t = e3t[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x
    
e1t = np.tile(e1t, (e3t.shape[0], 1, 1))
e2t = np.tile(e2t, (e3t.shape[0], 1, 1))
print(e3t.shape, e1t.shape)
volume = e1t * e2t * e3t

      
lat = lat[::sub, ::sub]
lon = lon[::sub, ::sub]
mask = mask[::sub, ::sub]
    
#depth_g = np.tile(depth, (lon.shape[1], lon.shape[0], 1)).T

date_from = np.zeros((len(flist_s)), dtype=object)
date_to = np.zeros((len(flist_s)), dtype=object)
for i in range(len(flist_s)):
    part = flist_s[i].split('_')[-1].split('.')[0].split('-')
    date_from[i] = dt.datetime.strptime(part[0], '%Y%m')
    date_to[i] = dt.datetime.strptime(part[1], '%Y%m')


date_use = np.nonzero((date_from >= st_date) & (date_to < en_date))[0]
date_from = date_from[date_use]
str_year = str(st_date.year) + '-' + str(en_date.year)

# output a decacal monthly mean
mn = np.array([t.month for t in date_from])
yr = np.array([t.year for t in date_from])
yr_uni = np.unique(yr)


ref = 'days since 1950-01-01'
date = np.zeros((len(flist_s)), dtype=object)
riv_time = np.ma.zeros((12, lon.shape[0], lon.shape[1]))
for m in range(1, 13, 3):
    for y in yr_uni:
        ind = np.nonzero((mn == m) & (yr == y))[0][0] + date_use[0]
        with nc.Dataset(flist_s[ind], 'r') as nc_fid:
            riv = nc_fid.variables[v_map['riv']][:, yi1::sub, ::sub] # time, j, i
            time = nc_fid.variables[v_map['time']][:]

        riv = np.ma.masked_where((riv==1e20), riv)  
        
        for i in range(3):
            riv_time[m-1+i, :, :] = riv_time[m-1+i, :, :] + riv[i, :, :]
            date = cftime.num2date(time, ref, calendar='360_day')[i]
            print(date)

riv_time = riv_time / len(yr_uni)
riv_time = riv_time.filled(-1e20)
np.savez(out_file + 'riv_mn_' + str_year + '.npz', riv_map=riv_time, lat=lat, lon=lon)


