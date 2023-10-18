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


domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


# Define start and end date for decade mean
if 1:
    st_date = dt.datetime(1990, 1, 1)
    en_date = dt.datetime(2000, 1, 1)
else:
    st_date = dt.datetime(2040, 1, 1)
    en_date = dt.datetime(2050, 1, 1)


# In[ ]:


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
v_map['lat'] = 'lat'
v_map['lon'] = 'lon'
v_map['depth'] = 'lev'
v_map['time'] = 'time'
v_map['temp'] = 'thetao'
v_map['sal'] = 'so' 
v_map['evap'] = 'evspsbl'
v_map['prec'] = 'pr'

    
with nc.Dataset(flist_evap[0], 'r') as nc_fid:
    evap = nc_fid.variables[v_map['evap']][0, ...]
with nc.Dataset(flist_prec[0], 'r') as nc_fid:
    prec = nc_fid.variables[v_map['prec']][0, ...]
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]


# Subset data

# In[ ]:


# Meshgrid lat and lon

lon, lat = np.meshgrid(lon, lat)


# In[ ]:


lon_bnds, lat_bnds = (0, 360), (60, 90)
yi1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])

lat = lat[yi1:, :]
lon = lon[yi1:, :]
evap = evap[yi1:, :]
prec = prec[yi1:, :]


# Time slice. Notes dates are 360 day years so use cftime

# In[ ]:


# change this to decrease resolution but decrease run time
sub = 1

      
lat = lat[::sub, ::sub]
lon = lon[::sub, ::sub]
evap = evap[::sub, ::sub]
prec = prec[::sub, ::sub]
    
print(lon.shape)

date_from = np.zeros((len(flist_evap)), dtype=object)
date_to = np.zeros((len(flist_evap)), dtype=object)
for i in range(len(flist_evap)):
    part = flist_evap[i].split('_')[-1].split('.')[0].split('-')
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
date = np.zeros((len(flist_evap)), dtype=object)
evap_time = np.ma.zeros((12, lon.shape[0], lon.shape[1]))
prec_time = np.ma.zeros((12, lon.shape[0], lon.shape[1]))
for m in range(1, 13):
    for y in yr_uni:
        ind = np.nonzero(yr == y)[0][0] + date_use[0]
        with nc.Dataset(flist_evap[ind], 'r') as nc_fid:
            evap = nc_fid.variables[v_map['evap']][m-1, yi1::sub, ::sub] # time, lev, j, i
            time = nc_fid.variables[v_map['time']][m-1]
        with nc.Dataset(flist_prec[ind], 'r') as nc_fid:
            prec = nc_fid.variables[v_map['prec']][m-1, yi1::sub, ::sub]
    
        evap = np.ma.masked_where((evap==1e20), evap)
        prec = np.ma.masked_where((prec==1e20), prec)       

        evap_time[m-1, :, :] = evap_time[m-1, :, :] + evap 
        prec_time[m-1, :, :] = prec_time[m-1, :, :] + prec 
        date[i] = cftime.num2date(time, ref, calendar='360_day')
        print(date[i])

evap_time = evap_time / len(yr_uni)
prec_time = prec_time / len(yr_uni)
evap_time = evap_time.filled(-1e20)
prec_time = prec_time.filled(-1e20)
np.savez(out_file + 'pr_ev_mn_' + str_year + '.npz', evap_map=evap_time, prec_map=prec_time, lat=lat, lon=lon)


