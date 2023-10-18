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


var = 'siconc' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
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

fn_nemo_dat_sic1 = make_path(var, i_o, freq, 'hist-1950')
fn_nemo_dat_sit1 = make_path('sithick', i_o, freq, 'hist-1950')
fn_nemo_dat_t1 = make_path('thetao', 'O', freq, 'hist-1950')

fn_nemo_dat_sic2 = make_path(var, i_o, freq, time_s)
fn_nemo_dat_sit2 = make_path('sithick', i_o, freq, time_s)
fn_nemo_dat_t2 = make_path('thetao', 'O', freq, time_s)

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


use_xarray = False
now = dt.datetime.now()

flist_sic = sorted(glob.glob(fn_nemo_dat_sic1))
flist_sit = sorted(glob.glob(fn_nemo_dat_sit1))
flist_t = sorted(glob.glob(fn_nemo_dat_t1))

flist_sic.extend(sorted(glob.glob(fn_nemo_dat_sic2)))
flist_sit.extend(sorted(glob.glob(fn_nemo_dat_sit2)))
    

if use_xarray:
    nemo_t = coast.Gridded(fn_data = flist_t[:12], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
    nemo_s = coast.Gridded(fn_data = flist_s[:12], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
    nemo_t.dataset['salinity'] = nemo_s.dataset.salinity

    print(nemo_t.dataset.longitude.shape)
else:
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
    v_map['siconc'] = 'siconc'
    v_map['sithick'] = 'sithick'

    with nc.Dataset(flist_sic[0], 'r') as nc_fid:
        siconc = nc_fid.variables[v_map['siconc']][0, ...]
    with nc.Dataset(flist_sit[0], 'r') as nc_fid:
        sithick = nc_fid.variables[v_map['sithick']][0, ...]
        
    with nc.Dataset(flist_t[0], 'r') as nc_fid:
        lat = nc_fid.variables[v_map['lat']][:]
        lon = nc_fid.variables[v_map['lon']][:]

    with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
        e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
        e2t = nc_fid.variables[v_map['e2t']][0, ...]
        e3t = nc_fid.variables[v_map['e3t_0']][0, ...] # t, z, y, x


# Subset data

# In[ ]:


yi1 = 2800

if use_xarray:
    yi2 = nemo_t.dataset.longitude.shape[0]
    ind_y = np.arange(yi1, yi2).astype(int)
    #print(yi2, ind_y)
    nemo_t_subset = nemo_t.isel(y_dim=ind_y)
    print(nemo_t_subset.dataset)
else:
    lat = lat[yi1:, :]
    lon = lon[yi1:, :]
    siconc = siconc[yi1:, :]
    sithick = sithick[yi1:, :]


# Time slice. Notes dates are 360 day years so use cftime

# In[ ]:


# change this to decrease resolution but decrease run time
sub = 10

if use_xarray:
    ds_dom = xr.open_dataset(fn_nemo_dom).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
    #e1t = ds_dom.e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
    #e2t = ds_dom.e2t[yi1+1:-1:sub, 1:-1:sub]
    #e3t = ds_dom.e3t_0[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x
    ind_y = np.arange(yi1+1, yi2-1, sub).astype(int)
    ind_x = np.arange(1, nemo_t_subset.dataset.longitude.shape[1]-1, sub).astype(int)
    e1t = ds_dom.e1t.isel(y_dim=ind_y, x_dim=ind_x) # y, x
    e2t = ds_dom.e2t.isel(y_dim=ind_y, x_dim=ind_x)
    e3t = ds_dom.e3t_0.isel(y_dim=ind_y, x_dim=ind_x) # z, y, x   
else:
    e1t = e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
    e2t = e2t[yi1+1:-1:sub, 1:-1:sub]
    e3t = e3t[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x

area = e1t * e2t
    
if use_xarray:
    ind_y = np.arange(0, nemo_t_subset.dataset.longitude.shape[0], sub).astype(int)
    ind_x = np.arange(0, nemo_t_subset.dataset.longitude.shape[1], sub).astype(int)
    nemo_t_subset = nemo_t_subset.isel(y_dim=ind_y, x_dim=ind_x)
    lat = nemo_t_subset.dataset.latitude.values
    lon = nemo_t_subset.dataset.longitude.values
    #lat = nemo_t_subset.dataset.latitude.values[::sub, ::sub]
    #lon = nemo_t_subset.dataset.longitude.values[::sub, ::sub]
    depth = nemo_t_subset.dataset.depth_0.values[:]
else:
    lat = lat[::sub, ::sub]
    lon = lon[::sub, ::sub]
    siconc = siconc[::sub, ::sub]
    sithick = sithick[::sub, ::sub]


mask_south = np.zeros((lat.shape[0], lat.shape[1], 3), dtype=bool)
mask_south[:, :, 0] = lat < 70
mask_south[:, :, 1] = lat < 75
mask_south[:, :, 2] = lat < 80
    
if use_xarray:
    heat_time = np.ma.zeros((nemo_t_subset.dataset.t_dim.shape[0]))
    date = np.zeros((nemo_t_subset.dataset.t_dim.shape[0]), dtype=object)
    for i in range(nemo_t_subset.dataset.t_dim.shape[0]):
        temp = nemo_t_subset.dataset.temperature.isel(t_dim=i).to_masked_array() # time, lev, j, i
        sal = nemo_t_subset.dataset.salinity.isel(t_dim=i).to_masked_array()
        #temp = nemo_t_subset.dataset.temperature[i, :, ::sub, ::sub].to_masked_array() # time, lev, j, i
        #sal = nemo_t_subset.dataset.salinity[i, :, ::sub, ::sub].to_masked_array()
        rho, ct = calc_rho(sal, temp, depth_g, lon, lat)
    
        heat_cont = np.ma.sum(calc_heat(rho, ct) * volume, axis=0) # vertically integrated
        heat_pole = np.ma.masked_where(mask_south[:, :, 0], heat_cont)
        heat_time[i] = np.ma.sum(heat_pole)
        date[i] = nemo_t_subset.dataset.time.isel(t_dim=i).values
        print(date[i])
else:
    sic_time = np.ma.zeros((len(flist_sic) * 12, 3))
    sit_time = np.ma.zeros((len(flist_sic) * 12, 3))
    ref = 'days since 1950-01-01'
    date = np.zeros((len(flist_sic) * 12), dtype=object)
    c = 0
    for i in range(len(flist_sic)):
        with nc.Dataset(flist_sic[i], 'r') as nc_fid:
            sic = nc_fid.variables[v_map['siconc']][:, yi1::sub, ::sub] # time, j, i
            time = nc_fid.variables[v_map['time']][:]
        with nc.Dataset(flist_sit[i], 'r') as nc_fid:
            sit = nc_fid.variables[v_map['sithick']][:, yi1::sub, ::sub]
        
        sic = np.ma.masked_where((sic==1e20), sic)
        sit = np.ma.masked_where((sit==1e20), sit)       

        for k in range(12):
            for j in range(mask_south.shape[2]):
                sic_pole = np.ma.masked_where(mask_south[:, :, j], sic[k, :, :])
                sit_pole = np.ma.masked_where(mask_south[:, :, j], sit[k, :, :])
                sic_time[c, j] = np.ma.sum(sic_pole) 
                sit_time[c, j] = np.ma.sum(sit_pole)
            date[c] = cftime.num2date(time, ref, calendar='360_day')[k]
            print(date[c])
            c = c + 1

                                         


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
sic_time = sic_time.filled(-1e20)
sit_time = sit_time.filled(-1e20)
np.savez(out_file + 'ice_time.npz', sic_time=sic_time, sit_time=sit_time, date=date)

