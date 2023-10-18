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

out_file = './Processed_test/'


# In[ ]:


use_coast = True
use_xarray = 0#True
use_dask = 0#True
# change this to decrease resolution but decrease run time
sub = 1

now = dt.datetime.now()

flist_t = sorted(glob.glob(fn_nemo_dat_t1))
flist_s = sorted(glob.glob(fn_nemo_dat_s1))

flist_t.extend(sorted(glob.glob(fn_nemo_dat_t2)))
flist_t = flist_t[:12]
flist_s.extend(sorted(glob.glob(fn_nemo_dat_s2)))
flist_s = flist_s[:12]

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
    
if use_coast:
    nemo_t = coast.Gridded(fn_data = flist_t[:12], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
    nemo_s = coast.Gridded(fn_data = flist_s[:12], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
    nemo_t.dataset['salinity'] = nemo_s.dataset.salinity

    print(nemo_t.dataset.longitude.shape)
    
elif use_xarray:
    with nc.Dataset(flist_t[0], 'r') as nc_fid:
        lat = nc_fid.variables['latitude'][:]
        lon = nc_fid.variables['longitude'][:]
    
    def _preprocess(ds, lon_bnds, lat_bnds):
        return ds.isel(i=lon_bnds, j=lat_bnds)
    
    lon_bnds, lat_bnds = (-180, 180), (60, 90)
    #lon_bnds, lat_bnds = (0, 180), (60, 90)
    lat_sel = np.arange(np.min(np.nonzero((lat >= lat_bnds[0]))[0]), np.max(np.nonzero((lat <= lat_bnds[1]))[0]), sub)
    lon_sel = np.arange(np.min(np.nonzero((lon >= lon_bnds[0]))[0]), np.max(np.nonzero((lon <= lon_bnds[1]))[0]), sub)
    partial_func = partial(_preprocess, lon_bnds=lon_sel, lat_bnds=lat_sel)

    my_chunks = {'j':50, 'i':50, 'time':1}
    ds = xr.open_mfdataset(flist_t[:12], chunks=my_chunks, combine='nested', concat_dim='time', preprocess=partial_func, parallel=True).rename({"thetao": "temperature", "lev": "z_dim", "i": "x_dim", "j": "y_dim"})
    ds_t = xr.open_mfdataset(flist_s[:12], chunks=my_chunks, combine='nested', concat_dim='time', preprocess=partial_func, parallel=True).rename({"so": "salinity", "lev": "z_dim", "i": "x_dim", "j": "y_dim"})
    ds = ds.assign(salinity=ds_t.salinity)
    
elif use_dask:
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

    #def _preprocess(ds, lon_bnds, lat_bnds):
    #    return ds.vindex([[:, :, lat_bnds, lon_bnds]])

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

else: 
    with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
        e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
        e2t = nc_fid.variables[v_map['e2t']][0, ...]
        e3t = nc_fid.variables[v_map['e3t_0']][0, ...] # t, z, y, x
        tmask = nc_fid.variables[v_map['tmask']][0, :, 1:-1, 1:-1]
        
    with nc.Dataset(flist_t[0], 'r') as nc_fid:
        lat = nc_fid.variables[v_map['lat']][:]
        lon = nc_fid.variables[v_map['lon']][:]
        depth = nc_fid.variables[v_map['depth']][:]
        temp = nc_fid.variables[v_map['temp']][0, ...]


    lon_bnds, lat_bnds = (-180, 180), (60, 90)
    lat_sel = np.arange(np.min(np.nonzero((lat >= lat_bnds[0]))[0]), np.max(np.nonzero((lat <= lat_bnds[1]))[0]), sub)
    lon_sel = np.arange(np.min(np.nonzero((lon >= lon_bnds[0]))[0]), np.max(np.nonzero((lon <= lon_bnds[1]))[0]), sub)
    
    yi1 = lat_sel[0]
    
    temp = np.ma.masked_where((temp==1e20) | (tmask==1), temp)
    mask = temp.mask


# Subset data

# In[ ]:


#print(nemo_t.dataset)


# In[ ]:


if use_coast:
    yi1 = 2800
    yi2 = nemo_t.dataset.longitude.shape[0]
    ind_y = np.arange(yi1, yi2).astype(int)
    #print(yi2, ind_y)
    nemo_t_subset = nemo_t.isel(y_dim=ind_y)
    print(nemo_t_subset.dataset)
elif use_xarray | use_dask:
    pass
else:
    lat = lat[yi1:, :]
    lon = lon[yi1:, :]
    mask = mask[:, yi1:, :]


# Time slice. Notes dates are 360 day years so use cftime

# In[ ]:


# Does not work
#st_date = cftime.datetime(2050, 12, 1)
#en_date = cftime.datetime(2051, 1, 1) 
#nemo_t_subset = nemo_t_subset.time_slice(st_date, en_date)


# In[ ]:


#nemo_t_subset.construct_density(eos='EOS10')#, pot_dens=True)

def calc_rho(sp, tp, depth, lon, lat):
    pres = sw.p_from_z(depth * -1, lat)
    sa = sw.SA_from_SP(sp, pres, lon, lat)
    ct = sw.CT_from_pt(sa, tp)
    rho = sw.rho(sa, ct, pres)
    return rho, ct

def calc_heat(rho, ct=[]):
    if len(ct) == 0:
        rho, ct = rho
    cap = 3991.86795711963 # 3985 # J kg-1 K-1
    #cap = sw.cp0 # use with conservative temp
    return ct * rho * cap


if use_coast:
    ds_dom = xr.open_dataset(fn_nemo_dom).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
    #e1t = ds_dom.e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
    #e2t = ds_dom.e2t[yi1+1:-1:sub, 1:-1:sub]
    #e3t = ds_dom.e3t_0[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x
    ind_y = np.arange(yi1+1, yi2-1, sub).astype(int)
    ind_x = np.arange(1, nemo_t_subset.dataset.longitude.shape[1]-1, sub).astype(int)
    e1t = ds_dom.e1t.isel(y_dim=ind_y, x_dim=ind_x) # y, x
    e2t = ds_dom.e2t.isel(y_dim=ind_y, x_dim=ind_x)
    e3t = ds_dom.e3t_0.isel(y_dim=ind_y, x_dim=ind_x) # z, y, x   
elif use_xarray:
    def _preprocess(ds, lon_bnds, lat_bnds):
        return ds.isel(x=lon_bnds, y=lat_bnds)
    
    partial_func = partial(_preprocess, lon_bnds=lon_sel+1, lat_bnds=lat_sel+1)
    my_chunks = {'y': 100, 'x': 100}                           
    ds_dom = xr.open_mfdataset(fn_nemo_dom, chunks=my_chunks, combine='nested', preprocess=partial_func, parallel=True).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
    e1t = ds_dom.e1t # y, x
    e2t = ds_dom.e2t
    e3t = ds_dom.e3t_0 # z, y, x
elif use_dask:
    pass
else:
    e1t = e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
    e2t = e2t[yi1+1:-1:sub, 1:-1:sub]
    e3t = e3t[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x
    
e1t = np.tile(e1t, (e3t.shape[0], 1, 1))
e2t = np.tile(e2t, (e3t.shape[0], 1, 1))
print(e3t.shape, e1t.shape)
volume = e1t * e2t * e3t

      
if use_coast:
    ind_y = np.arange(0, nemo_t_subset.dataset.longitude.shape[0], sub).astype(int)
    ind_x = np.arange(0, nemo_t_subset.dataset.longitude.shape[1], sub).astype(int)
    nemo_t_subset = nemo_t_subset.isel(y_dim=ind_y, x_dim=ind_x)
    lat = nemo_t_subset.dataset.latitude.values
    lon = nemo_t_subset.dataset.longitude.values
    #lat = nemo_t_subset.dataset.latitude.values[::sub, ::sub]
    #lon = nemo_t_subset.dataset.longitude.values[::sub, ::sub]
    depth = nemo_t_subset.dataset.depth_0.values[:]
elif use_xarray:
    #ind_y = np.arange(0, ds.longitude.shape[0], sub).astype(int)
    #ind_x = np.arange(0, ds.longitude.shape[1], sub).astype(int)
    #nemo_t_subset = ds.isel(j=ind_y, i=ind_x)
    lat = ds.latitude.values
    lon = ds.longitude.values
    depth = ds.z_dim.values
elif use_dask:
    pass
else:
    lat = lat[::sub, ::sub]
    lon = lon[::sub, ::sub]
    mask = mask[:, ::sub, ::sub]

depth_g = np.tile(depth, (lon.shape[1], lon.shape[0], 1)).T
mask_south = np.zeros((lat.shape[0], lat.shape[1], 3), dtype=bool)
mask_south[:, :, 0] = lat < 70
mask_south[:, :, 1] = lat < 75
mask_south[:, :, 2] = lat < 80

if use_coast:
    heat_time = np.ma.zeros((nemo_t_subset.dataset.t_dim.shape[0]))
    date = np.zeros((nemo_t_subset.dataset.t_dim.shape[0]), dtype=object)
    for i in range(nemo_t_subset.dataset.t_dim.shape[0]):
        temp = nemo_t_subset.dataset.temperature.isel(t_dim=i).to_masked_array() # time, lev, j, i
        sal = nemo_t_subset.dataset.salinity.isel(t_dim=i).to_masked_array()
        #temp = nemo_t_subset.dataset.temperature[i, :, ::sub, ::sub].to_masked_array() # time, lev, j, i
        #sal = nemo_t_subset.dataset.salinity[i, :, ::sub, ::sub].to_masked_array()
        rho, ct = calc_rho(sal, temp, depth_g, lon, lat)
        print(rho.size, volume.shape)
    
        heat_cont = np.ma.sum(calc_heat(rho, ct) * volume, axis=0) # vertically integrated
        heat_pole = np.ma.masked_where(mask_south[:, :, 0], heat_cont)
        heat_time[i] = np.ma.sum(heat_pole)
        date[i] = nemo_t_subset.dataset.time.isel(t_dim=i).values
        print(date[i])
        
elif use_xarray:
    def hc(heat, volume):
        return heat * volume
    
    heat_time = dask.array.zeros((ds.time.shape[0], 3))
    date = np.zeros((ds.time.shape[0]), dtype=object)

    if 1:
        depth_g = xr.DataArray(depth_g, dims=("z_dim", "y_dim", "x_dim")).chunk({'x_dim': 100, 'y_dim': 100})
        volume = xr.DataArray(volume, dims=("z_dim", "y_dim", "x_dim")).chunk({'x_dim': 100, 'y_dim': 100})
        mask_south = xr.DataArray(np.invert(mask_south), dims=("y_dim", "x_dim", "band")).chunk({'x_dim': 100, 'y_dim': 100})
        date = ds.time.values
        heat_time = (
            hc(calc_heat(calc_rho(
            ds.salinity, ds.temperature, depth_g, 
            ds.longitude.values, ds.latitude.values)), 
            volume)
            .sum(dim='z_dim')
            .expand_dims(dim={"band": 3})
            .where(mask_south)
            .sum(dim=['x_dim', 'y_dim'])
            )
        print(heat_time.shape)
    else:
        for i in range(ds.time.shape[0]):
            rho, ct = calc_rho(ds.salinity.isel(time=i), ds.temperature.isel(time=i), depth_g, ds.longitude.values, ds.latitude.values)
            heat = calc_heat(rho, ct)
            print(heat.shape, volume.shape)
            heat_v = hc(heat, volume)
            heat_cont = dask.array.sum(heat_v, axis=0) # vertically integrated
            heat_pole = dask.array.ma.masked_where(mask_south[:, :, 0], heat_cont)
            heat_time[i] = dask.array.sum(heat_pole)
            date[i] = ds.time.isel(time=i).values
            print(date[i]) 

elif use_dask:
    def hc(heat, volume):
        return heat * volume

    ref = 'days since 1950-01-01'
    date = np.zeros((len(flist_t)), dtype=object)
    for i in range(len(flist_t)):
        with nc.Dataset(flist_t[i], 'r') as nc_fid:
            time = nc_fid.variables[v_map['time']][:]
        date[i] = cftime.num2date(time, ref, calendar='360_day')[0]
        
    lat_g = dask.array.from_array(lat[np.newaxis, np.newaxis, :, :], chunks={0:1, 0:-1, 1:100, 2:100})
    lon_g = dask.array.from_array(lon[np.newaxis, np.newaxis, :, :], chunks={0:1, 0:-1, 1:100, 2:100})
    depth_g = dask.array.from_array(depth_g[np.newaxis, :, :, :], chunks={0:1, 0:-1, 1:100, 2:100})
    volume = dask.array.from_array(volume[np.newaxis, :, :, :], chunks={0:1, 0:-1, 1:100, 2:100})
    mask_south = dask.array.from_array(mask_south[np.newaxis, :, :, :], chunks={0:1, 0:100, 1:100, 2:-1})
    lat_g = lat_g.repeat(depth_g.shape[1], axis=1).repeat(12, axis=0)
    lon_g = lon_g.repeat(depth_g.shape[1], axis=1).repeat(12, axis=0)
    depth_g = depth_g.repeat(12, axis=0)
    volume = volume.repeat(12, axis=0)
    mask_south = mask_south.repeat(12, axis=0)

    heat_time = np.ma.zeros((len(flist_t), 3))
    for i in range(len(flist_t) // 12):
        i1 = int(i * 12)
        i2 = int((i + 1) * 12)
        
        ds_t, ds_s = dask_load_ts(flist_t[i1:i2], flist_s[i1:i2], tmask, y1, y2, x1, x2, sub)
        
        heat_part = (
            dask.array.ma.masked_where(mask_south, 
            hc(
            calc_heat(
            calc_rho(
            ds_s, ds_t, depth_g, 
            lon_g, lat_g)), 
            volume)        # t, z, y, x
            .sum(axis=1)   # t, y, x
            [:, :, :, np.newaxis].repeat(3, axis=3)) # t, y, x, mask
            .sum(axis=(1, 2)) # t, mask
            )
        print(date[i1])
        print(heat_part)
        heat_time[i1:i2, :] = heat_part.compute()
    print(heat_time.shape)
    
else:
    heat_time = np.ma.zeros((len(flist_t), 3))
    ref = 'days since 1950-01-01'
    date = np.zeros((len(flist_t)), dtype=object)
    for i in range(12):#len(flist_t)):
        with nc.Dataset(flist_t[i], 'r') as nc_fid:
            temp = nc_fid.variables[v_map['temp']][0, :, yi1::sub, ::sub] # time, lev, j, i
            time = nc_fid.variables[v_map['time']][:]
        with nc.Dataset(flist_s[i], 'r') as nc_fid:
            sal = nc_fid.variables[v_map['sal']][0, :, yi1::sub, ::sub]
        
        temp = np.ma.masked_where((temp==1e20), temp)
        sal = np.ma.masked_where((sal==1e20), sal)       
        rho, ct = calc_rho(sal, temp, depth_g, lon, lat)
    
        heat_cont = np.ma.sum(calc_heat(rho, ct) * volume, axis=0) # vertically integrated
        for j in range(mask_south.shape[2]):
            heat_pole = np.ma.masked_where(mask_south[:, :, j], heat_cont)
            heat_time[i, j] = np.ma.sum(heat_pole) 
        date[i] = cftime.num2date(time, ref, calendar='360_day')[0]
        print(date[i])

                                         


# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
print(heat_time)
if use_dask:
    heat_time = heat_time.filled(-1e20)
np.savez(out_file + 'heat_content.npz', heat_time=heat_time, date=date)

