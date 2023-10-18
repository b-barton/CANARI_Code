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


use_xarray = True
use_dask = False
sub = 1

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

def hc(heat, volume):
    return heat * volume
    
if use_xarray:
    with nc.Dataset(flist_t[0], 'r') as nc_fid:
        lat = nc_fid.variables['latitude'][:]
        lon = nc_fid.variables['longitude'][:]
    
    lon_bnds, lat_bnds = (-180, 180), (-90, 90)

    lat_sel = np.arange(np.min(np.nonzero((lat >= lat_bnds[0]))[0]), np.max(np.nonzero((lat <= lat_bnds[1]))[0]), sub)
    lon_sel = np.arange(np.min(np.nonzero((lon >= lon_bnds[0]))[0]), np.max(np.nonzero((lon <= lon_bnds[1]))[0]), sub)

    my_chunks = {'j':100, 'i':100, 'time':1}
    ds = xr.open_mfdataset(flist_t[:12], chunks=my_chunks, combine='nested', concat_dim='time', parallel=True).rename({"thetao": "temperature", "lev": "z_dim", "i": "x_dim", "j": "y_dim"})
    ds_t = xr.open_mfdataset(flist_s[:12], chunks=my_chunks, combine='nested', concat_dim='time', parallel=True).rename({"so": "salinity", "lev": "z_dim", "i": "x_dim", "j": "y_dim"})
    ds = ds.assign(salinity=ds_t.salinity)

    my_chunks = {'y': 100, 'x': 100}                           
    ds_dom = xr.open_mfdataset(fn_nemo_dom, chunks=my_chunks, combine='nested', parallel=True).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
    
    def _preprocess(ds, lon_bnds, lat_bnds):
        return ds.isel(x=lon_bnds, y=lat_bnds)
    
    partial_func = partial(_preprocess, lon_bnds=lon_sel[:], lat_bnds=lat_sel[:])
    my_chunks = {'y': 100, 'x': 100}                           
    ds_dom = xr.open_mfdataset(fn_nemo_dom, chunks=my_chunks, combine='nested', preprocess=partial_func, parallel=True).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
                                                                                                                                         
    e1t = ds_dom.e1t # y, x
    e2t = ds_dom.e2t
    e3t = ds_dom.e3t_0 # z, y, x

    
    e1t = np.tile(e1t, (e3t.shape[0], 1, 1))
    e2t = np.tile(e2t, (e3t.shape[0], 1, 1))
    volume = e1t * e2t * e3t

    depth_g = np.tile(ds.z_dim.values, (ds.longitude.shape[1], ds.longitude.shape[0], 1)).T
    mask_south = np.zeros((ds.latitude.shape[0], ds.latitude.shape[1], 3), dtype=bool)
    mask_south[:, :, 0] = ds.latitude < 70
    mask_south[:, :, 1] = ds.latitude < 75
    mask_south[:, :, 2] = ds.latitude < 80

    depth_g = xr.DataArray(depth_g, dims=("z_dim", "y_dim", "x_dim")).chunk({'x_dim': 100, 'y_dim': 100})
    volume = xr.DataArray(volume, dims=("z_dim", "y_dim", "x_dim")).expand_dims(dim={"time": 1}).chunk({'x_dim': 100, 'y_dim': 100})
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
    
    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in flist_t]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['temp']][:, :, y1:y2:sub, x1:x2:sub], shape=temp[:, :, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_t = dask.array.concatenate(lazy_arrays[:12], axis=0)
    ds_t = ds_t.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    tmask = dask.array.from_array(tmask)
    tmask = tmask.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    tmask = tmask.repeat(ds_t.shape[0], axis=0)
    ds_t = dask.array.ma.masked_array(ds_t, mask=tmask)
    print(ds_t) # t, z, y, x

    lazy_arrays = [dask.delayed(nc.Dataset)(fn, 'r') for fn in flist_s]
    lazy_arrays = [dask.array.from_delayed(
                        x.variables[v_map['sal']][:, :, y1:y2:sub, x1:x2:sub], shape=temp[:, :, y1:y2:sub, x1:x2:sub].shape, dtype=np.float64) 
                        for x in lazy_arrays]
    ds_s = dask.array.concatenate(lazy_arrays[:12], axis=0)
    ds_s = ds_s.rechunk(chunks={0:1, 1:-1, 2:100, 3:100})#, balance=True)
    ds_s = dask.array.ma.masked_array(ds_s, mask=tmask)



# Save data

# In[ ]:


print('Runtime:', dt.datetime.now() - now)
print(heat_time)
heat_time = heat_time.filled(-1e20)
np.savez(out_file + 'test_heat_content.npz', heat_time=heat_time, date=date)

