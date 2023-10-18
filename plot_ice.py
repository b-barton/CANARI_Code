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


# In[ ]:


#fn_en4 = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/en4_processed.nc'
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

fn_nemo_dat_sic = make_path(var, i_o, freq, time_s)
fn_nemo_dat_sit = make_path('sithick', i_o, freq, time_s)
fn_nemo_dat_t = make_path('thetao', 'O', freq, time_s)

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


flist_sic = sorted(glob.glob(fn_nemo_dat_sic))
flist_sit = sorted(glob.glob(fn_nemo_dat_sit))
flist_t = sorted(glob.glob(fn_nemo_dat_t))

use_xarray = False
if use_xarray:
    nemo_t = coast.Gridded(fn_data = flist_si[-1], fn_domain = fn_nemo_dom, config=fn_config_t_grid)

    print(nemo_t.dataset.longitude.shape)
    with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
        tmask = nc_fid.variables['tmask'][0, :, 1:-1, 1:-1]
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
    
    with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
        tmask = nc_fid.variables[v_map['tmask']][0, :, 1:-1, 1:-1]
        
    with nc.Dataset(flist_sic[0], 'r') as nc_fid:
        lat = nc_fid.variables[v_map['lat']][:]
        lon = nc_fid.variables[v_map['lon']][:]
        siconc = nc_fid.variables[v_map['siconc']][0, ...]
    with nc.Dataset(flist_sit[0], 'r') as nc_fid:
        sithick = nc_fid.variables[v_map['sithick']][0, ...]
        
    with nc.Dataset(flist_t[0], 'r') as nc_fid:
        lat = nc_fid.variables[v_map['lat']][:]
        lon = nc_fid.variables[v_map['lon']][:]
        
    #siconc = np.ma.masked_where((siconc==1e20) | (siconc==0), siconc)
    #mask = siconc.mask
    


# Subset data

# In[ ]:


#print(nemo_t.dataset)


# In[ ]:


yi1 = 2300
    
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


# Does not work
#st_date = cftime.datetime(2050, 12, 1)
#en_date = cftime.datetime(2051, 1, 1) 
#nemo_t_subset = nemo_t_subset.time_slice(st_date, en_date)


# In[ ]:


# change this to decrease resolution but decrease run time
sub = 5
if use_xarray:
    ind_y = np.arange(0, nemo_t_subset.dataset.longitude.shape[0], sub).astype(int)
    ind_x = np.arange(0, nemo_t_subset.dataset.longitude.shape[1], sub).astype(int)
    nemo_t_subset = nemo_t_subset.isel(y_dim=ind_y, x_dim=ind_x)
        
    lat = nemo_t_subset.dataset.latitude.values
    lon = nemo_t_subset.dataset.longitude.values
    lat = np.ma.masked_where((lon==0), lat)
    lon = np.ma.masked_where((lon==0), lon)
    siconc = nemo_t_subset.dataset.siconc.to_masked_array() # time, lev, j, i
else:
    lat = lat[::sub, ::sub]
    lon = lon[::sub, ::sub]
    siconc = siconc[::sub, ::sub]
    sithick = sithick[::sub, ::sub]
print(siconc.shape)


# Plot

# In[ ]:


def set_circle(ax):
  # Compute a circle in axes coordinates, which we can use as a boundary
  # for the map. We can pan/zoom as much as we like - the boundary will be
  # permanently circular.
  theta = np.linspace(0, 2 * np.pi, 100)
  center, radius = [0.5, 0.5], 0.5
  verts = np.vstack([np.sin(theta), np.cos(theta)]).T
  circle = mpath.Path(verts * radius + center)
  ax.set_boundary(circle, transform=ax.transAxes)


# In[ ]:


data_crs = ccrs.PlateCarree()
mrc = ccrs.NorthPolarStereo(central_longitude=0.0)

my_cm = plt.cm.plasma


# In[ ]:


fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_axes([0.03, 0.03, 0.3, 0.85], projection=mrc)
ax2 = fig1.add_axes([0.36, 0.03, 0.3, 0.85], projection=mrc)
ax3 = fig1.add_axes([0.69, 0.03, 0.3, 0.85], projection=mrc)
cax1 = fig1.add_axes([0.03, 0.94, 0.3, 0.02])
cax2 = fig1.add_axes([0.36, 0.94, 0.3, 0.02])
cax3 = fig1.add_axes([0.69, 0.94, 0.3, 0.02])

cs1 = ax1.pcolormesh(lon, lat, siconc[:, :], transform=data_crs, cmap=my_cm)
cs2 = ax2.pcolormesh(lon, lat, sithick[:, :], transform=data_crs, cmap=my_cm)

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
#ax3.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
#ax2.gridlines()
#ax3.gridlines()

ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)
#ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax1)
set_circle(ax2)
#set_circle(ax3)


ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
fig1.colorbar(cs2, cax=cax2, orientation='horizontal')
#fig1.colorbar(cs3, cax=cax3, orientation='horizontal')

cax1.set_xlabel('Ice Concentration (%)')
cax2.set_xlabel('Ice Thickness (m)')
#cax3.set_xlabel('Density (kg/m$^{3}$)')


# In[ ]:


fig1.savefig('./Figures/ice_conc.png')

