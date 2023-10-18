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


#fn_en4 = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/en4_processed.nc'
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

fn_nemo_dat_t = make_path('thetao', i_o, freq, time_s)
fn_nemo_dat_s = make_path('so', i_o, freq, time_s)

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


flist_t = sorted(glob.glob(fn_nemo_dat_t))
flist_s = sorted(glob.glob(fn_nemo_dat_s))
nemo_t = coast.Gridded(fn_data = flist_t[:3], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
nemo_s = coast.Gridded(fn_data = flist_s[:3], fn_domain = fn_nemo_dom, config=fn_config_t_grid, multiple=True)
nemo_t.dataset['salinity'] = nemo_s.dataset.salinity

print(nemo_t.dataset.longitude.shape)


# Subset data

# In[ ]:


#print(nemo_t.dataset)


# In[ ]:


yi1 = 2800
yi2 = nemo_t.dataset.longitude.shape[0]
ind_y = np.arange(yi1, yi2).astype(int)
#print(yi2, ind_y)
nemo_t_subset = nemo_t.isel(y_dim=ind_y)


# Time slice. Notes dates are 360 day years so use cftime

# In[ ]:


# Does not work
#st_date = cftime.datetime(2050, 12, 1)
#en_date = cftime.datetime(2051, 1, 1) 
#nemo_t_subset = nemo_t_subset.time_slice(st_date, en_date)


# In[ ]:


print(nemo_t_subset.dataset)


# In[ ]:


#nemo_t_subset.construct_density(eos='EOS10')#, pot_dens=True)

def calc_rho(sp, tp, depth, lon, lat):
    pres = sw.p_from_z(depth * -1, lat)
    sa = sw.SA_from_SP(sp, pres, lon, lat)
    ct = sw.CT_from_pt(sa, tp)
    rho = sw.rho(sa, ct, 0)
    return rho, ct

def calc_heat(rho, ct):
    cap = 3991.86795711963 # 3985 # J kg-1 K-1
    #cap = sw.cp0 # use with conservative temp
    return ct * rho * cap

# change this to decrease resolution but decrease run time
sub = 5

ds_dom = xr.open_dataset(fn_nemo_dom).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
e1t = ds_dom.e1t[yi1+1:-1:sub, 1:-1:sub] # y, x
e2t = ds_dom.e2t[yi1+1:-1:sub, 1:-1:sub]
e3t = ds_dom.e3t_0[:, yi1+1:-1:sub, 1:-1:sub] # z, y, x
e1t = np.tile(e1t, (e3t.shape[0], 1, 1))
e2t = np.tile(e2t, (e3t.shape[0], 1, 1))
print(e3t.shape, e1t.shape)
volume = e1t * e2t * e3t


lat = nemo_t_subset.dataset.latitude.values[::sub, ::sub]
lon = nemo_t_subset.dataset.longitude.values[::sub, ::sub]
depth = nemo_t_subset.dataset.depth_0.values[:]
depth_g = np.tile(depth, (lon.shape[1], lon.shape[0], 1)).T

mask_south = lat < 70
heat_time = np.ma.zeros((nemo_t_subset.dataset.t_dim.shape[0]))
for i in range(nemo_t_subset.dataset.t_dim.shape[0]):
    temp = nemo_t_subset.dataset.temperature[i, :, ::sub, ::sub].to_masked_array() # time, lev, j, i
    sal = nemo_t_subset.dataset.salinity[i, :, ::sub, ::sub].to_masked_array()
    rho, ct = calc_rho(sal, temp, depth_g, lon, lat)

    heat_cont = np.ma.sum(calc_heat(rho, ct) * volume, axis=0) # vertically integrated
    heat_pole = np.ma.masked_where(mask_south, heat_cont)
    heat_time[i] = np.ma.sum(heat_pole)


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
ax2 = fig1.add_axes([0.36, 0.03, 0.6, 0.85])

cax1 = fig1.add_axes([0.03, 0.94, 0.3, 0.02])

cs1 = ax1.pcolormesh(lon, lat, heat_cont, transform=data_crs, cmap=my_cm)
ax2.plot(heat_time)

ax1.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax1)

ax2.set_ylabel('Heat Content (Jm$^{-2}$)')

ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
cax1.set_xlabel('Heat Content (Jm$^{-3}$)')


# In[ ]:


fig1.savefig('./Figures/heat_content.png')

