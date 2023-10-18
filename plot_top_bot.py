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
var = 'thetao' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
i_o = 'O' # SI or O for sea ice or ocean
freq = 'mon' # mon or day
root = '/badc/cmip6/data/CMIP6/HighResMIP/NERC/HadGEM3-GC31-HH/hist-1950/r1i1p1f1/'
root = '/badc/cmip6/data/CMIP6/HighResMIP/MOHC/HadGEM3-GC31-HH/highres-future/r1i1p1f1/'
fn_nemo_dat_t = root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_hist-1950_r1i1p1f1_gn_*.nc'
fn_nemo_dat_t = root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_highres-future_r1i1p1f1_gn_*.nc'

var = 'so' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
fn_nemo_dat_s = root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_hist-1950_r1i1p1f1_gn_*.nc'
fn_nemo_dat_s = root + i_o + freq + '/' + var + '/gn/latest/' + var + '_' + i_o + freq + '_HadGEM3-GC31-HH_highres-future_r1i1p1f1_gn_*.nc'

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'

out_file = './Processed/'


# In[ ]:


flist_t = sorted(glob.glob(fn_nemo_dat_t))
flist_s = sorted(glob.glob(fn_nemo_dat_s))
nemo_t = coast.Gridded(fn_data = flist_t[-1], fn_domain = fn_nemo_dom, config=fn_config_t_grid)
nemo_s = coast.Gridded(fn_data = flist_s[-1], fn_domain = fn_nemo_dom, config=fn_config_t_grid)
nemo_t.dataset['salinity'] = nemo_s.dataset.salinity

with nc.Dataset(fn_nemo_bathy, 'r') as nc_fid:
  bathy = nc_fid.variables['Bathymetry'][:]

print(nemo_t.dataset.longitude.shape)


# Subset data

# In[ ]:


#print(nemo_t.dataset)


# In[ ]:


lon_min = 0
lon_max = 360
lat_min = 70
lat_max = 90


# In[ ]:


yi1 = 2300
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
  return rho

# change this to decrease resolution but decrease run time
sub = 5

lat = nemo_t_subset.dataset.latitude.values[::sub, ::sub]
lon = nemo_t_subset.dataset.longitude.values[::sub, ::sub]
depth = nemo_t_subset.dataset.depth_0.values[:]
temp = nemo_t_subset.dataset.temperature[0, :, ::sub, ::sub].to_masked_array() # time, lev, j, i
sal = nemo_t_subset.dataset.salinity[0, :, ::sub, ::sub].to_masked_array()
depth_g = np.tile(depth, (sal.shape[2], sal.shape[1], 1)).T
rho = calc_rho(sal, temp, depth_g, lon, lat)



# Find bottom index

# In[ ]:


bot_ind = np.zeros(lon.shape, dtype=int)
bot_temp = np.ma.zeros(lon.shape)
bot_sal = np.ma.zeros(lon.shape)
bot_rho = np.ma.zeros(lon.shape)
print(lon.shape)


# In[ ]:


for i in range(lon.shape[0]):
  for j in range(lon.shape[1]):
    ind = np.nonzero(np.invert(temp.mask[:, i, j]))[0]
    if len(ind) == 0:
      ind = 0
    else:
      ind = ind[-1]
    bot_ind[i, j] = ind
    bot_temp[i, j] = temp[ind, i, j]
    bot_sal[i, j] = sal[ind, i, j]
    bot_rho[i, j] = rho[ind, i, j]


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
#mrc = 'polar'
my_cm = plt.cm.plasma


# In[ ]:


fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_axes([0.1, 0.05, 0.35, 0.8], projection=mrc)
ax2 = fig1.add_axes([0.55, 0.05, 0.35, 0.8], projection=mrc)
cax1 = fig1.add_axes([0.1, 0.94, 0.35, 0.02])
cax2 = fig1.add_axes([0.55, 0.94, 0.35, 0.02])

cs1 = ax1.pcolormesh(lon, lat, lon, transform=data_crs, cmap=my_cm)
cs2 = ax2.pcolormesh(lon, lat, lat, transform=data_crs, cmap=my_cm)

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax2.gridlines()

ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)

set_circle(ax1)
set_circle(ax2)

fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
fig1.colorbar(cs2, cax=cax2, orientation='horizontal')

cax1.set_xlabel('Longitude')
cax2.set_xlabel('Latitude')


# In[ ]:


fig2 = plt.figure(figsize=(12, 6))
ax1 = fig2.add_axes([0.03, 0.03, 0.3, 0.85], projection=mrc)
ax2 = fig2.add_axes([0.36, 0.03, 0.3, 0.85], projection=mrc)
ax3 = fig2.add_axes([0.69, 0.03, 0.3, 0.85], projection=mrc)
cax1 = fig2.add_axes([0.03, 0.94, 0.3, 0.02])
cax2 = fig2.add_axes([0.36, 0.94, 0.3, 0.02])
cax3 = fig2.add_axes([0.69, 0.94, 0.3, 0.02])

cs1 = ax1.pcolormesh(lon, lat, temp[0, :, :], vmin=-2, vmax=15, transform=data_crs, cmap=my_cm)
cs2 = ax2.pcolormesh(lon, lat, sal[0, :, :], vmin=25, vmax=36, transform=data_crs, cmap=my_cm)
cs3 = ax3.pcolormesh(lon, lat, rho[0, :, :], vmin=1020, vmax=1028.5, transform=data_crs, cmap=my_cm)

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
ax3.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax2.gridlines()
ax3.gridlines()

ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)
ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax1)
set_circle(ax2)
set_circle(ax3)


ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig2.colorbar(cs1, cax=cax1, orientation='horizontal')
fig2.colorbar(cs2, cax=cax2, orientation='horizontal')
fig2.colorbar(cs3, cax=cax3, orientation='horizontal')

cax1.set_xlabel('Temperature ($^{\circ}C$)')
cax2.set_xlabel('Salinity')
cax3.set_xlabel('Density (kg/m$^{3}$)')


# In[ ]:


fig3 = plt.figure(figsize=(12, 6))
ax1 = fig3.add_axes([0.03, 0.03, 0.3, 0.85], projection=mrc)
ax2 = fig3.add_axes([0.36, 0.03, 0.3, 0.85], projection=mrc)
ax3 = fig3.add_axes([0.69, 0.03, 0.3, 0.85], projection=mrc)
cax1 = fig3.add_axes([0.03, 0.94, 0.3, 0.02])
cax2 = fig3.add_axes([0.36, 0.94, 0.3, 0.02])
cax3 = fig3.add_axes([0.69, 0.94, 0.3, 0.02])

cs1 = ax1.pcolormesh(lon, lat, bot_temp, vmin=-2, vmax=7, transform=data_crs, cmap=my_cm)
cs2 = ax2.pcolormesh(lon, lat, bot_sal, vmin=34.5, vmax=35.5, transform=data_crs, cmap=my_cm)
cs3 = ax3.pcolormesh(lon, lat, bot_rho, vmin=1027.5, vmax=1028.5, transform=data_crs, cmap=my_cm)

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
ax3.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax2.gridlines()
ax3.gridlines()

ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)
ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax1)
set_circle(ax2)
set_circle(ax3)


fig3.colorbar(cs1, cax=cax1, orientation='horizontal')
fig3.colorbar(cs2, cax=cax2, orientation='horizontal')
fig3.colorbar(cs3, cax=cax3, orientation='horizontal')

cax1.set_xlabel('Temperature ($^{\circ}C$)')
cax2.set_xlabel('Salinity')
cax3.set_xlabel('Density (kg/m$^{3}$)')

ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


# In[ ]:


fig1.savefig('./Figures/surface_lat_lon.png')
fig2.savefig('./Figures/surface_t_s.png')
fig3.savefig('./Figures/bottom_t_s.png')

