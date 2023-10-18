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


out_file = './Processed/'


# In[ ]:


# Load timeseries data

data = np.load(out_file + 'ice_time.npz', allow_pickle=True)
sic_time = data['sic_time']
sit_time = data['sit_time']
date = data['date']
data.close()

sic_time = np.ma.masked_where(sic_time==-1e20, sic_time)
sit_time = np.ma.masked_where(sit_time==-1e20, sit_time)

data = np.load(out_file + 'ice_mn_1990-2000.npz', allow_pickle=True)
sit_map1 = data['sit_map']
lat = data['lat']
lon = data['lon']
data.close()

sit_map1 = np.ma.masked_where(sit_map1==-1e20, sit_map1)
sit_mean1 = np.ma.mean(sit_map1, axis=0)
sit_std1 = np.ma.std(sit_map1, axis=0)

data = np.load(out_file + 'ice_mn_2040-2050.npz', allow_pickle=True)
sit_map2 = data['sit_map']
lat = data['lat']
lon = data['lon']
data.close()

sit_map2 = np.ma.masked_where(sit_map2==-1e20, sit_map2)
sit_mean2 = np.ma.mean(sit_map2, axis=0)
sit_std2 = np.ma.std(sit_map2, axis=0)

plot_date = np.zeros((len(date)), dtype=object)
for i in range(len(date)):
    bb = date[i].timetuple()
    plot_date[i] = dt.datetime(bb[0], bb[1], 1)


# In[ ]:


# Running mean

run = 12
sit_run = np.ma.zeros((sit_time.shape[0] - run, sit_time.shape[1]))
date_run = np.zeros((len(plot_date) - run), dtype=object)

for i in range(len(date_run)):
    sit_run[i] = np.ma.mean(sit_time[i:i + run, :], axis=0)
    date_run[i] = plot_date[i + (run // 2)]
    


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


fig1 = plt.figure(figsize=(12, 8))
ax5 = fig1.add_axes([0.1, 0.66, 0.85, 0.3])

ax1 = fig1.add_axes([0.01, 0.04, 0.23, 0.45], projection=mrc)
ax2 = fig1.add_axes([0.26, 0.04, 0.23, 0.45], projection=mrc)
ax3 = fig1.add_axes([0.51, 0.04, 0.23, 0.45], projection=mrc)
ax4 = fig1.add_axes([0.76, 0.04, 0.23, 0.45], projection=mrc)

cax1 = fig1.add_axes([0.03, 0.56, 0.19, 0.02])
cax2 = fig1.add_axes([0.28, 0.56, 0.19, 0.02])
cax3 = fig1.add_axes([0.53, 0.56, 0.19, 0.02])
cax4 = fig1.add_axes([0.78, 0.56, 0.19, 0.02])

cs1 = ax1.pcolormesh(lon, lat, sit_mean1, transform=data_crs, cmap=my_cm, vmin=0, vmax=6)
cs2 = ax2.pcolormesh(lon, lat, sit_mean2 - sit_mean1, transform=data_crs, cmap=my_cm, vmin=-3, vmax=0)

cs3 = ax3.pcolormesh(lon, lat, sit_std1, transform=data_crs, cmap=my_cm, vmin=0, vmax=1)
cs4 = ax4.pcolormesh(lon, lat, sit_std2 - sit_std1, transform=data_crs, cmap=my_cm, vmin=-0.5, vmax=0.5)

ax5.plot(plot_date, sit_time[:, 0], color='tab:blue', alpha=0.5)
ax5.plot(plot_date, sit_time[:, 1], color='tab:orange', alpha=0.5)
ax5.plot(plot_date, sit_time[:, 2], color='tab:green', alpha=0.5)

ax5.plot(date_run, sit_run[:, 0], color='tab:blue', label='> 70 N')
ax5.plot(date_run, sit_run[:, 1], color='tab:orange', label='> 75 N')
ax5.plot(date_run, sit_run[:, 2], color='tab:green', label='> 80 N')

ax5.plot([dt.datetime(1990, 1, 1), dt.datetime(1990, 1, 1), dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 1), dt.datetime(1990, 1, 1)], 
         [-4e22, 4e22, 4e22, -4e22, -4e22], 'k', zorder=105)
ax5.plot([dt.datetime(2040, 1, 1), dt.datetime(2040, 1, 1), dt.datetime(2050, 1, 1), dt.datetime(2050, 1, 1), dt.datetime(2040, 1, 1)], 
         [-4e22, 4e22, 4e22, -4e22, -4e22], 'k', zorder=105)

ax1.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax1)

ax2.add_feature(cfeature.LAND, zorder=100)
ax2.gridlines()
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax2)

ax3.add_feature(cfeature.LAND, zorder=100)
ax3.gridlines()
ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax3)

ax4.add_feature(cfeature.LAND, zorder=100)
ax4.gridlines()
ax4.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax4)

ax5.set_ylim([0, 3.5e13])
ax5.set_xlim([dt.datetime(1950, 1, 1), dt.datetime(2051, 1, 1)])
ax5.legend(loc='upper right')
ax5.set_ylabel('Sea volume (m$^{3}$)')

ax5.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1.annotate('(b) 1990s', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(c) 2040s', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(d) 1990s', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(e) 2040s', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
cax1.set_xlabel('Sea ice thick. (m)')

fig1.colorbar(cs2, cax=cax2, orientation='horizontal')
cax2.set_xlabel('Diff. SIT (m)')

fig1.colorbar(cs3, cax=cax3, orientation='horizontal')
cax3.set_xlabel('Sea ice thick. StD. (m)')

fig1.colorbar(cs4, cax=cax4, orientation='horizontal')
cax4.set_xlabel('Diff. SIT StD. (m)')


# In[ ]:


fig1.savefig('./Figures/sit_time.png')

