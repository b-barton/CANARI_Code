#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import glob
import netCDF4 as nc
import datetime as dt
import sys
import gsw as sw
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('agg')

def calc_rho(sp, tp, depth, lon, lat):
  pres = sw.p_from_z(depth * -1, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
  return rho

#fn_en4 = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/en4_processed.nc'
dn_nemo_data = '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/means/'
fn_nemo_domain = '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/domain/mesh_hgr.nc'
fn_nemo_mask = '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/domain/mask.nc'
fn_nemo_bathy = '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/domain/bathymetry_ORCA12_V3.3.nc'
out_file = './Processed/'

#date_str = str(sys.argv[1])
date_str = '20000101'
date_use = dt.datetime.strptime(date_str, '%Y%m%d')
yr_use = str(date_use.year)
yr_end = date_use.year + (date_use.month + 1) // 13
mn_end = (date_use.month % 12) + 1
date_end = dt.datetime(yr_end, mn_end, 1)

date_mn = date_use.strftime('%Ym%m')
flist = (dn_nemo_data + yr_use 
        + '/ORCA0083-N06_' + date_mn + 'T.nc')
#flist = sorted(glob.glob(flist))

lon_min = 0
lon_max = 360
lat_min = 70
lat_max = 90

with  nc.Dataset(flist, 'r') as nc_fid:
  lon = nc_fid.variables['nav_lon'][:]
  lat = nc_fid.variables['nav_lat'][:] # y, x
  depth = nc_fid.variables['deptht'][:]
  date = nc_fid.variables['time_centered'][:]
  temp = nc_fid.variables['potemp'][0, :, :, :]
  sal = nc_fid.variables['salin'][0, :, :, :] # time, depth, y, x

with  nc.Dataset(fn_nemo_domain, 'r') as nc_fid:
  lon = nc_fid.variables['nav_lon'][:]
  lat = nc_fid.variables['nav_lat'][:] # y, x

with  nc.Dataset(fn_nemo_mask, 'r') as nc_fid:
  tmask = nc_fid.variables['tmask'][:] # t, z, y, x
  tmask_util = nc_fid.variables['tmaskutil'][:] # t, y, x

with  nc.Dataset(fn_nemo_bathy, 'r') as nc_fid:
  bathy = nc_fid.variables['Bathymetry'][:] # y, x


if 1:
  yi1 = 2300
  yi2 = lon.shape[0]
  lon = lon[yi1:yi2, :]
  lat = lat[yi1:yi2, :]
  temp = temp[:, yi1:yi2, :]
  sal = sal[:, yi1:yi2, :]
  bathy = bathy[yi1:yi2, :]
  tmask_util = tmask_util[:, yi1:yi2, :]

def polar_part(option, lon, lat, var):
  # var should be 2D
  xi1 = (lon.shape[1] // 2)
  if option == 1:
    lon_r = np.deg2rad(lon[::-1, xi1:])
    lat_r = -(lat[::-1, xi1:] - 90)
    var_r = var[::-1, xi1:]
  elif option == 2:
    lon_r = np.deg2rad(lon[::-1, :xi1] % 360)
    lat_r = -(lat[::-1, :xi1] - 90)
    var_r = var[::-1, :xi1]

  return lon_r, lat_r, var_r

print(bathy.shape)

temp = np.ma.masked_where(temp == 1e20, temp)
sal = np.ma.masked_where(sal == 1e20, sal)
bathy = np.ma.masked_where(tmask_util[0, :, :] == 0, bathy)

#fig1 = plt.figure(figsize=(9, 8))
#ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
#lon1, lat1, bathy1 = polar_part(1, lon, lat, bathy)
#lon2, lat2, bathy2 = polar_part(2, lon, lat, bathy)
#ax1.pcolormesh(lon1, lat1, bathy1)
#ax1.pcolormesh(lon2, lat2, bathy2)
#ax1.set_ylim([0, 30])
#plt.show()
#sys.exit()

depth_g = np.tile(depth, (sal.shape[2], sal.shape[1], 1)).T
rho = calc_rho(sal, temp, depth_g, np.mean(lon), np.mean(lat))

mask = temp.mask[0, :, :]
print(mask.shape)
lon_r = lon[np.invert(mask)]
lat_r = lat[np.invert(mask)]
b_ll = np.array([lon_r, lat_r]).T

lat_u = np.ma.mean(lat, axis=1)
lon_u = np.ma.mean(lon, axis=0)

# Find bottom index

bot_ind = np.zeros(lon.shape, dtype=int)
bot_temp = np.ma.zeros(lon.shape)
bot_sal = np.ma.zeros(lon.shape)
bot_rho = np.ma.zeros(lon.shape)

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

data_crs = ccrs.PlateCarree()
mrc = ccrs.NorthPolarStereo(central_longitude=0.0)
#mrc = 'polar'
my_cm = plt.cm.plasma

fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_axes([0.1, 0.05, 0.35, 0.8], projection=mrc)
ax2 = fig1.add_axes([0.55, 0.05, 0.35, 0.8], projection=mrc)
cax1 = fig1.add_axes([0.1, 0.94, 0.35, 0.02])
cax2 = fig1.add_axes([0.55, 0.94, 0.35, 0.02])

lon1, lat1, bathy1 = polar_part(1, lon, lat, bathy)
lon2, lat2, bathy2 = polar_part(2, lon, lat, bathy)
cs1 = ax1.pcolormesh(lon, lat, lon, transform=data_crs, cmap=my_cm)
#ax1.pcolormesh(lon2, lat2, lon2, cmap=my_cm)

cs2 = ax2.pcolormesh(lon, lat, lat, transform=data_crs, cmap=my_cm)
#cs2 = ax2.pcolormesh(lon1, lat1, lat1)
#ax2.pcolormesh(lon2, lat2, lat2)

#ax1.set_ylim([0, 30])
#ax2.set_ylim([0, 30])

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
ax1.gridlines()
ax2.gridlines()

ax1.set_extent([-180, 180, 60, 90], crs=data_crs)
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)

def set_circle(ax):
  # Compute a circle in axes coordinates, which we can use as a boundary
  # for the map. We can pan/zoom as much as we like - the boundary will be
  # permanently circular.
  theta = np.linspace(0, 2 * np.pi, 100)
  center, radius = [0.5, 0.5], 0.5
  verts = np.vstack([np.sin(theta), np.cos(theta)]).T
  circle = mpath.Path(verts * radius + center)
  ax.set_boundary(circle, transform=ax.transAxes)

set_circle(ax2)

fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
fig1.colorbar(cs2, cax=cax2, orientation='horizontal')

cax1.set_xlabel('Longitude')
cax2.set_xlabel('Latitude')


fig2 = plt.figure(figsize=(12, 6))
ax1 = fig2.add_axes([0.1, 0.1, 0.25, 0.75], projection=mrc)
ax2 = fig2.add_axes([0.4, 0.1, 0.25, 0.75], projection=mrc)
ax3 = fig2.add_axes([0.7, 0.1, 0.25, 0.75], projection=mrc)
cax1 = fig2.add_axes([0.1, 0.94, 0.25, 0.02])
cax2 = fig2.add_axes([0.4, 0.94, 0.25, 0.02])
cax3 = fig2.add_axes([0.7, 0.94, 0.25, 0.02])

lon1, lat1, temp1 = polar_part(1, lon, lat, temp[0, :, :])
lon2, lat2, temp2 = polar_part(2, lon, lat, temp[0, :, :])
cs1 = ax1.pcolormesh(lon1, lat1, temp1, vmin=-2, vmax=15, cmap=my_cm)
cs1 = ax1.pcolormesh(lon2, lat2, temp2, vmin=-2, vmax=15, cmap=my_cm)

lon1, lat1, sal1 = polar_part(1, lon, lat, sal[0, :, :])
lon2, lat2, sal2 = polar_part(2, lon, lat, sal[0, :, :])
cs2 = ax2.pcolormesh(lon1, lat1, sal1, vmin=25, vmax=36, cmap=my_cm)
cs2 = ax2.pcolormesh(lon2, lat2, sal2, vmin=25, vmax=36, cmap=my_cm)

lon1, lat1, rho1 = polar_part(1, lon, lat, rho[0, :, :])
lon2, lat2, rho2 = polar_part(2, lon, lat, rho[0, :, :])
cs3 = ax3.pcolormesh(lon1, lat1, rho1, vmin=1020, vmax=1028.5, cmap=my_cm)
cs3 = ax3.pcolormesh(lon2, lat2, rho2, vmin=1020, vmax=1028.5, cmap=my_cm)

ax1.set_ylim([0, 30])
ax2.set_ylim([0, 30])
ax3.set_ylim([0, 30])

ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig2.colorbar(cs1, cax=cax1, orientation='horizontal')
fig2.colorbar(cs2, cax=cax2, orientation='horizontal')
fig2.colorbar(cs3, cax=cax3, orientation='horizontal')

cax1.set_xlabel('Temperature ($^{\circ}C$)')
cax2.set_xlabel('Salinity')
cax3.set_xlabel('Density (kg/m$^{3}$)')


fig3 = plt.figure(figsize=(12, 6))
ax1 = fig3.add_axes([0.1, 0.1, 0.25, 0.75], projection=mrc)
ax2 = fig3.add_axes([0.4, 0.1, 0.25, 0.75], projection=mrc)
ax3 = fig3.add_axes([0.7, 0.1, 0.25, 0.75], projection=mrc)
cax1 = fig3.add_axes([0.1, 0.94, 0.25, 0.02])
cax2 = fig3.add_axes([0.4, 0.94, 0.25, 0.02])
cax3 = fig3.add_axes([0.7, 0.94, 0.25, 0.02])

lon1, lat1, temp1 = polar_part(1, lon, lat, bot_temp)
lon2, lat2, temp2 = polar_part(2, lon, lat, bot_temp)
cs1 = ax1.pcolormesh(lon1, lat1, temp1, vmin=-2, vmax=7, cmap=my_cm)
cs1 = ax1.pcolormesh(lon2, lat2, temp2, vmin=-2, vmax=7, cmap=my_cm)

lon1, lat1, sal1 = polar_part(1, lon, lat, bot_sal)
lon2, lat2, sal2 = polar_part(2, lon, lat, bot_sal)
cs2 = ax2.pcolormesh(lon1, lat1, sal1, vmin=34.5, vmax=35.5, cmap=my_cm)
cs2 = ax2.pcolormesh(lon2, lat2, sal2, vmin=34.5, vmax=35.5, cmap=my_cm)

lon1, lat1, rho1 = polar_part(1, lon, lat, bot_rho)
lon2, lat2, rho2 = polar_part(2, lon, lat, bot_rho)
cs3 = ax3.pcolormesh(lon1, lat1, rho1, vmin=1027.5, vmax=1028.5, cmap=my_cm)
cs3 = ax3.pcolormesh(lon2, lat2, rho2, vmin=1027.5, vmax=1028.5, cmap=my_cm)

ax1.set_ylim([0, 30])
ax2.set_ylim([0, 30])
ax3.set_ylim([0, 30])


fig3.colorbar(cs1, cax=cax1, orientation='horizontal')
fig3.colorbar(cs2, cax=cax2, orientation='horizontal')
fig3.colorbar(cs3, cax=cax3, orientation='horizontal')

cax1.set_xlabel('Temperature ($^{\circ}C$)')
cax2.set_xlabel('Salinity')
cax3.set_xlabel('Density (kg/m$^{3}$)')

ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/surface_lat_lon.png')
fig2.savefig('./Figures/surface_t_s.png')
fig3.savefig('./Figures/bottom_t_s.png')

