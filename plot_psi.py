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
import matplotlib.cm as cm
import cftime
import coast
import xarray as xr
import scipy.interpolate as sci
from pyproj import crs
from pyproj import Transformer
sys.path.append('/home/users/benbar/work/Function_Lib/pyCDFTOOLS')
sys.path.append('/home/users/benbar/work/Function_Lib/vector_fields')
sys.path.append('/home/users/benbar/work/Function_Lib')
import cdfpsi_mod
from psi_phi import uv2psiphi
import smudge


# In[ ]:


in_file = './CDFTOOLS_Process/psi.nc'

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'



out_file = './Processed/'


# In[ ]:


v_map = {}
v_map['e1t'] = 'e1t'
v_map['e2t'] = 'e2t'
v_map['e3t_0'] = 'e3t_0'
v_map['e3u_0'] = 'e3u_0'
v_map['e3v_0'] = 'e3v_0'
v_map['tmask'] = 'tmask'
v_map['lat'] = 'nav_lat'
v_map['lon'] = 'nav_lon'
v_map['depth'] = 'lev'
v_map['time'] = 'time'
v_map['u'] = 'uo'
v_map['v'] = 'vo'
v_map['psi'] = 'sobarstf'


with nc.Dataset(in_file, 'r') as nc_fid:
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]
    psi = nc_fid.variables[v_map['psi']][:]

with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    e1t = nc_fid.variables[v_map['e1t']][0, 1:-1, 1:-1] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, 1:-1, 1:-1]
    e3u = nc_fid.variables[v_map['e3u_0']][0, ...] # t, z, y, x
    e3v = nc_fid.variables[v_map['e3v_0']][0, ...] # t, z, y, x

print(psi.shape)
psi = np.squeeze(psi)


# In[ ]:


sub = 10
lon_bnds, lat_bnds = (-180, 180), (60, 90)
y1 = np.min(np.nonzero((lat >= lat_bnds[0]))[0])
y2 = np.max(np.nonzero((lat <= lat_bnds[1]))[0])
x1 = np.min(np.nonzero((lon >= lon_bnds[0]))[0])
x2 = np.max(np.nonzero((lon <= lon_bnds[1]))[0])
print(y1, y2, x1, x2, lon.shape)
x1 = 0
x2 = lon.shape[1]

lat_n = lat[y1:y2:sub, ::sub]
lon_n = lon[y1:y2:sub, ::sub]

psi_n = psi[y1:y2:sub, ::sub]
e1t = e1t[y1:y2:sub, ::sub]
e2t = e2t[y1:y2:sub, ::sub]

print(psi_n.shape)


# In[ ]:


# Cartopy bug

def polar_uv(u, v, lat):
    # Adjust u and v to work-around bug in cartopy quiver plotting
    u_src_crs = u / np.cos(lat / 180 * np.pi)
    v_src_crs = v * 1 # * np.cos(lat / (180 * np.pi))
    magnitude = (u**2 + v**2) ** 0.5
    magn_src_crs = (u_src_crs**2 + v_src_crs**2) ** 0.5
    u_new = u_src_crs * (magnitude / magn_src_crs)
    v_new = v_src_crs * (magnitude / magn_src_crs)
    return u_new, v_new


def stream_to_uv(stream, x, y, dxdy=False):
    if np.ma.is_masked(stream):
        stream_f = smudge.sea_over_land(stream[:, :, np.newaxis], stream[:, :], npoints=1)[:, :, 0]
        stream_f = stream_f.filled(0)
    else:
        stream_f = stream
        
    if dxdy:
        dx = x
        dy = y
    else:
        dx = np.gradient(x, axis=1)
        dy = np.gradient(y, axis=0)
        
    u_st = np.gradient(stream_f, axis=0) / dy
    v_st = -np.gradient(stream_f, axis=1) / dx
    if np.ma.is_masked(stream):
        u_st = np.ma.masked_where(stream.mask, u_st)
        v_st = np.ma.masked_where(stream.mask, v_st)
    return u_st, v_st

def uv_to_stream(u, v, x, y, dxdy=False, use_v=False):
    if np.ma.is_masked(u):
        u_f = smudge.sea_over_land(u[:, :, np.newaxis], u[:, :], npoints=1, fill=True)[:, :, 0]
        v_f = smudge.sea_over_land(v[:, :, np.newaxis], v[:, :], npoints=1, fill=True)[:, :, 0]
        u_f = u_f.filled(0)
        v_f = v_f.filled(0)
    else:
        u_f = u
        v_f = v
    if dxdy:
        dx = x
        dy = y
    else:
        dx = np.gradient(x, axis=1)
        dy = np.gradient(y, axis=0)
    if use_v:
        # use v
        dtrpv = v_f * dx
        # do zonal integration
        dpsiv = np.ma.zeros(v_f.shape)
        npjglo, npiglo = v_f.shape
        for ji in range(npiglo-2, -1, -1):
            dpsiv[:, ji] = dpsiv[:, ji+1] - dtrpv[:, ji]
    
        # normalise
        dpsi = dpsiv - dpsiv[npjglo-1, npiglo-1] # python indexing
    else:
        # use v
        dtrpu = u_f * dy
        # do meridional integration
        dpsiu = np.ma.zeros(u_f.shape)
        npjglo, npiglo = u_f.shape
        for jj in range(1, npjglo):
            dpsiu[jj, :] = dpsiu[jj-1, :] - dtrpu[jj, :]
    
        # normalise
        dpsi = dpsiu - dpsiu[npjglo-1, npiglo-1] # python indexing
    if np.ma.is_masked(u):
        dpsi = np.ma.masked_where(u.mask, dpsi)
    return dpsi * -1
        


# In[ ]:


# get u and v of psi
u_st, v_st = stream_to_uv(psi_n, e1t, e2t, dxdy=True)
u_ad, v_ad = polar_uv(u_st, v_st, lat_n)


# Interpolate onto polar grid

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


# Plot

# In[ ]:


fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_axes([0.04, 0.04, 0.45, 0.8])
#ax1 = fig2.add_axes([0.04, 0.52, 0.45, 0.4])
#ax2 = fig2.add_axes([0.04, 0.04, 0.45, 0.4])
ax3 = fig1.add_axes([0.51, 0.1, 0.45, 0.8], projection=mrc)

cax1 = fig1.add_axes([0.51, 0.96, 0.23, 0.02])

grid = np.indices(psi.shape)
ax1.contourf(lon, lat, psi, cmap=my_cm)#, vmin=0, vmax=0.3)
#ax1.pcolormesh(grid[1][:, 200:], grid[0][:, 200:], psi[:, 200:], cmap=my_cm)#, vmin=0, vmax=0.3)
#ax2.pcolormesh(grid[1][:, :200], grid[0][:, :200], psi[:, :200], cmap=my_cm)#, vmin=0, vmax=0.3)


cs3 = ax3.contourf(lon_n, lat_n, psi_n, transform=data_crs, cmap=my_cm)#, vmin=-2e4, vmax=2e5)

ax3.quiver(lon_n[::skip, ::skip], lat_n[::skip, ::skip], u_ad[::skip, ::skip], v_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy')#, regrid_shape=60, zorder=101)
cs3 = ax3.contourf(lon, lat, psi, transform=data_crs, cmap=my_cm)#, vmin=-2e4, vmax=2e5)

skip = 4
#ax1.quiver(grid[1][::skip, ::skip], grid[0][::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', zorder=101)
#ax1.quiver(grid[1][::skip, 200::skip], grid[0][::skip, 200::skip], upsi[::skip, 200::skip], vpsi[::skip, 200::skip], color='w', zorder=101)
#ax2.quiver(grid[1][::skip, :200:skip], grid[0][::skip, :200:skip], upsi[::skip, :200:skip], vpsi[::skip, :200:skip], color='w', zorder=101)

#ax3.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', transform=data_crs, angles='xy', regrid_shape=60, zorder=101)
#ax4.streamplot(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], u_psi_ad[::skip, ::skip], v_psi_ad[::skip, ::skip], linewidth=1, density=3, color='w', zorder=101)

ax3.add_feature(cfeature.LAND, zorder=100)
ax3.gridlines()
ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax3)

fig1.colorbar(cs3, cax=cax1, orientation='horizontal')
#cax1.set_xlabel('Speed (m s$^{-1}$)')


# In[ ]:


fig1.savefig('./Figures/psi.png')

