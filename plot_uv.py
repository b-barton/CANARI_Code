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
sys.path.append('/home/users/benbar/work/Function_Lib/vector_fields')
sys.path.append('/home/users/benbar/work/Function_Lib/grid_angle')
from psi_phi import uv2psiphi
from nemo_grid_angle import GridAngle


# In[ ]:


var = 'uo' # thetao, so, uo, vo, siconc, siage, sivol, sithick, siu, siv, 
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

fn_nemo_dat_u1 = make_path('uo', i_o, freq, 'hist-1950')
fn_nemo_dat_v1 = make_path('vo', i_o, freq, 'hist-1950')
fn_nemo_dat_u2 = make_path('uo', i_o, freq, time_s)
fn_nemo_dat_v2 = make_path('vo', i_o, freq, time_s)
fn_nemo_dat_t1 = make_path('thetao', 'O', freq, 'hist-1950')

domain_root = '/gws/nopw/j04/nemo_vol5/acc/eORCA12-N512/domain/'
fn_nemo_dom1 = domain_root + 'eORCA12_coordinates.nc'
fn_nemo_dom = domain_root + 'mesh_mask_eORCA12_v2.4.nc'
fn_nemo_bathy = domain_root + 'eORCA12_bathymetry_v2.4.nc'
fn_config_t_grid = './config/gc31_nemo_grid_t.json'



out_file = './Processed/'


# In[ ]:


flist_u = sorted(glob.glob(fn_nemo_dat_u1))
flist_u.extend(sorted(glob.glob(fn_nemo_dat_u2)))
               
flist_v = sorted(glob.glob(fn_nemo_dat_v1))
flist_v.extend(sorted(glob.glob(fn_nemo_dat_v2)))
flist_t = sorted(glob.glob(fn_nemo_dat_t1))

v_map = {}
v_map['e1t'] = 'e1t'
v_map['e2t'] = 'e2t'
v_map['e3t_0'] = 'e3t_0'
v_map['e3u_0'] = 'e3u_0'
v_map['e3v_0'] = 'e3v_0'
v_map['tmask'] = 'tmask'
v_map['lat'] = 'latitude'
v_map['lon'] = 'longitude'
v_map['depth'] = 'lev'
v_map['time'] = 'time'
v_map['u'] = 'uo'
v_map['v'] = 'vo'


with nc.Dataset(flist_t[0], 'r') as nc_fid:
    lat = nc_fid.variables[v_map['lat']][:]
    lon = nc_fid.variables[v_map['lon']][:]
    lev = nc_fid.variables[v_map['depth']][:]

    
with nc.Dataset(fn_nemo_dom, 'r') as nc_fid:
    e1t = nc_fid.variables[v_map['e1t']][0, ...] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, ...]
    e3u = nc_fid.variables[v_map['e3u_0']][0, ...] # t, z, y, x
    e3v = nc_fid.variables[v_map['e3v_0']][0, ...] # t, z, y, x

if 0:
    ilev = 74 # 74 max
    d_depthu = np.sum(e3u[:ilev, 1:-1, 1:-1], axis=0)
    d_depthv = np.sum(e3v[:ilev, 1:-1, 1:-1], axis=0)
    print(lev[ilev])
    
    with nc.Dataset(flist_u[0], 'r') as nc_fid:
        #lat = nc_fid.variables[v_map['lat']][:]
        #lon = nc_fid.variables[v_map['lon']][:]
        for i in range(ilev):
            if i == 0:
                u_tmp = nc_fid.variables[v_map['u']][0, i, ...]
                u = np.ma.masked_where((u_tmp==1e20), u_tmp).filled(0) * e3u[i, 1:-1, 1:-1]
                u_mask = u_tmp==1e20
            else:
                u_tmp = nc_fid.variables[v_map['u']][0, i, ...]
                u = u + (np.ma.masked_where((u_tmp==1e20), u_tmp).filled(0) * e3u[i, 1:-1, 1:-1])
    
    u = np.ma.masked_where(u_mask, u)
    u = u / d_depthu
    print(np.ma.max(u))
    #u = np.ma.mean(u, axis=0)
    
    with nc.Dataset(flist_v[0], 'r') as nc_fid:
        for i in range(ilev):
            if i == 0:
                v_tmp = nc_fid.variables[v_map['v']][0, i, ...]
                v = np.ma.masked_where((v_tmp==1e20), v_tmp).filled(0) * e3v[i, 1:-1, 1:-1]
                v_mask = v_tmp==1e20
            else:
                v_tmp = nc_fid.variables[v_map['v']][0, i, ...]
                v = v + (np.ma.masked_where((v_tmp==1e20), v_tmp).filled(0) * e3v[i, 1:-1, 1:-1])
    
    v = np.ma.masked_where(v_mask, v)
    v = v / d_depthv
    #v = np.ma.mean(v, axis=0)
    
    #u = (u * 0) + 0.5
    #v = (v * 0) - 0.5

    np.savez(out_file + 'u_v_avg.npz', u=u, v=v, u_mask=u.mask, v_mask=v.mask)

else:
    data= np.load(out_file + 'u_v_avg.npz')
    u = data['u']
    v = data['v']
    u = np.ma.masked_where(data['u_mask'], u)
    v = np.ma.masked_where(data['v_mask'], v)
    data.close()
    


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

lat = lat[y1:y2:sub, ::sub]
lon = lon[y1:y2:sub, ::sub]
u = u[y1:y2:sub, ::sub]
v = v[y1:y2:sub, ::sub]


# In[ ]:


# Co-locate u and v onto t-grid

if 0:
    print(dir(coast.diagnostics.circulation))
    uv_grid = coast.diagnostics.circulation.CurrentsOnT(fn_data=[flist_u[0] + flist_v[0]], fn_domain=fn_nemo_dom, config=fn_config_t_grid, multiple=True, engine='netcdf4')
    t_uv = uv_grid.currents_on_t(u, v)

    ut = t_uv.ds_u["ut_velocity"]
    vt = t_uv.ds_v["vt_velocity"]

else:
    def currents_on_t(u, v):
        u_on_t_points = u * 1
        v_on_t_points = v * 1
        u_on_t_points[:, 1:] = 0.5 * (
            u[:, 1:] + u[:, :-1]
            )
        v_on_t_points[1:, :] = 0.5 * (
            v[1:, :] + v[:-1, :]
            )
        return u_on_t_points, v_on_t_points

    ut, vt = currents_on_t(u, v)



# In[ ]:


# Get angle of NEMO grid relative to North

def make_proj(x_origin, y_origin):
    aeqd = crs.CRS.from_proj4("+proj=aeqd +lon_0={:}".format(x_origin) + " +lat_0={:}".format(y_origin) + " +ellps=WGS84")
    return aeqd

def rotate_vel(u, v, angle, to_north=True):

    # use compass directions
    speed = (u ** 2 + v ** 2) ** 0.5
    direction = np.arctan2(u, v) * (180 / np.pi)
    
    # subtract the orientation angle of transect from compass North 
    # then u is across channel
    if to_north:
        new_direction = direction + angle
    else:
        new_direction = direction - angle
    
    u = speed * np.sin(new_direction * (np.pi / 180))
    v = speed * np.cos(new_direction * (np.pi / 180))
    return u, v

# Get angle using a metre grid transform

crs_wgs84 = crs.CRS('epsg:4326')

grid_angle = np.zeros(lon.shape)
for j in range(lon.shape[0] - 1):
    for i in range(lon.shape[1] - 1):
        crs_aeqd = make_proj(lon[j, i], lat[j, i])
        to_metre = Transformer.from_crs(crs_wgs84, crs_aeqd) 
        x_grid, y_grid = to_metre.transform(lat[j:j + 2, i:i + 2], lon[j:j + 2, i:i + 2])
        grid_angle[j, i] = np.arctan2((x_grid[1, 0] - x_grid[0, 0]), 
          (y_grid[1, 0] - y_grid[0, 0])) * (180 / np.pi) # relative to compass North

grid_angle[:, -1] = grid_angle[:, -2]
grid_angle[-1, :] = grid_angle[-2, :]
    
u_new, v_new = rotate_vel(ut, vt, grid_angle)


# Get grid angle in alternative way (function from PyNEMO)
# Extract the source rotation angles on the T-Points as the C-Grid

src_ga = GridAngle(fn_nemo_dom, x1, x2, y1, y2, 't')
# coord_fname, imin, imax, jmin, jmax, cd_type
gcos_u = src_ga.cosval[::sub, ::sub]
gsin_u = src_ga.sinval[::sub, ::sub]

src_ga = GridAngle(fn_nemo_dom, x1, x2, y1, y2, 't')
# coord_fname, imin, imax, jmin, jmax, cd_type
gcos_v = src_ga.cosval[::sub, ::sub]
gsin_v = src_ga.sinval[::sub, ::sub]

def rotate(u, v, gcos, gsin, cd_todo):
    if cd_todo == 'e':
        # rotation from the grid to real zonal direction, ie ij -> e
        vel = ((u * gcos) + ((v * -1) * gsin))
    elif cd_todo == 'n':
        # meridinal direction, ie ij -> n
        vel = ((u * gcos) + (v * gsin))
    return vel


#u_new = rotate(ut, vt, gcos_u, gsin_u, 'e')
#v_new = rotate(ut, vt, gcos_v, gsin_v, 'n')


# Unit vectors

speed = ((u_new ** 2) + (v_new ** 2)) ** 0.5
direc = np.arctan2(v_new, u_new)

# Convert to unit vectors for plotting
u_unit = 1 * np.cos((direc))
v_unit = 1 * np.sin((direc))


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

u_unit, v_unit = polar_uv(u_unit, v_unit, lat)
u_new_ad, v_new_ad = polar_uv(u_new, v_new, lat)

# Test some constants

u0_test = (u_new * 0) + 0
v0_test = (v_new * 0) + 0
u1_test = (u_new * 0) + 0.2
v1_test = (v_new * 0) + 0.2

u10_test_ad, v10_test_ad = polar_uv(u1_test, v0_test, lat)
u01_test_ad, v01_test_ad = polar_uv(u0_test, v1_test, lat)
u11_test_ad, v11_test_ad = polar_uv(u1_test, v1_test, lat)

u10_test_r = rotate(u1_test, v0_test, gcos_u, gsin_u, 'e')
v10_test_r = rotate(u1_test, v0_test, gcos_v, gsin_v, 'n')
u01_test_r = rotate(u0_test, v1_test, gcos_u, gsin_u, 'e')
v01_test_r = rotate(u0_test, v1_test, gcos_v, gsin_v, 'n')
u11_test_r = rotate(u1_test, v1_test, gcos_u, gsin_u, 'e')
v11_test_r = rotate(u1_test, v1_test, gcos_v, gsin_v, 'n')
u10_test_r, v10_test_r = rotate_vel(u1_test, v0_test, grid_angle)
u01_test_r, v01_test_r = rotate_vel(u0_test, v1_test, grid_angle)
u11_test_r, v11_test_r = rotate_vel(u1_test, v1_test, grid_angle)

u10_test_r_ad, v10_test_r_ad = polar_uv(u10_test_r, v10_test_r, lat)
u01_test_r_ad, v01_test_r_ad = polar_uv(u01_test_r, v01_test_r, lat)
u11_test_r_ad, v11_test_r_ad = polar_uv(u11_test_r, v11_test_r, lat)

#u_test_r, v_test_r = currents_on_t(u_test_r, v_test_r)
#u_test_ad_r, v_test_ad_r = polar_uv(u_test_r, v_test_r, lat)

data_crs = ccrs.PlateCarree()
mrc = ccrs.NorthPolarStereo(central_longitude=0.0)
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_axes([0.03, 0.69, 0.2, 0.28], projection=mrc)
ax2 = fig1.add_axes([0.28, 0.69, 0.2, 0.28], projection=mrc)
ax3 = fig1.add_axes([0.53, 0.69, 0.2, 0.28], projection=mrc)
ax4 = fig1.add_axes([0.78, 0.69, 0.2, 0.28], projection=mrc)

ax5 = fig1.add_axes([0.03, 0.36, 0.2, 0.28], projection=mrc)
ax6 = fig1.add_axes([0.28, 0.36, 0.2, 0.28], projection=mrc)
ax7 = fig1.add_axes([0.53, 0.36, 0.2, 0.28], projection=mrc)
ax8 = fig1.add_axes([0.78, 0.36, 0.2, 0.28], projection=mrc)

ax9 = fig1.add_axes([0.03, 0.03, 0.2, 0.28], projection=mrc)
ax10 = fig1.add_axes([0.28, 0.03, 0.2, 0.28], projection=mrc)
ax11 = fig1.add_axes([0.53, 0.03, 0.2, 0.28], projection=mrc)
ax12 = fig1.add_axes([0.78, 0.03, 0.2, 0.28], projection=mrc)

skip = 8
ax1.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u1_test[::skip, ::skip], v0_test[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax5.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u0_test[::skip, ::skip], v1_test[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax9.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u1_test[::skip, ::skip], v1_test[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)

ax2.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u10_test_ad[::skip, ::skip], v10_test_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax6.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u01_test_ad[::skip, ::skip], v01_test_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax10.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u11_test_ad[::skip, ::skip], v11_test_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)

ax3.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u10_test_r[::skip, ::skip], v10_test_r[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax7.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u01_test_r[::skip, ::skip], v01_test_r[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax11.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u11_test_r[::skip, ::skip], v11_test_r[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)

ax4.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u10_test_r_ad[::skip, ::skip], v10_test_r_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax8.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u01_test_r_ad[::skip, ::skip], v01_test_r_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)
ax12.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u11_test_r_ad[::skip, ::skip], v11_test_r_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy', zorder=101)

ax1.add_feature(cfeature.LAND, zorder=100)
ax2.add_feature(cfeature.LAND, zorder=100)
ax3.add_feature(cfeature.LAND, zorder=100)
ax4.add_feature(cfeature.LAND, zorder=100)
ax5.add_feature(cfeature.LAND, zorder=100)
ax6.add_feature(cfeature.LAND, zorder=100)
ax7.add_feature(cfeature.LAND, zorder=100)
ax8.add_feature(cfeature.LAND, zorder=100)
ax9.add_feature(cfeature.LAND, zorder=100)
ax10.add_feature(cfeature.LAND, zorder=100)
ax11.add_feature(cfeature.LAND, zorder=100)
ax12.add_feature(cfeature.LAND, zorder=100)


ax1.annotate('u1 v0', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('u0 v1', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax9.annotate('u1 v1', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1.set_title('Plain')
ax2.set_title('Polar Correct')
ax3.set_title('Grid Rotate')
ax4.set_title('GR and PC')

fig1.savefig('./Figures/polar_quiver.png')


# Interpolate onto polar grid

# In[ ]:


# stick tops together
half = u_new.shape[0]
i_pol = int(u_new.shape[0] * 2)
j_pol = int(u_new.shape[1] * 0.5)
u_pol = np.ma.zeros((i_pol, j_pol))
v_pol = np.ma.zeros((i_pol, j_pol))

u_pol[:half, :] = u_new[:, j_pol:]
u_pol[half:, :] = np.fliplr(np.flipud(u_new[:, :j_pol]))  * -1
v_pol[:half, :] = v_new[:, j_pol:]
v_pol[half:, :] = np.fliplr(np.flipud(v_new[:, :j_pol]))  * -1

lon_pol = np.ma.zeros((i_pol, j_pol))
lat_pol = np.ma.zeros((i_pol, j_pol))

lon_pol[:half, :] = lon[:, j_pol:]
lon_pol[half:, :] = np.fliplr(np.flipud(lon[:, :j_pol]))
lat_pol[:half, :] = lat[:, j_pol:]
lat_pol[half:, :] = np.fliplr(np.flipud(lat[:, :j_pol]))

#lat_pol = lat_pol[:50, :]
#lon_pol = lon_pol[:50, :]
#u_pol = u_pol[:50, :]
#v_pol = v_pol[:50, :]
print(lon_pol.shape, u_pol.shape)


if 0:
    # Generate new grid on NSIDC Polar Stereographic projection on WGS84
    crs_ps = crs.CRS('epsg:3413')
    #x_grid, y_grid = np.meshgrid(np.linspace(-3850, 3750, 608), np.linspace(-5350, 5850, 896))
    x_grid, y_grid = np.meshgrid(np.linspace(-3850, 3750, 304) * 1000, np.linspace(-5350, 5850, 448) * 1000)
    to_latlon = Transformer.from_crs(crs_ps, crs_wgs84) 
    lat_grid, lon_grid = to_latlon.transform(x_grid, y_grid)
    plt.subplot(211)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.pcolormesh(lon_grid)
    ax2.pcolormesh(lat_grid)
    #plt.savefig('grid.png')

else:
    # Generate regular lat lon grid
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 180 + (1/2), 1/2), np.arange(60, 90 + (1/4), 1/4))
    
interp_u = sci.NearestNDInterpolator(list(zip(lon.flatten(), lat.flatten())), u_new.flatten())
u_grid = interp_u(lon_grid, lat_grid)
interp_v = sci.NearestNDInterpolator(list(zip(lon.flatten(), lat.flatten())), v_new.flatten())
v_grid = interp_v(lon_grid, lat_grid)
interp_mask = sci.NearestNDInterpolator(list(zip(lon.flatten(), lat.flatten())), u_new.mask.flatten())
mask_grid = interp_mask(lon_grid, lat_grid)

u_grid = np.ma.masked_where(mask_grid, u_grid)
v_grid = np.ma.masked_where(mask_grid, v_grid)

print(lon_grid.shape, u_grid.shape)

u_grid_ad, v_grid_ad = polar_uv(u_grid, v_grid, lat_grid)


# Calculate streamfunction

# In[ ]:


psi, upsi, vpsi, phi, uphi, vphi = uv2psiphi(lon_grid, lat_grid, u_grid.filled(np.nan), v_grid.filled(np.nan), ZBC='periodic')

# the edges look odd
upsi[:, 0] = upsi[:, 1] * 1
upsi[:, -1] = upsi[:, -2] * 1
vpsi[:, 0] = vpsi[:, 1] * 1
vpsi[:, -1] = vpsi[:, -2] * 1

u_psi_ad, v_psi_ad = polar_uv(upsi, vpsi, lat_grid)


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
ax1 = fig1.add_axes([0.01, 0.45, 0.23, 0.45], projection=mrc)
ax2 = fig1.add_axes([0.26, 0.45, 0.23, 0.45], projection=mrc)
ax3 = fig1.add_axes([0.51, 0.06, 0.45, 0.8], projection=mrc)
ax4 = fig1.add_axes([0.04, 0.04, 0.45, 0.35])


cax1 = fig1.add_axes([0.01, 0.96, 0.23, 0.02])
cax2 = fig1.add_axes([0.26, 0.96, 0.23, 0.02])
cax3 = fig1.add_axes([0.51, 0.96, 0.23, 0.02])
#cax4 = fig1.add_axes([0.76, 0.96, 0.23, 0.02])

cs1 = ax1.pcolormesh(lon, lat, ut, transform=data_crs, cmap=my_cm, vmin=-0.02, vmax=0.02)
cs2 = ax2.pcolormesh(lon, lat, vt, transform=data_crs, cmap=my_cm, vmin=-0.02, vmax=0.02)
#cs2 = ax2.pcolormesh(lon_grid, lat_grid, u_grid, transform=data_crs, cmap=my_cm, vmin=-0.2, vmax=0.2)

cs3 = ax3.pcolormesh(lon, lat, speed, transform=data_crs, cmap=my_cm, vmin=0, vmax=0.03)

grid = np.indices(speed.shape)
ax4.pcolormesh(grid[1][:, 50:370], grid[0][:, 50:370], speed[:, 50:370], cmap=my_cm, vmin=0, vmax=0.03)


skip = 2

#ax2.quiver(lon[::skip, ::skip], lat[::skip, ::skip], (u_unit / np.cos(lat / 180 * np.pi))[::skip, ::skip], v_unit[::skip, ::skip], color='w', transform=data_crs, zorder=101, width=0.006, scale=50.0)
#ax2.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_unit[::skip, ::skip], (v_unit * np.cos(lat / 180 * np.pi))[::skip, ::skip], color='w', transform=data_crs, zorder=101, width=0.006, scale=50.0)

# units
#ax3.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_unit[::skip, ::skip], v_unit[::skip, ::skip], color='w', transform=data_crs, angles='xy', zorder=101, width=0.008, scale=50.0)#, regrid_shape=50)
#ax3.quiver(lon[::skip, ::skip], lat[::skip, ::skip], (u_unit / np.cos(lat / 180 * np.pi))[::skip, ::skip], v_unit[::skip, ::skip], color='w', transform=data_crs, angles = 'xy', zorder=101, width=0.008, scale=50.0, regrid_shape=50)

# size varying
#ax3.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_new_ad[::skip, ::skip], v_new_ad[::skip, ::skip], color='w', transform=data_crs, angles='xy', zorder=101)#, width=0.008, scale=50000.0, regrid_shape=30)
ax3.streamplot(lon, lat, u_new_ad.filled(np.nan), v_new_ad.filled(np.nan), transform=data_crs, linewidth=1, density=5, color='w', zorder=101)
#ax3.streamplot(lon, lat, u_new.filled(np.nan), v_new.filled(np.nan), transform=data_crs, linewidth=1, density=5, color='w', zorder=101)
#ax3.streamplot(lon_grid, lat_grid, u_grid_ad.filled(np.nan), v_grid_ad.filled(np.nan), transform=data_crs, linewidth=1, density=6, color='w', zorder=101)

#ax4.quiver(grid[1][::skip, 200::skip], grid[0][::skip, 200::skip], u[::skip, 200::skip], v[::skip, 200::skip], color='w', zorder=101)
ax4.streamplot(grid[1][:, 50:370], grid[0][:, 50:370], ut[:, 50:370], vt[:, 50:370], linewidth=1, density=3, color='w', zorder=101)



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

ax1.annotate('(a)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(c)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(d)', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
cax1.set_xlabel('U (m s$^{-1}$)')

fig1.colorbar(cs2, cax=cax2, orientation='horizontal')
cax2.set_xlabel('V (m s$^{-1}$)')

fig1.colorbar(cs3, cax=cax3, orientation='horizontal')
cax3.set_xlabel('Speed (m s$^{-1}$)')



# In[ ]:


fig2 = plt.figure(figsize=(12, 8))
ax1 = fig2.add_axes([0.04, 0.04, 0.45, 0.8])
#ax1 = fig2.add_axes([0.04, 0.52, 0.45, 0.4])
#ax2 = fig2.add_axes([0.04, 0.04, 0.45, 0.4])
ax3 = fig2.add_axes([0.51, 0.1, 0.45, 0.8], projection=mrc)

cax1 = fig2.add_axes([0.51, 0.96, 0.23, 0.02])

grid = np.indices(psi.shape)
ax1.contourf(grid[1], grid[0], psi, cmap=my_cm)#, vmin=0, vmax=0.3)
#ax1.pcolormesh(grid[1][:, 200:], grid[0][:, 200:], psi[:, 200:], cmap=my_cm)#, vmin=0, vmax=0.3)
#ax2.pcolormesh(grid[1][:, :200], grid[0][:, :200], psi[:, :200], cmap=my_cm)#, vmin=0, vmax=0.3)

cs3 = ax3.contourf(lon_grid, lat_grid, psi, transform=data_crs, cmap=my_cm)#, vmin=-8e3, vmax=8e3)

skip = 4
ax1.quiver(grid[1][::skip, ::skip], grid[0][::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', zorder=101)
#ax1.quiver(grid[1][::skip, 200::skip], grid[0][::skip, 200::skip], upsi[::skip, 200::skip], vpsi[::skip, 200::skip], color='w', zorder=101)
#ax2.quiver(grid[1][::skip, :200:skip], grid[0][::skip, :200:skip], upsi[::skip, :200:skip], vpsi[::skip, :200:skip], color='w', zorder=101)

ax3.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', transform=data_crs, angles='xy', regrid_shape=60, zorder=101)
#ax4.streamplot(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], u_psi_ad[::skip, ::skip], v_psi_ad[::skip, ::skip], linewidth=1, density=3, color='w', zorder=101)

ax3.add_feature(cfeature.LAND, zorder=100)
ax3.gridlines()
ax3.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax3)

fig1.colorbar(cs3, cax=cax1, orientation='horizontal')
#cax1.set_xlabel('Speed (m s$^{-1}$)')


# In[ ]:


fig1.savefig('./Figures/uv_test.png')
fig2.savefig('./Figures/stream.png')

