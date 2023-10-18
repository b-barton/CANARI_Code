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
    e1t = nc_fid.variables[v_map['e1t']][0, 1:-1, 1:-1] # t, y, x
    e2t = nc_fid.variables[v_map['e2t']][0, 1:-1, 1:-1]
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

e1t = e1t[y1:y2:sub, ::sub]
e2t = e2t[y1:y2:sub, ::sub]


# In[ ]:


# Co-locate u and v onto t-grid

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
def get_angle(lon, lat):
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
    return grid_angle

grid_angle = get_angle(lon, lat)
u_new, v_new = rotate_vel(ut, vt, grid_angle)


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

u_new_ad, v_new_ad = polar_uv(u_new, v_new, lat)


def stream_to_uv(stream, x, y):
    if np.ma.is_masked(stream):
        stream_f = smudge.sea_over_land(stream[:, :, np.newaxis], stream[:, :], npoints=1)[:, :, 0]
        stream_f = stream_f.filled(0)
    else:
        stream_f = stream
    u_st = np.gradient(stream_f, axis=0) / np.gradient(y, axis=0)
    v_st = -np.gradient(stream_f, axis=1) / np.gradient(x, axis=1)
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


# Test some constants

y = np.arange(10, 40, 1)
x = np.arange(-20, 0, 1)
xx, yy = np.meshgrid(x, y)

def distance_2d(x_point, y_point, x, y):
    return np.hypot(x-x_point, y-y_point)


stream1 = distance_2d(-15, 20, xx, yy)
stream2 = distance_2d(-30, 40, xx, yy)
stream3 = distance_2d(-10, 10, xx, yy) + distance_2d(-10, 40, xx, yy) - distance_2d(-15, 20, xx, yy)

print(np.mean(stream1))
stream2 = np.ma.masked_where(stream1 <= np.mean(stream1) / 4, stream2)

u_st1, v_st1 = stream_to_uv(stream1, xx, yy)
u_st2, v_st2 = stream_to_uv(stream2, xx, yy)
u_st3, v_st3 = stream_to_uv(stream3, xx, yy)

phi1 = (uv_to_stream(u_st1, v_st1, xx, yy) + uv_to_stream(u_st1, v_st1, xx, yy, use_v=True)) / 2
phi2 = (uv_to_stream(u_st2, v_st2, xx, yy) + uv_to_stream(u_st2, v_st2, xx, yy, use_v=True)) / 2
phi3 = (uv_to_stream(u_st3, v_st3, xx, yy) + uv_to_stream(u_st3, v_st3, xx, yy, use_v=True)) / 2

u_se1, v_se1 = stream_to_uv(phi1, xx, yy)
u_se2, v_se2 = stream_to_uv(phi2, xx, yy)
u_se3, v_se3 = stream_to_uv(phi3, xx, yy)

print(u_st1.shape, xx.shape)
      
#data_crs = ccrs.PlateCarree()
#mrc = ccrs.NorthPolarStereo(central_longitude=0.0)

fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_axes([0.03, 0.69, 0.2, 0.28])
ax2 = fig1.add_axes([0.28, 0.69, 0.2, 0.28])
ax3 = fig1.add_axes([0.53, 0.69, 0.2, 0.28])
ax4 = fig1.add_axes([0.78, 0.69, 0.2, 0.28])

ax5 = fig1.add_axes([0.03, 0.36, 0.2, 0.28])
ax6 = fig1.add_axes([0.28, 0.36, 0.2, 0.28])
ax7 = fig1.add_axes([0.53, 0.36, 0.2, 0.28])
ax8 = fig1.add_axes([0.78, 0.36, 0.2, 0.28])

ax9 = fig1.add_axes([0.03, 0.03, 0.2, 0.28])
ax10 = fig1.add_axes([0.28, 0.03, 0.2, 0.28])
ax11 = fig1.add_axes([0.53, 0.03, 0.2, 0.28])
ax12 = fig1.add_axes([0.78, 0.03, 0.2, 0.28])

skip = 8
ax1.pcolormesh(xx, yy, stream1, cmap=plt.cm.plasma)
ax5.pcolormesh(xx, yy, stream2, cmap=plt.cm.plasma)
ax9.pcolormesh(xx, yy, stream3, cmap=plt.cm.plasma)

ax2.quiver(xx, yy, u_st1, v_st1, color='k', zorder=101)
ax6.quiver(xx, yy, u_st2, v_st2, color='k', zorder=101)
ax10.quiver(xx, yy, u_st3, v_st3, color='k', zorder=101)

ax3.pcolormesh(xx, yy, phi1, cmap=plt.cm.plasma)
ax7.pcolormesh(xx, yy, phi2, cmap=plt.cm.plasma)
ax11.pcolormesh(xx, yy, phi3, cmap=plt.cm.plasma)

ax4.quiver(xx, yy, u_se1, v_se1, color='k', zorder=101)
ax8.quiver(xx, yy, u_se2, v_se2, color='k', zorder=101)
ax12.quiver(xx, yy, u_se3, v_se3, color='k', zorder=101)




#ax1.annotate('u1 v0', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax5.annotate('u0 v1', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax9.annotate('u1 v1', (0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

#ax1.set_title('Plain')
#ax2.set_title('Polar Correct')
#ax3.set_title('Grid Rotate')
#ax4.set_title('GR and PC')

fig1.savefig('./Figures/stream_test.png')

#sys.exit()


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


# In[ ]:


# stick tops together

half = u_new.shape[0]
i_pol = int(u_new.shape[0] * 2)
j_pol = int(u_new.shape[1] * 0.5)
u_pol = np.ma.zeros((i_pol, j_pol))
v_pol = np.ma.zeros((i_pol, j_pol))
angle_pol = np.ma.zeros((i_pol, j_pol))

u_pol[:half, :] = ut[:, j_pol:]
u_pol[half:, :] = np.fliplr(np.flipud(ut[:, :j_pol])) * -1
v_pol[:half, :] = vt[:, j_pol:]
v_pol[half:, :] = np.fliplr(np.flipud(vt[:, :j_pol])) * -1
angle_pol[:half, :] = grid_angle[:, j_pol:]
angle_pol[half:, :] = np.fliplr(np.flipud(grid_angle[:, :j_pol]))

lon_pol = np.ma.zeros((i_pol, j_pol))
lat_pol = np.ma.zeros((i_pol, j_pol))

lon_pol[:half, :] = lon[:, j_pol:]
lon_pol[half:, :] = np.fliplr(np.flipud(lon[:, :j_pol]))
lat_pol[:half, :] = lat[:, j_pol:]
lat_pol[half:, :] = np.fliplr(np.flipud(lat[:, :j_pol]))

e1t_pol = np.ma.zeros((i_pol, j_pol))
e2t_pol = np.ma.zeros((i_pol, j_pol))

e1t_pol[:half, :] = e1t[:, j_pol:]
e1t_pol[half:, :] = np.fliplr(np.flipud(e1t[:, :j_pol]))
e2t_pol[:half, :] = e2t[:, j_pol:]
e2t_pol[half:, :] = np.fliplr(np.flipud(e2t[:, :j_pol]))

#lat_pol = lat_pol[:half, :]
#lon_pol = lon_pol[:half, :]
#u_pol = u_pol[:half, :]
#v_pol = v_pol[:half, :]
#e1t_pol = e1t_pol[:half, :]
#e2t_pol = e2t_pol[:half, :]
#angle_pol = angle_pol[:half, :]
print(lon_pol.shape, u_pol.shape)


# Mask areas outside Arctic Basin



fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_axes([0.04, 0.04, 0.45, 0.8])
ax2 = fig1.add_axes([0.51, 0.1, 0.45, 0.8], projection=mrc)

grid = np.indices(u_pol.shape)
    
skip = 2
ax1.quiver(grid[1][::skip, ::skip], grid[0][::skip, ::skip], u_pol[::skip, ::skip], v_pol[::skip, ::skip], color='k', zorder=101)

u_pol_fr = u_pol * 1
u_pol_fr[half:, :] = u_pol_fr[half:, :] * -1
v_pol_fr = v_pol * 1
v_pol_fr[half:, :] = v_pol_fr[half:, :] * -1

u_pol_p, v_pol_p = rotate_vel(u_pol_fr, v_pol_fr, angle_pol)
u_pol_ad, v_pol_ad = polar_uv(u_pol_p, v_pol_p, lat_pol)
ax2.quiver(lon_pol[::skip, ::skip], lat_pol[::skip, ::skip], u_pol_ad[::skip, ::skip], v_pol_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy')#, regrid_shape=60, zorder=101)



skip = 4
#ax1.quiver(grid[1][::skip, ::skip], grid[0][::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', zorder=101)
#ax1.quiver(grid[1][::skip, 200::skip], grid[0][::skip, 200::skip], upsi[::skip, 200::skip], vpsi[::skip, 200::skip], color='w', zorder=101)
#ax2.quiver(grid[1][::skip, :200:skip], grid[0][::skip, :200:skip], upsi[::skip, :200:skip], vpsi[::skip, :200:skip], color='w', zorder=101)

#ax3.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], upsi[::skip, ::skip], vpsi[::skip, ::skip], color='w', transform=data_crs, angles='xy', regrid_shape=60, zorder=101)
#ax4.streamplot(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], u_psi_ad[::skip, ::skip], v_psi_ad[::skip, ::skip], linewidth=1, density=3, color='w', zorder=101)

ax2.add_feature(cfeature.LAND, zorder=100)
ax2.gridlines()
ax2.set_extent([-180, 180, 60, 90], crs=data_crs)
set_circle(ax2)

fig1.savefig('./Figures/uv_fold_together.png')

#sys.exit()


# In[ ]:


if 0:
    # Generate new grid on NSIDC Polar Stereographic projection on WGS84
    crs_wgs84 = crs.CRS('epsg:4326')
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
    
if 0:
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


use_grid = False
use_pol = False
if 0:
    use_grid = True
    psi, upsi, vpsi, phi, uphi, vphi = uv2psiphi(lon_grid, lat_grid, u_grid.filled(np.nan), v_grid.filled(np.nan), ZBC='periodic')
    #psi = (uv_to_stream(u_grid, v_grid, e1t, e2t) + uv_to_stream(u_grid, v_grid, e1t, e2t, use_v=True)) / 2 
    psi = np.ma.masked_where(u_grid.mask, psi)
    upsi = np.ma.masked_where(u_grid.mask, upsi)
    vpsi = np.ma.masked_where(v_grid.mask, vpsi)
    
    # the edges look odd
    upsi[:, 0] = upsi[:, 1] * 1
    upsi[:, -1] = upsi[:, -2] * 1
    vpsi[:, 0] = vpsi[:, 1] * 1
    vpsi[:, -1] = vpsi[:, -2] * 1

    u_psi_ad, v_psi_ad = polar_uv(upsi, vpsi, lat_grid)
elif 1:
    glamf, gphif, psi1, opt_dic = cdfpsi_mod.cdfpsi('', flist_u[0], 'uo', flist_v[0], 'vo', fn_nemo_dom, kt=0, ll_v=False)
    glamf, gphif, psi2, opt_dic = cdfpsi_mod.cdfpsi('', flist_u[0], 'uo', flist_v[0], 'vo', fn_nemo_dom, kt=0, ll_v=True)
    psi = psi1 #(psi1 + psi2) / 2
    psi = psi[y1:y2:sub, ::sub]
else:
    use_pol = True

    psi = (uv_to_stream(u_pol, v_pol, e1t_pol, e2t_pol, dxdy=True) + uv_to_stream(u_pol, v_pol, e1t_pol, e2t_pol, dxdy=True, use_v=True)) / 2
    


# Plot

# In[ ]:


fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_axes([0.04, 0.04, 0.45, 0.8])
#ax1 = fig2.add_axes([0.04, 0.52, 0.45, 0.4])
#ax2 = fig2.add_axes([0.04, 0.04, 0.45, 0.4])
ax3 = fig1.add_axes([0.51, 0.1, 0.45, 0.8], projection=mrc)

cax1 = fig1.add_axes([0.51, 0.96, 0.23, 0.02])

grid = np.indices(psi.shape)
ax1.contourf(grid[1], grid[0], psi, cmap=my_cm)#, vmin=0, vmax=0.3)
#ax1.pcolormesh(grid[1][:, 200:], grid[0][:, 200:], psi[:, 200:], cmap=my_cm)#, vmin=0, vmax=0.3)
#ax2.pcolormesh(grid[1][:, :200], grid[0][:, :200], psi[:, :200], cmap=my_cm)#, vmin=0, vmax=0.3)


if use_grid:
    cs3 = ax3.contourf(lon_grid, lat_grid, psi, transform=data_crs, cmap=my_cm)#, vmin=-2e4, vmax=2e5)
elif use_pol:
    u_pol_fr = u_pol * 1
    u_pol_fr[half:, :] = u_pol_fr[half:, :] * -1
    v_pol_fr = v_pol * 1
    v_pol_fr[half:, :] = v_pol_fr[half:, :] * -1

    u_pol_p, v_pol_p = rotate_vel(u_pol_fr, v_pol_fr, angle_pol)
    u_pol_ad, v_pol_ad = polar_uv(u_pol_p, v_pol_p, lat_pol)
    skip = 2
    cs3 = ax3.contourf(lon_pol, lat_pol, psi, transform=data_crs, cmap=my_cm)
    ax3.quiver(lon_pol[::skip, ::skip], lat_pol[::skip, ::skip], u_pol_ad[::skip, ::skip], v_pol_ad[::skip, ::skip], color='k', transform=data_crs, angles='xy')#, regrid_shape=60, zorder=101)
else:
    cs3 = ax3.pcolormesh(lon, lat, psi, transform=data_crs, cmap=my_cm)#, vmin=-2e4, vmax=2e5)

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


fig1.savefig('./Figures/stream_nemo.png')

