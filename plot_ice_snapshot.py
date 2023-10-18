#!/opt/local/bin/python

# --------------------------------------------------------------------------- #

# Import required modules.
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import measure
import netCDF4 as nc
import datetime as dt
import shapely
import matplotlib.colors as colors
import zarr

# --------------------------------------------------------------------------- #

# Specify where the input data lives.


nemo_data = '/badc/cmip6/data/CMIP6/HighResMIP/NERC/HadGEM3-GC31-HM/highres-future/r1i2p1f1/'

domain = 'INPUTS/domain_cfg_noisf_points.nc'

# Specify the names of the different files that I want to load from.
# idir_day = /bodc/BAS210039/CORE2NYF-ORCH0083-LIM3/*/d01/I/

monthly = True
if monthly:
  idir = 'SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-HM_highres-future_r1i2p1f1_gn_205001-205012.nc'

#gridfile = 'TIDY/ARCHIVE/MESH/mesh_mask.nc'

out_file = './Processed/'

mask_basic = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/ZPS/ice_class_day_mask_weddell_*.npz'

mask_thick = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/ZPS/thick_class_day_mask_weddell_*.npz'

#class_file1 = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/ice_class_day_mask_weddel.npz'
#class_file2 = '/gws/nopw/j04/orchestra_vol2/anurser/AABW/ice_class_day_mask_weddell.zip'
#class_file3 = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/ZPS/ice_class_day_mask_weddell_*.npz'

#class_thick = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/ZPS/thick_class_day_mask_weddell_*.npz'

dist = '/gws/nopw/j04/orchestra_vol2/benbar/Data/Processed/Ice/coastal_distance.npz'

# Dates

st_date = dt.datetime(1980, 8, 15)
en_date = dt.datetime(2017, 12, 1)

# --------------------------------------------------------------------------- #

# Load the relevant mask.

with nc.Dataset(nemodir + gridfile, 'r') as nc_fid:
  tmask = np.squeeze(nc_fid.variables['tmask'][0, 0, :, :])
print(tmask.shape, tmask.max(), tmask.min())

# Load the grid spacings.

with nc.Dataset(nemodir + gridfile, 'r') as nc_fid:
  e1t = np.squeeze(nc_fid.variables['e1t'])
  e2t = np.squeeze(nc_fid.variables['e2t'])
  lon = nc_fid.variables['nav_lon'][:]
  lat = nc_fid.variables['nav_lat'][:] # y, x

# Select Atlantic Sector

#for i in range(lon.shape[1]):
#  print(lon[1200, i])

atl = 1
ross = 0
if atl:
  reg_str = '_atlantic'
elif ross:
  reg_str = '_ross'
else:
  reg_str = ''

if atl:

  lon_min = -70
  lon_max = 30
  lat_min = -90
  lat_max = -30

  shift = np.nonzero(lon[1200, :] == np.ma.min(lon[1200, :]))[0][0]
  lon = np.ma.append(lon[:, shift:], lon[:, :shift], axis=1)
  lat = np.ma.append(lat[:, shift:], lat[:, :shift], axis=1)
  e1t = np.ma.append(e1t[:, shift:], e1t[:, :shift], axis=1)
  e2t = np.ma.append(e2t[:, shift:], e2t[:, :shift], axis=1)
  tmask = np.ma.append(tmask[:, shift:], tmask[:, :shift], axis=1)

  xi1 = np.min(np.nonzero(lon[1200, :] >= lon_min)[0])
  xi2 = np.max(np.nonzero(lon[1200, :] < lon_max)[0])
  yi1 = np.min(np.nonzero(lat[:, 1720] >= lat_min)[0])
  yi2 = np.max(np.nonzero(lat[:, 1720] < lat_max)[0])
  lon = lon[yi1:yi2, xi1:xi2].T
  lat = lat[yi1:yi2, xi1:xi2].T
  e1t = e1t[yi1:yi2, xi1:xi2].T
  e2t = e2t[yi1:yi2, xi1:xi2].T
  tmask = tmask[yi1:yi2, xi1:xi2].T
  lon = lon.filled(0)

  def align(var, shift, xi1, yi1, xi2, yi2):
    var = np.ma.append(var[:, shift:], var[:, :shift], axis=1)
    var = var[yi1:yi2, xi1:xi2].T
    return var

elif ross:

  lon_min = 150
  lon_max = 290
  lat_min = -90
  lat_max = -65

  shift1 = np.nonzero(lon[1200, :] == np.ma.min(lon[1200, :]))[0][0]
  lon = np.ma.append(lon[:, shift1:], lon[:, :shift1], axis=1)
  lat = np.ma.append(lat[:, shift1:], lat[:, :shift1], axis=1)
  e1t = np.ma.append(e1t[:, shift1:], e1t[:, :shift1], axis=1)
  e2t = np.ma.append(e2t[:, shift1:], e2t[:, :shift1], axis=1)
  tmask = np.ma.append(tmask[:, shift1:], tmask[:, :shift1], axis=1)

  shift2 = np.nonzero(lon[1200, :] >= 0)[0][0]
  lon = np.ma.append(lon[:, shift2:], lon[:, :shift2], axis=1)
  lat = np.ma.append(lat[:, shift2:], lat[:, :shift2], axis=1)
  e1t = np.ma.append(e1t[:, shift2:], e1t[:, :shift2], axis=1)
  e2t = np.ma.append(e2t[:, shift2:], e2t[:, :shift2], axis=1)
  tmask = np.ma.append(tmask[:, shift2:], tmask[:, :shift2], axis=1)

  ind_lon = lon < 0
  lon[ind_lon] = lon[ind_lon] + 360
  lon[:, 1000:][lon[:, 1000:] < 20] = lon[:, 1000:][lon[:, 1000:] < 20] + 360
  lon = lon.filled(360)

  xi1 = np.min(np.nonzero(lon[1200, :] >= lon_min)[0])
  xi2 = np.max(np.nonzero(lon[1200, :] < lon_max)[0])
  yi1 = np.min(np.nonzero(lat[:, 1720] >= lat_min)[0])
  yi2 = np.max(np.nonzero(lat[:, 1720] < lat_max)[0])
  lon = lon[yi1:yi2, xi1:xi2].T
  lat = lat[yi1:yi2, xi1:xi2].T
  e1t = e1t[yi1:yi2, xi1:xi2].T
  e2t = e2t[yi1:yi2, xi1:xi2].T
  tmask = tmask[yi1:yi2, xi1:xi2].T

else:
  lon = lon.T
  lat = lat.T
  e1t = e1t.T
  e2t = e2t.T
  tmask = tmask.T

nx = e1t.shape[0]
ny = e1t.shape[1]
nz = 75

# --------------------------------------------------------------------------- #

# Find the files that I want to classify the ice cover for.

if monthly:
  ifiles = sorted(glob.glob(''.join([nemodir, idir, '*'])))
else:
  ifiles = sorted(glob.glob(''.join([nemo_data, idir, '*'])))

tfiles = sorted(glob.glob(''.join([nemo_data, tdir, '*'])))

cfiles1 = sorted(glob.glob(mask_basic))
cfiles2 = sorted(glob.glob(mask_thick))

date_files = np.zeros((len(ifiles)), dtype=object)
for i in range(len(ifiles)):
  date_files[i] = dt.datetime.strptime(ifiles[i].split('_')[-3], '%Y%m%d')

idate = np.nonzero((date_files >= st_date) & (date_files < en_date))[0]
ifiles = ifiles[idate[0]:idate[-1] + 1]
tfiles = tfiles[idate[0]:idate[-1] + 1]


with nc.Dataset(ifiles[0], 'r') as nc_fid:
    siconc = np.squeeze(nc_fid.variables['siconc'])
    #sithic = np.squeeze(nc_fid.variables['sithic'])
    sithic = np.squeeze(nc_fid.variables['sivolu']) # m
    si_u = np.squeeze(nc_fid.variables['sivelu'])
    si_v = np.squeeze(nc_fid.variables['sivelv'])

with nc.Dataset(tfiles[0], 'r') as nc_fid:
    pyc_dep = np.squeeze(nc_fid.variables['somxlkara'])
    salt = np.squeeze(nc_fid.variables['sosfldow']) # 1e-3/m2/s
    water = np.squeeze(nc_fid.variables['sowaflup']) * -1 # kg/m2/s 
    # now positive down


add_class = 1
if add_class:
  # add the ice class data
  for i in range(len(cfiles1)):
    if str(st_date.year) in cfiles1[i]:
      cind = i
  cfiles1 = cfiles1[cind]

  data = np.load(cfiles1, allow_pickle=True)
  date_class_basic = data['date_list']
  cat_basic = data['category']
  lon_class = data['lon']
  lat_class = data['lat']
  dind = np.nonzero(date_class_basic == st_date)[0][0]
  ice_class_basic = data['ice_class'][dind, :, :]
  data.close()

  for i in range(len(cfiles2)):
    if str(st_date.year) in cfiles2[i]:
      cind = i
  cfiles2 = cfiles2[cind]

  data = np.load(cfiles2, allow_pickle=True)
  date_class_thick = data['date_list']
  cat_thick = data['category']
  lon_class = data['lon']
  lat_class = data['lat']
  dind = np.nonzero(date_class_thick == st_date)[0][0]
  ice_class_thick = data['ice_class'][dind, :, :]
  data.close()

data = np.load(out_file + 'coastal_distance.npz')
lon_dist = data['lon']
lat_dist = data['lat']
dist_coast = data['dist_coast']
data.close()


# Atlantic Sector
if atl:
  siconc = align(siconc, shift, xi1, yi1, xi2, yi2)   
  sithic = align(sithic, shift, xi1, yi1, xi2, yi2)
  si_u = align(si_u, shift, xi1, yi1, xi2, yi2)
  si_v = align(si_v, shift, xi1, yi1, xi2, yi2)
  pyc_dep = align(pyc_dep, shift, xi1, yi1, xi2, yi2)
  salt = align(salt, shift, xi1, yi1, xi2, yi2)
  water = align(water, shift, xi1, yi1, xi2, yi2)
elif ross:
  siconc = np.ma.append(siconc[:, shift1:], siconc[:, :shift1], axis=1)
  siconc = np.ma.append(siconc[:, shift2:], siconc[:, :shift2], axis=1)
  siconc = siconc[yi1:yi2, xi1:xi2].T
  pyc_dep = np.ma.append(pyc_dep[:, shift1:], pyc_dep[:, :shift1], axis=1)
  pyc_dep = np.ma.append(pyc_dep[:, shift2:], pyc_dep[:, :shift2], axis=1)
  pyc_dep = pyc_dep[yi1:yi2, xi1:xi2].T
  salt = np.ma.append(salt[:, shift1:], salt[:, :shift1], axis=1)
  salt = np.ma.append(salt[:, shift2:], salt[:, :shift2], axis=1)
  salt = salt[yi1:yi2, xi1:xi2].T
  water = np.ma.append(water[:, shift1:], water[:, :shift1], axis=1)
  water = np.ma.append(water[:, shift2:], water[:, :shift2], axis=1)
  water = water[yi1:yi2, xi1:xi2].T
else:
  siconc = siconc.T
  pyc_dep = pyc_dep.T
  salt = salt.T
  water = water.T
siconc[tmask == 0] = 999
siconc = np.ma.masked_where(tmask == 0, siconc)
sithic[tmask == 0] = 999
sithic = np.ma.masked_where(tmask == 0, sithic)
#sithic = np.ma.masked_where(siconc <= 0.05, sithic)
si_u[tmask == 0] = 999
si_u = np.ma.masked_where(tmask == 0, si_u)
si_v[tmask == 0] = 999
si_v = np.ma.masked_where(tmask == 0, si_v)

pyc_dep[tmask == 0] = 999
pyc_dep = np.ma.masked_where(tmask == 0, pyc_dep)
salt[tmask == 0] = 999
salt = np.ma.masked_where(tmask == 0, salt)
water[tmask == 0] = 999
water = np.ma.masked_where(tmask == 0, water)

grad_u = np.gradient(si_u)[0]
grad_v = np.gradient(si_v)[1]
si_div = (grad_u / e1t) + (grad_v / e2t)

# units change

salt = (salt * 1e-3) * 1000 # from 1e-3 / m2 / s to g / m2 / s because water is in kg
melt_sal = salt / water # g / kg
print (np.ma.mean(melt_sal))
#ice_vol = (water / 1000) # kg / m2 / sec to m3 / sec
sal_ref = 35.0
icem = (((sal_ref - melt_sal) / sal_ref) * water) #/ 1e6 # Sv of freshwater



#fig1 = plt.figure(figsize=(8.0, 7.0))
#ax1 = fig1.add_axes([0.1, 0.1, 0.75, 0.8])
#ax1c = fig1.add_axes([0.87, 0.3, 0.02, 0.4])

#cs = ax1.pcolormesh(lon.T)
#cs = ax1.pcolormesh(lon[:, 1000], lat[10, :], lon.T)

#plt.show()

fig1 = plt.figure(figsize=(8.0, 7.0))
ax1 = fig1.add_axes([0.1, 0.1, 0.75, 0.8])
ax1c = fig1.add_axes([0.87, 0.3, 0.02, 0.4])

pyc_contours = [400, 1000, 4000]
cs = ax1.pcolormesh(lon[:, 1000], lat[10, :], siconc.T, zorder=99)
ax1.contour(lon[:, 1000], lat[10, :], pyc_dep.T, levels=pyc_contours, cmap=plt.get_cmap('jet'), vmin=0, vmax=4500, zorder=100)

#cs = ax1.pcolormesh(lon.T)

#ax1.set_xlim([-220, 370])
#ax1.set_xlim([-40, 20])
#ax1.set_ylim([-73, -65])
ax1.set_ylim([-80, -50])

cbar = fig1.colorbar(cs, cax=ax1c)
ax1c.set_ylabel('Ice Concentration (%)')


fig2 = plt.figure(figsize=(11.0, 9.0))
ax1 = fig2.add_axes([0.05, 0.69, 0.33, 0.27])
ax2 = fig2.add_axes([0.55, 0.69, 0.33, 0.27])
ax3 = fig2.add_axes([0.05, 0.37, 0.33, 0.27])
ax4 = fig2.add_axes([0.55, 0.37, 0.33, 0.27])
ax5 = fig2.add_axes([0.05, 0.05, 0.33, 0.27])
ax6 = fig2.add_axes([0.55, 0.05, 0.33, 0.27])
ax1c = fig2.add_axes([0.39, 0.69, 0.01, 0.27])
ax2c = fig2.add_axes([0.89, 0.69, 0.01, 0.27])
ax3c = fig2.add_axes([0.39, 0.37, 0.01, 0.27])
ax4c = fig2.add_axes([0.89, 0.37, 0.01, 0.27])
ax5c = fig2.add_axes([0.39, 0.05, 0.01, 0.27])
ax6c = fig2.add_axes([0.89, 0.05, 0.01, 0.27])


pyc_contours = [400, 1000, 4000]
cs1 = ax1.pcolormesh(lon[:, 1000], lat[10, :], siconc.T, zorder=99)
ax1.contour(lon[:, 1000], lat[10, :], pyc_dep.T, levels=pyc_contours, cmap=plt.get_cmap('jet'), vmin=0, vmax=4500, zorder=100)

cs2 = ax2.pcolormesh(lon[:, 1000], lat[10, :], sithic.T, vmin=0, vmax=3, zorder=99)
ax2.contour(lon[:, 1000], lat[10, :], sithic.T, [0.4], colors='k', zorder=99)

#cs4 = ax4.pcolormesh(lon[:, 1000], lat[10, :], salt.T*100, vmin=-0.2, vmax=0.2, zorder=99)
#cs4 = ax4.pcolormesh(lon[:, 1000], lat[10, :], water.T, vmin=-0.2, vmax=0.2, zorder=99)
cs4 = ax4.pcolormesh(lon[:, 1000], lat[10, :], icem.T*1000, vmin=-0.2, vmax=0.2, zorder=99)
ax4.contour(lon_dist[:, 1000], lat_dist[10, :], dist_coast.T, [100], colors='k', zorder=99)

cs3 = ax3.pcolormesh(lon[:, 1000], lat[10, :], si_div.T, vmin=-0.000001, vmax=0.000001, zorder=99)

if add_class:
  ice_class_thick[ice_class_thick == 9] = 8
  cat_thick = ['Land', 'Ocean', 'Loose Ice', 'MIZ', 'Pack Ice', 'Open Pack Ice', 'Coastal\nPolynya Thick', 'Inner Open\nWater']
  cat_basic = ['Land', 'Ocean', 'Loose Ice', 'MIZ', 'Pack Ice', 'Open Pack Ice', 'Coastal\nPolynya Basic', 'Inner Open\nWater']
  #mycmap = colors.ListedColormap(['darkgreen', 'blue', 'red', 'cornflowerblue', 'orange', 'lime', 'magenta', 'y'])
  mycmap = colors.ListedColormap(['silver', 'darkblue', 'c', 'r', 'b', 'darkorange', 'g', 'darkmagenta'])

  boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
  norm = colors.BoundaryNorm(boundaries, mycmap.N, clip=True)

  cs5 = ax5.pcolormesh(lon_class[:, -1], lat_class[0, :], ice_class_basic.T, vmin=0, vmax=8, cmap=mycmap, norm=norm, zorder=99)
  cbar = fig2.colorbar(cs5, cax=ax5c, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
  cbar.ax.set_yticklabels(cat_basic, fontsize=9)

  cs6 = ax6.pcolormesh(lon_class[:, -1], lat_class[0, :], ice_class_thick.T, vmin=0, vmax=8, cmap=mycmap, norm=norm, zorder=99)
  cbar = fig2.colorbar(cs6, cax=ax6c, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
  cbar.ax.set_yticklabels(cat_thick, fontsize=9)


#ax1.set_ylim([-80, -50])
#ax2.set_ylim([-80, -50])

ax1.set_xlim([-50, 20])
ax1.set_ylim([-75, -60])
ax2.set_xlim([-50, 20])
ax2.set_ylim([-75, -60])
ax3.set_xlim([-50, 20])
ax3.set_ylim([-75, -60])
ax4.set_xlim([-50, 20])
ax4.set_ylim([-75, -60])
ax5.set_xlim([-50, 20])
ax5.set_ylim([-75, -60])
ax6.set_xlim([-50, 20])
ax6.set_ylim([-75, -60])

cbar = fig1.colorbar(cs1, cax=ax1c)
ax1c.set_ylabel('Ice Concentration (%)')

cbar = fig1.colorbar(cs2, cax=ax2c)
ax2c.set_ylabel('Thickness (m)')

#cbar = fig1.colorbar(cs4, cax=ax4c)
#ax4c.set_ylabel('Salt Flux (x10$^{-5}$ m$^{2}$ s$^{-1}$)')

cbar = fig1.colorbar(cs4, cax=ax4c)
ax4c.set_ylabel('Freshwater Flux (x10$^{-3}$ kg m$^{2}$ s$^{-1}$)')

cbar = fig1.colorbar(cs3, cax=ax3c)
ax3c.set_ylabel('Ice Divergence (s$^{-1}$)')


ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/ice_conc_snap' + reg_str + '_' + st_date.strftime('%Y%m%d') + '.png')
fig2.savefig('./Figures/salt_snap' + reg_str + '_' + st_date.strftime('%Y%m%d') + '.png')
