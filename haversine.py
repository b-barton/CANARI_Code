import numpy as np
from math import *

def dist(lon1,lat1,lon2,lat2):
  
  lon1_rad = np.radians(lon1)
  lon2_rad = np.radians(lon2)
  lat1_rad = np.radians(lat1)
  lat2_rad = np.radians(lat2)
  #Assumes degrees input
  #Calculates in metres
  R = 6371000 #Radius of earth in metres (roughly)   
  ## Uses Haversine formula
  a1 = (sin((lat2_rad-lat1_rad)/2))**2 
  a2 = (cos(lat1_rad))*(cos(lat2_rad))*((sin((lon2_rad-lon1_rad)/2))**2)
  a = a1 + a2
  c = 2*atan2(sqrt(a),sqrt(1-a))
  d = R*c
  
  return d

def area(lon_in,lat_in):

  ## At the moment it assumes lat and lon are 1d. If 2d, need to do something about populating dx and dy  
  if np.ndim(lon_in) == 1:
    lon = lon_in.copy()
    ## If it's 1d
    londim = lon.shape[0]
    lon_start = lon[0]-(lon[1]-lon[0])/2
    lon_end = lon[-1]+(lon[-1]-lon[-2])/2
    lon_av = np.array((lon[-(londim-1):]+lon[0:londim-1])/2)#lon_av = np.array((lon[1:-1]+lon[0:-2])/2)
    lon_loop = np.concatenate(([lon_start],lon_av,[lon_end]))
  else:
    #Do this bit
    lon = lon_in.copy()
    londim = lon.shape[1]
  
  if np.ndim(lat_in) == 1:
    lat = lat_in.copy()
    latdim = lat.shape[0]
    lat_start = lat[0]-(lat[1]-lat[0])/2
    lat_end = lat[-1]+(lat[-1]-lat[-2])/2
    lat_av = (lat[-(latdim-1):]+lat[0:latdim-1])/2
    lat_loop = np.concatenate(([lat_start],lat_av,[lat_end]))
  else:
    #Do this bit
    lat = lat_in.copy()
    latdim = lat.shape[0]
  
  dx = -999*np.ones([latdim,londim])
  dy = -999*np.ones([latdim,londim])

  #Loop over longitudes to populate latitude distances
  for loncnt in range(0,londim):
    for lat_inc in range(0,latdim):
      dy[lat_inc,loncnt] = dist(lon[loncnt],lat_loop[lat_inc],lon[loncnt],lat_loop[lat_inc+1]) 
#      dy[lat_inc,loncnt] = CalculateArea.dist(float(lon[loncnt]),float(lat_loop[lat_inc]),float(lon[loncnt]),float(lat_loop[lat_inc+1])) 

  for latcnt in range(0,latdim):
    for lon_inc in range(0,londim):
      dx[latcnt,lon_inc] = dist(lon_loop[lon_inc],lat[latcnt],lon_loop[lon_inc+1],lat[latcnt]) 
#      dx[latcnt,lon_inc] = CalculateArea.dist(lon_loop[lon_inc],lat[latcnt],lon_loop[lon_inc+1],lat[latcnt]) 

  return dx, dy

