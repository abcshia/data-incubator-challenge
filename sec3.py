## General imports
import numpy as np
import pandas as pd
import os,inspect
import math
import pickle
import json

# Get this current script file's directory:
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Set working directory
os.chdir(loc)
os.chdir('..') # parent directory
# from myFunctions import gen_FTN_data
# from meSAX import *
# os.chdir(loc) # change back to loc


# from dtw_featurespace import *
# from dtw import dtw
# from fastdtw import fastdtw

# to avoid tk crash
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## google maps exanple code
# https://googlemaps.github.io/google-maps-services-python/docs/#
# https://developers.google.com/maps/documentation/distance-matrix/intro
import googlemaps
from datetime import datetime

# Server
with open('api-key.txt','r') as f:
    apikey = f.read()
gmaps = googlemaps.Client(key=apikey)

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="transit",
                                     departure_time=now)



# Client
gmaps = googlemaps.Client(client_id=client_id, client_secret=client_secret)

# Geocoding and address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="transit",
                                     departure_time=now)

## distance matrix query test
import googlemaps
from datetime import datetime

# address list
address_list = ['2116 Allston Way, Berkeley, CA 94704',
                '32115 Union Landing Blvd, Union City, CA 94587',
                '4370 Thornton Ave, Fremont, CA 94536',
                '668 Barber Ln, Milpitas, CA 95035',
                '447 Great Mall Dr, Milpitas, CA 95035',
                '655 Knight Way, Stanford, CA 94305',
                '2855 Stevens Creek Blvd, Santa Clara, CA 95050'
                ]


# Server access
with open('api-key.txt','r') as f:
    apikey = f.read()
gmaps = googlemaps.Client(key=apikey)

# origins = '2116 Allston Way, Berkeley, CA 94704'
# destinations = '32115 Union Landing Blvd, Union City, CA 94587'

origins = address_list[:2]
destinations = address_list[2:]

# Distance matrix querry:
dmatrix = gmaps.distance_matrix(origins, destinations, mode='driving') # departure_time, arrival_time

rows = dmatrix['rows']
o_id = 0 # origin id
d_id = 0 # destination id
distance = rows[o_id]['elements'][d_id]['distance']['value'] # [m]
duration = rows[o_id]['elements'][d_id]['duration']['value'] # [sec]


## Load/Save
# Save
with open('dmatrix.pickle','wb') as f:
    pickle.dump(dmatrix,f)
# Load
with open('dmatrix.pickle','rb') as f:
    dmatrix = pickle.load(f)


## gmplot: Plot on Google Maps
import gmplot

# Place map
gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13)

# Polygon
golden_gate_park_lats, golden_gate_park_lons = zip(*[
    (37.771269, -122.511015),
    (37.773495, -122.464830),
    (37.774797, -122.454538),
    (37.771988, -122.454018),
    (37.773646, -122.440979),
    (37.772742, -122.440797),
    (37.771096, -122.453889),
    (37.768669, -122.453518),
    (37.766227, -122.460213),
    (37.764028, -122.510347),
    (37.771269, -122.511015)
    ])
gmap.plot(golden_gate_park_lats, golden_gate_park_lons, 'cornflowerblue', edge_width=3)

# Scatter points
top_attraction_lats, top_attraction_lons = zip(*[
    (37.769901, -122.498331),
    (37.768645, -122.475328),
    (37.771478, -122.468677),
    (37.769867, -122.466102),
    (37.767187, -122.467496),
    (37.770104, -122.470436)
    ])
gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=40, marker=False)

# Marker
hidden_gem_lat, hidden_gem_lon = 37.770776, -122.461689
gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')


# api key required now
with open('api-key.txt','r') as f:
    apikey = f.read()
gmap.apikey = apikey

# Draw
gmap.draw("my_map.html")

##
import gmplot

# set map focus
gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13)

# coordinates
latitudes = golden_gate_park_lats
longitudes = golden_gate_park_lons
more_lats, more_lngs = top_attraction_lats, top_attraction_lons
marker_lats, marker_lngs = (37.770776, -122.461689)

# Plot
# gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=10)
gmap.polygon(latitudes, longitudes, 'cornflowerblue', edge_width=3)
gmap.scatter(more_lats, more_lngs, '#3B0B39', size=40, marker=False)
# gmap.scatter(marker_lats, marker_lngs, 'k', marker=True)
# gmap.heatmap(heat_lats, heat_lngs)

# api key required now
with open('api-key.txt','r') as f:
    apikey = f.read()
gmap.apikey = apikey

gmap.draw("mymap.html")

##

#Set different latitude and longitude points
Charminar_top_attraction_lats, Charminar_top_attraction_lons = zip(*[
   (17.3833, 78.4011),(17.4239, 78.4738),(17.3713, 78.4804),(17.3616, 78.4747),
   (17.3578, 78.4717),(17.3604, 78.4736),(17.2543, 78.6808),(17.4062, 78.4691),
   (17.3950, 78.3968),(17.3587, 78.2988),(17.4156, 78.4750)])
#declare the center of the map, and how much we want the map zoomed in
gmap = gmplot.GoogleMapPlotter(17.3616, 78.4747, 13)
# Scatter map
gmap.scatter( Charminar_top_attraction_lats, Charminar_top_attraction_lons, '#FF0000',size = 50, marker = False )
# Plot method Draw a line in between given coordinates
gmap.plot(Charminar_top_attraction_lats, Charminar_top_attraction_lons, 'cornflowerblue', edge_width = 3.0)

# api key required now
with open('api-key.txt','r') as f:
    apikey = f.read()
gmap.apikey = apikey


gmap.draw("mymap2.html")












































