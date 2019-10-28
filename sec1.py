## General imports
import numpy as np
import pandas as pd
import os,inspect
import math
import pickle

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

## Data cleaning

# Load csv file
fileLoc = r'C:\Users\James\Desktop\Data Incubator dataset\NYPD_Motor_Vehicle_Collisions2.csv'
# df = pd.read_csv(fileLoc,header=0,nrows=10000,encoding='utf8') # engine='python'
df = pd.read_csv(fileLoc,header=0,encoding='utf8') # engine='python'

# Data up to December 31, 2018
from datetime import datetime as dt
dateFormat = '%m/%d/%Y-%H:%M'
endDate = dt.strptime('12/31/2018-23:59',dateFormat)

indices = [i for i in df.index if dt.strptime(df.DATE[i]+'-'+df.TIME[i],dateFormat)>endDate]

# # Save
# with open(loc+'\\indices.pickle','wb') as f:
#     pickle.dump(indices,f)
# Load
with open(loc+'\\indices.pickle','rb') as f:
    indices = pickle.load(f)
    
df.drop(index=indices,inplace=True)

## Q1: total number of persons injured in the dataset 
[print(name) for name in df.columns]

np.sum(df['NUMBER OF PERSONS INJURED'])
np.sum(df['NUMBER OF PEDESTRIANS INJURED'])
np.sum(df['NUMBER OF CYCLIST INJURED'])
np.sum(df['NUMBER OF MOTORIST INJURED'])
#---------
# 368034 
#---------

## Q2&Q3: What proportion of collisions in 2016 resulted in injury or death of a cyclist?
'''
Obtain the number of vehicles involved in each collision in 2016. Group the collisions by zip code and compute the sum of all vehicles involved in collisions in each zip code, then report the maximum of these values.
'''
indices_2016 = [i for i in df.index if dt.strptime(df.DATE[i],'%m/%d/%Y').year==2016]
# Save
with open(loc+'\\indices_2016.pickle','wb') as f:
    pickle.dump(indices_2016,f)
# Load
with open(loc+'\\indices_2016.pickle','rb') as f:
    indices_2016 = pickle.load(f)
df_2016 = df.loc[indices_2016]


df_v = df_2016[['VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5']]

# total number of vehicles involved collisions in 2016
total_v_number_collision = np.array(df_v.isna()==False).sum()
# number of vehicles involved in each collision in 2016
v_number_collision = np.array(df_v.isna()==False).sum(axis=1)

# unique zipcodes
zip_arr = np.array(df_2016['ZIP CODE'])
mask = ~np.array(df_2016['ZIP CODE'].isna())
zipcodes = np.unique(zip_arr[mask])
zipcodes = np.array(zipcodes,dtype=np.int)

zip_dict={00000:0} # key=zipcode,val=number of collisions # assign 00000 for NAN
for i,zipcode in enumerate(zip_arr):
    if np.isnan(zipcode):
        zip_dict[00000] += v_number_collision[i]
    elif zipcode in zip_dict.keys():
        zip_dict[zipcode] += v_number_collision[i]
    else:
        zip_dict[zipcode] = v_number_collision[i]

inv_zip_dict = {val:key for key,val in zip_dict.items()} # key=number of collisions,val=zipcode
# max_collision = np.max(list(inv_zip_dict.keys()))
# inv_zip_dict[max_collision]

# Sort
df_zip_col = pd.DataFrame(zip_dict.items(),columns=['ZIP CODE','Collisions'])
df_zip_col.sort_values(by='Collisions',ascending=False)

# --------------------
# ZIP CODE  Collisions
# 11207.0        5703
# --------------------

np.sum(df_2016['NUMBER OF CYCLIST INJURED']) # total cyclist injured
np.sum(df_2016['NUMBER OF CYCLIST KILLED']) # total cyclist killed
df_2016['NUMBER OF CYCLIST INJURED'][df_2016['NUMBER OF CYCLIST INJURED']>0] # collisions that injured cyclist(s) 
df_2016['NUMBER OF CYCLIST KILLED'][df_2016['NUMBER OF CYCLIST KILLED']>0] # collisions that killed cyclist(s)
# number of collisions that injured cyclist(s) 
n_injured = df_2016['NUMBER OF CYCLIST INJURED'][df_2016['NUMBER OF CYCLIST INJURED']>0].shape[0]
# number of collisions that killed cyclist(s) 
n_killed = df_2016['NUMBER OF CYCLIST KILLED'][df_2016['NUMBER OF CYCLIST KILLED']>0].shape[0]
# number of collisions that injured or killed cyclist(s) 
mask = df_2016[['NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED']].sum(axis=1)>0
n_injured_or_killed = df_2016[mask].shape[0]

# ---------------------------------------------------
# number of collisions that injured cyclist(s): 4956
# number of collisions that killed cyclist(s): 20
# number of collisions that injured or killed cyclist(s): 4976
# ---------------------------------------------------
print('Proportion of collisions in 2016 resulted in injury or death:')
print('Injured: {}%'.format(n_injured/df_2016.shape[0]*100))
print('Death: {}%'.format(n_killed/df_2016.shape[0]*100))
print('Injured or Death: {}%'.format(n_injured_or_killed/df_2016.shape[0]*100))
# -------------------------------------------------------------
# Proportion of collisions in 2016 resulted in injury or death:
# Injured: 2.1567705885424826%
# Death: 0.008703674691454732%
# Injured or Death: 2.1654742632339374%
# -------------------------------------------------------------



## Q4: Do winter driving conditions lead to more multi-car collisions?
'''
Compute the rate of multi car collisions as the proportion of the number of collisions involving 3 or more cars to the total number of collisions for each month of 2017. Calculate the chi-square test statistic for testing whether a collision is more likely to involve 3 or more cars in January than in May.
'''
indices_2017 = [i for i in df.index if dt.strptime(df.DATE[i],'%m/%d/%Y').year==2017]
# Save
with open(loc+'\\indices_2017.pickle','wb') as f:
    pickle.dump(indices_2017,f)
# Load
with open(loc+'\\indices_2017.pickle','rb') as f:
    indices_2017 = pickle.load(f)
df_2017 = df.loc[indices_2017]


df_v = df_2017[['VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5']]
# total number of vehicles involved collisions in 2017
total_v_number_collision = np.array(~df_v.isna()).sum()
# number of vehicles involved in each collision in 2017
v_number_collision = np.array(~df_v.isna()).sum(axis=1)


# collisions involving 3 or more cars in 2017
df_2017_3more = df_2017['VEHICLE TYPE CODE 3'][~df_2017['VEHICLE TYPE CODE 3'].isna()]
df_2017['DATE'].loc[df_2017_3more.index]


# months
# months = ['{:02d}'.format(i) for i in range(1,13)]

mon_dict={} # key=month,val=collisions
for i in df_2017.index:
    month = dt.strptime(df_2017['DATE'].loc[i],'%m/%d/%Y').month
    if month in mon_dict.keys():
        mon_dict[month] += 1
    else:
        mon_dict[month] = 1

total_collisions=0 # total collisions in 2017 / Sanity check
for k,v in mon_dict.items():
    print('{}:{}'.format(k,v))
    total_collisions+=v

mon_dict_3more={} # key=month,val=collisions
for i in df_2017_3more.index:
    month = dt.strptime(df_2017['DATE'].loc[i],'%m/%d/%Y').month
    if month in mon_dict_3more.keys():
        mon_dict_3more[month] += 1
    else:
        mon_dict_3more[month] = 1

total_collisions_3more=0 # total collisions involing 3 or more cars in 2017 / Sanity check
for k,v in mon_dict_3more.items():
    print('{}:{}'.format(k,v))
    total_collisions_3more+=v

# rate of multi car collisions as the proportion of the number of collisions involving 3 or more cars to the total number of collisions for each month of 2017
for month in range(1,13):
    print('month {}: {:.10f}%'.format(month,mon_dict_3more[month]/mon_dict[month]*100))
# ----------------
# month 1: 35.39%
# month 2: 6.49%
# month 3: 4.75%
# month 4: 5.33%
# month 5: 5.41%
# month 6: 5.40%
# month 7: 5.51%
# month 8: 5.52%
# month 9: 5.85%
# month 10: 5.77%
# month 11: 5.98%
# month 12: 6.02%
# ----------------


from scipy.stats import chisquare

# obs = np.empty((12,2),dtype=np.int)
# for i in range(12):
#     obs[i,0] = mon_dict_3more[i+1]
#     obs[i,1] = mon_dict[i+1]

# Jan vs May
# obs = np.empty((2,2),dtype=np.int)
# obs[:,0] = np.array([mon_dict_3more[1],mon_dict_3more[5]])
# obs[:,1] = np.array([mon_dict[1],mon_dict[5]])
# chisquare(obs,ddof=1)

obs = np.array([mon_dict_3more[1],mon_dict_3more[5]])
exp = np.array([mon_dict[1],mon_dict[5]])
print(chisquare(obs,f_exp=exp,ddof=1))

# ----------------------------------------------------------------
# Power_divergenceResult(statistic=26127.092042289787, pvalue=nan)
# ----------------------------------------------------------------


## Q5: What proportion of all collisions in 2016 occurred in Brooklyn? Only consider entries with a non-null value for BOROUGH.

df_brooklyn_2016 = df_2016[df_2016['BOROUGH']=='BROOKLYN']
print('{:.8f}%'.format(df_brooklyn_2016.shape[0]/df_2016.shape[0]*100))

# ------------
# 20.65425523%
# ------------

## Q6: For each borough, compute the number of accidents per capita involving alcohol in 2017. Report the highest rate among the 5 boroughs. Use populations as given by https://en.wikipedia.org/wiki/Demographics_of_New_York_City.

# population data from https://en.wikipedia.org/wiki/Demographics_of_New_York_City (2017)
population = {'BRONX':1471160,'BROOKLYN':2648771,'MANHATTAN':1664727,'QUEENS':2358582,'STATEN ISLAND':479458}

df_bor_2017 = df_2017['BOROUGH']


con_factors = ['CONTRIBUTING FACTOR VEHICLE 1',
               'CONTRIBUTING FACTOR VEHICLE 2',
               'CONTRIBUTING FACTOR VEHICLE 3',
               'CONTRIBUTING FACTOR VEHICLE 4',
               'CONTRIBUTING FACTOR VEHICLE 5'
               ]

true_table_alcohol_2017 = (df_2017[con_factors]=='Alcohol Involvement').sum(axis=1)>0
df_alcohol_2017 = df_2017[true_table_alcohol_2017]
df_alcohol_borough_2017 = df_alcohol_2017['BOROUGH']

# for case in df_2017[con_factors].iterrows():
#     if np.sum(case=='Alcohol Involvement')>0:
#         print(case)
# 
# for factor in con_factors:
#     index = df_2017[factor][df_2017[factor] == 'Alcohol Involvement'].index
#     print(np.array(index))
#     # print('{}'.format())
    
# number of accidents per capita involving alcohol in 2017
for k,pop in population.items():
    cases = df_alcohol_borough_2017[df_alcohol_borough_2017==k].shape[0]
    print('Accidents per capita involving alcohol in {} in 2017: {:.10f}% ({}/{})'.format(k,cases/pop*100,cases,pop))


# overall number of accidents per capita in 2017
for k,pop in population.items():
    cases = df_bor_2017[df_bor_2017==k].shape[0]
    print('Accidents per capita in {} in 2017: {:.10f}% ({}/{})'.format(k,cases/pop*100,cases,pop))

# -------------------------------------------------------------------------------------------
# Accidents per capita involving alcohol in BRONX in 2017: 0.0186247587% (274/1471160)
# Accidents per capita involving alcohol in BROOKLYN in 2017: 0.0227275216% (602/2648771)
# Accidents per capita involving alcohol in MANHATTAN in 2017: 0.0155581065% (259/1664727)
# Accidents per capita involving alcohol in QUEENS in 2017: 0.0217079584% (512/2358582)
# Accidents per capita involving alcohol in STATEN ISLAND in 2017: 0.0208568842% (100/479458)
# Accidents per capita in BRONX in 2017: 1.4543625438% (21396/1471160)
# Accidents per capita in BROOKLYN in 2017: 1.6955788175% (44912/2648771)
# Accidents per capita in MANHATTAN in 2017: 1.9086613000% (31774/1664727)
# Accidents per capita in QUEENS in 2017: 1.6375941138% (38624/2358582)
# Accidents per capita in STATEN ISLAND in 2017: 1.3014695761% (6240/479458)
# -------------------------------------------------------------------------------------------


## Q7: Consider the total number of collisions each year from 2013-2018. Is there an apparent trend? Fit a linear regression for the number of collisions per year and report its slope.

def get_year(date):
    year = int(dt.strptime(date,'%m/%d/%Y').year)
    return(year)

df_year = df['DATE'].apply(get_year)

years = range(2013,2019)
year_collisions={}
for year in years:
    year_collisions[year] = df_year[df_year==year].shape[0]
    
from sklearn.linear_model import LinearRegression
clf = LinearRegression()

X = np.array(list(year_collisions.keys())).reshape(-1,1)
y = np.array(list(year_collisions.values())).reshape(-1,1)

clf.fit(X,y)

plt.scatter(X,y)
plt.plot(X,clf.predict(X))
plt.show()

print('slope: {}\nintercept:{}'.format(clf.coef_,clf.intercept_))

# ------------------------------
# slope: [[6447.91428571]]
# intercept:[-12775821.07619048]
# ------------------------------


## Q8: We can use collision locations to estimate the areas of the zip code regions. Represent each as an ellipse with semi-axes given by a single standard deviation of the longitude and latitude. For collisions in 2017, estimate the number of collisions per square kilometer of each zip code region. Considering zipcodes with at least 1000 collisions, report the greatest value for collisions per square kilometer. Note: Some entries may have invalid or incorrect (latitude, longitude) coordinates. Drop any values that are invalid or seem unreasonable for New York City.

# https://www.latlong.net/place/new-york-city-ny-usa-1848.html
# New York City, NY, USA
# Latitude and longitude coordinates are: 40.730610, -73.935242.

# df_2017['ZIP CODE']
# df_2017[~df_2017['ZIP CODE'].isna()]
# df_zip_2017 = df_2017['ZIP CODE'].dropna() # or df_2017['ZIP CODE'][~df_2017['ZIP CODE'].isna()]
# 
# df_lat_long_2017 = df_2017[['LATITUDE','LONGITUDE']]
# mask = (~df_lat_long_2017.isna()).sum(axis=1)==2 # dropping nan values
# df_lat_long_2017[mask] # or just df_2017['LATITUDE'].dropna()


# Drop all nans for zip code, latitude, and longitude:
df_lat_long_zip_2017 = df_2017[['LATITUDE','LONGITUDE','ZIP CODE']]
mask = (~df_lat_long_zip_2017.isna()).sum(axis=1)==3 # dropping nan values, all True for each row(no nans)
df_lat_long_zip_2017 = df_lat_long_zip_2017[mask]
# Drop zero values
df_lat_long_zip_2017 = df_lat_long_zip_2017[df_lat_long_zip_2017['LATITUDE']!=0]

# Visual sanity check
lat = df_lat_long_zip_2017['LATITUDE']
long = df_lat_long_zip_2017['LONGITUDE']
plt.scatter(lat,long)
plt.show()


zipcodes = np.unique(df_lat_long_zip_2017['ZIP CODE'])
zipcodes = np.array(zipcodes,dtype=np.int)
# find number of collisions in each zip code area
zip2latlong={} # key=zipcode, val=corresponding lat-long dataframe
for zipcode in zipcodes:
    collisions = (df_lat_long_zip_2017['ZIP CODE'] == zipcode).sum()
    print('{} collisions at {}'.format(collisions,zipcode))
    if collisions >= 1000: # consider with at least 1000 collisions
        zip2latlong[zipcode] = df_lat_long_zip_2017[df_lat_long_zip_2017['ZIP CODE'] == zipcode]

# distance approximation: 
# theta: latitude angle, phi: longitude angle
# d(lat) ~ R*cos(phi)*d(theta)
# d(long) ~ R*d(phi)
# R: approximated radius in New York area
# ellipse area = pi*A*B, A and B are semi-axes
# Reference: https://en.wikipedia.org/wiki/Earth_radius
re = 6378.1370 # [km], Equatorial radius
rp = 6356.7523 # [km], Polar radius
phi = 40.730610*math.pi/180 # geodetic latitude φ of New York
R = math.sqrt(((re**2*math.cos(phi))**2+(rp**2*math.sin(phi))**2) / ((re*math.cos(phi))**2+(rp*math.sin(phi))**2))

zip2stdcol={} # key=zipcode, val= [lat std,long std,#collisions,area]
for zip,data in zip2latlong.items():
    stds = list(data.std(axis=0)[:2])
    collisions = data.shape[0]
    area = math.pi*R*math.cos(phi)*stds[0]*R*stds[1]
    stds.append(collisions)
    stds.append(area)
    zip2stdcol[zip] = stds
df_zip_area_2017 = pd.DataFrame(zip2stdcol)
df_zip_area_2017 = df_zip_area_2017.T
df_zip_area_2017.reset_index

df_zip_area_2017.columns = ['LAT STD','LONG STD','COLLISIONS','AREA']
df_zip_area_2017['ZIP CODE'] = df_zip_area_2017.index


df_zip_area_2017['DENSITY']=df_zip_area_2017['COLLISIONS']/df_zip_area_2017['AREA']
print(df_zip_area_2017.sort_values(by='DENSITY',ascending=False))

# ---------------------------------------------
# 1.562786 [collisions/sq.km] at ZIP code 10022
# ---------------------------------------------




















