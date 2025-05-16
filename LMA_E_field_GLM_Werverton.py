# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:45:40 2023

@author: lealn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load LMA data

filename='C:/Users/suporte/GIT/masterdegree/LL_240720_234000_0600.dat/LL_240720_234000_0600.dat'

headerlength = 55  # Note that this is 10 for a LYLOUT 
                   # but 45 for a processed LMA file ... and it may
                   # vary!
datatypes = (float, float, float, float, float, int, float)
# With this specifier, genfromtxt already converts every array from
# text to the proper type.
sfm, lat, lon, alt, Xisq, nstn, dBW = np.genfromtxt(filename,dtype=datatypes,unpack=True,skip_header=headerlength,comments="#",usecols=[0,1,2,3,4,5,6])
data = {'sfm':sfm,'lat':lat,'lon':lon,'alt':alt, 'Xisq':Xisq, 'nstn':nstn, 'dBW':dBW}


#Filter based on power dBW, nstn (number of stations)
df=pd.DataFrame(data)
filtered_df = df.query('nstn > 6')


# Seconds from midnight to real seconds
# "Real" Seconds is saved on s
m, s = divmod(filtered_df['sfm'], 60)
h, m = divmod(m, 60)


#### Load E-field data
#Ts = 2023,8,22,5,4,35,640160-2208

#ts time stamp E-field (seconds.microseconds)
ts=35.640160

#Time before (tb) and after (ta) the TS to be ploted (x axis)
#It depends on each record
tb = ts - 00e-3
ta = ts + 400e-3


#waveform file name
ID = '2023,8,22,5,4,35,640160_Efield'

#Load E-field file (Pico scope file)
tipo={'time': 'string', 'Ch A':  'string', 'Ch B':  'string', 'Ch C':  'string'}
column_headers= ['time','Ch A', 'Ch B', 'Ch C']
#Efield = pd.read_csv(f'{ID}.csv',sep = ',', header=None, names=column_headers,engine='c',dtype=tipo, skiprows=3)
#Efield = pd.DataFrame(Efield)

    
# This part of the code fix the saturation levels
pA = '5'
pB = '2'
pC = '5'

nA = '-5'
nB = '-2'
nC = '-5'

# for i in range (len(Efield)):
    
#     if Efield["Ch A"].values[i] == '-∞': Efield["Ch A"].values[i] = nA
#     if Efield["Ch B"].values[i] == '-∞': Efield["Ch B"].values[i] = nB
#     if Efield["Ch C"].values[i] == '-∞': Efield["Ch C"].values[i] = nC
    
#     if Efield["Ch A"].values[i] == '∞': Efield["Ch A"].values[i] = pA
#     if Efield["Ch B"].values[i] == '∞': Efield["Ch B"].values[i] = pB
#     if Efield["Ch C"].values[i] == '∞': Efield["Ch C"].values[i] = pC


#Set all the values as float
# c = Efield.columns

# for i in c:  
#     Efield[i] = Efield[i].astype(float)


# Auxiliary vector to set time axis
pretrigger = -600
postrigger = 1400

# aux = (Efield["time"].values >= pretrigger) & (Efield["time"].values <= postrigger) 
# Efield = Efield[aux]


#Make time axis (creating time vector)

t_left = ts-0.6
t_right = ts+1.4


# length_pre = int(len(Efield)*0.3)
# vector_time_pre = np.linspace(t_left,ts,length_pre)


# length_pos = int(len(Efield)*0.7)
# vector_time_pos = np.linspace(ts,t_right,length_pos)


# vector_time_total = np.append(vector_time_pre, vector_time_pos)

#########################################
# Read GLM group data

# dados_GLM = pd.read_csv("2023,8,22,5,4,35,640160_GLM.csv",header=None)

# dados_GLM.columns = ['number_of_groups', 'group_lat', 'group_lon', 'Year',
#        'Month', 'Day', 'Hour', 'Minute', 'Second', 'Second_micro', 'group_area',
#        'group_energy', 'group_parent_flash_id', 'group_quality_flag']



# ##########################################

# ######## Ploting...
# my_cmap = plt.get_cmap("jet")
# rescale = lambda s: (s - np.min(s)) / (np.max(s) - np.min(s))

# plt.figure(figsize=(25,15))

# # LMA
# plt.subplot(4,1,1)

# plt.scatter(s, filtered_df['alt']/1000, color=my_cmap(rescale(s)),s = 8)
# plt.ylabel("km",size=22)
# plt.xlim(tb-0.01, ta+0.01)
# #plt.ylim(ll,ul)
# plt.legend(["LMA Altitude"],loc='upper right', fontsize=20)

# plt.title(' LMA - Efield - GLM  2023,8,22,5,4,35,640160',size=25)


# # Efield
# plt.subplot(4,1,2)

# plt.plot(vector_time_total, Efield["Ch A"], color='blue')
# plt.ylabel("Electric Field (d.u)",size=18)
# plt.xlim(tb-0.01, ta+0.01)
# #plt.ylim(-5,4)

# #The number that multiply Efield["Ch B"] is empirical to help visualization
# plt.plot(vector_time_total, Efield["Ch B"]*7, color='red')
# plt.ylabel("Electric Field (d.u)",size=18)
# plt.xlim(tb-0.01, ta+0.01)
# #plt.ylim(-0.4,1)
# plt.legend(["Fast antenna","Slow antenna"],loc='lower right', fontsize=20)

# #GLM
# plt.subplot(4,1,3)

# plt.bar(dados_GLM["Second"], dados_GLM["group_energy"]/1e-15,width=0.002,align='center')
# #plt.xlabel("Time (s)",size=20)
# plt.ylabel("GLM \n group energy (fJ)",size=18)
# plt.xlim(tb-0.01, ta+0.01)
# plt.ylim(0,20)


# plt.subplot(4,1,4)

# plt.bar(dados_GLM["Second"], dados_GLM["group_energy"]/1e-15,width=0.002,align='center')
# #plt.xlabel("Time (s)",size=20)
# plt.ylabel("GLM \n group energy (fJ)",size=18)
# plt.xlim(tb, ta)
# plt.xlim(tb-0.01, ta+0.01)

# plt.savefig(f'{ID}.png', dpi = 200)
