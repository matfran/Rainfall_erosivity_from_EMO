# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:37:26 2021


@author: Francis Matthews
"""


import pandas as pd 
import geopandas as gpd
import numpy as np
import os 
from EMO_functions import ei30_from_ts, mask_snow, match_emo5_variables

main_dir = os.getcwd()
REDES_reference_path = 'R_factor_20150622_REF_Ens.csv'
REDES_ref = gpd.read_file(REDES_reference_path)

os.chdir(main_dir)

#read all input datasets
emo5_pd = pd.read_csv('REDES_station_timeseries_pd.csv')
emo5_pd.index = pd.to_datetime(emo5_pd['Date'], dayfirst = True)
emo5_rg =  pd.read_csv('REDES_station_timeseries_rg.csv')
emo5_rg.index = pd.to_datetime(emo5_pd['Date'], dayfirst = True)
emo5_tn = pd.read_csv('REDES_station_timeseries_tn.csv')
emo5_tn.index = pd.to_datetime(pd.to_datetime(emo5_tn['Date'], dayfirst = True).dt.date)
emo5_tx = pd.read_csv('REDES_station_timeseries_tx.csv')
emo5_tx.index = pd.to_datetime(pd.to_datetime(emo5_tx['Date'], dayfirst = True).dt.date)
emo5_pr = pd.read_csv('REDES_station_timeseries_pr6_emo1.csv')
emo5_pr.index = pd.to_datetime(pd.to_datetime(emo5_pr['Date'], dayfirst = True, format='%Y-%m-%d %H:%M:%S'))
emo5_pr = emo5_pr.drop(columns = ['Date_str'])
cols = emo5_pr.columns[1:]
#converted to mm/6hr instead of mm/day
emo5_pr[cols] = emo5_pr[cols].multiply(0.25)
    

alpha_p = 'alpha_params_v2.shp'
beta_p = 'beta_params_v2.shp'
alpha_m = gpd.read_file(alpha_p)
beta_m = gpd.read_file(beta_p)

#process EI30 at all stations and compile a dataframe
df_sim_compiled = pd.DataFrame()
count = 0
for i in np.arange(len(REDES_ref)):
    st = REDES_ref.iloc[i]
    EnS = st['EnS_name']
    station_name = 'Station_Id ' + str(st['Station_Id'])
    emo5_pr_st = pd.DataFrame(emo5_pr[station_name])
    emo5_ei_m = ei30_from_ts(emo5_pr_st, EnS, station_name, alpha_m, beta_m, time_resolution = 6)
    df_sim_compiled = df_sim_compiled.append(emo5_ei_m)
    count = count + 1
    
#reset index   
df_sim_compiled = df_sim_compiled.reset_index(drop = True)
#associate other EMO meterological variables
df_sim_compiled = match_emo5_variables(df_sim_compiled, emo5_pd, emo5_rg, emo5_tn, emo5_tx)
#mask snow (where temp is < 1 degree)
df_sim_compiled = mask_snow(df_sim_compiled, ei30_cols = ['RE EMO'])



