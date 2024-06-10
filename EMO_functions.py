# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:52:16 2023

@author: u0133999
"""
import pandas as pd 
import geopandas as gpd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from scipy import stats
import hydroeval as he

def ei30_from_ts(emo5_pr_st, EnS, col_name, alpha_m, beta_m, time_resolution = 6):
    """
    Process 6-hr rainfall erosivity into EI30 events by aggregating rainfall
    into events and then applying a monthly power-law equation to predict the rainfall EI30.
    For a full description of the method:
        https://www.sciencedirect.com/science/article/pii/S0341816222001436
    The function processes one station at a time.
    Parameters
    ----------
    emo5_pr_st : pd.DataFrame
        All precipitation timesteps for all station locations.
    EnS : String
        The EnS name of the station.
    col_name : String
        The station name (or column)
    alpha_m : gpd.DataFrame
        The shapefile of monthly alpha parameters
    beta_m : gpd.DataFrame
        The shapefile of monthly beta parameters
    time_resolution : Int, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    emo5_ei_m : pd.DataFrame
        A dataframe with the processed EI30 events per station in long format.

    """

    name = int(re.findall("\d+", col_name)[0])

    #define first the timesteps which potentially belong to an event
    emo5_pr_st['Timestamp'] = emo5_pr_st.index
    emo5_pr_st['potential event'] = (emo5_pr_st[col_name] > 1.27).astype(int)
    emo5_pr_sliced = emo5_pr_st[emo5_pr_st['potential event'] == 1].copy(deep = True)
    #when only potential event timesteps are sliced, get delta t
    emo5_pr_sliced['T-1'] = emo5_pr_sliced['Timestamp'].shift()
    emo5_pr_sliced['delta t (h)'] = (emo5_pr_sliced['Timestamp'] - emo5_pr_sliced['T-1']).astype('timedelta64[m]')/(60)
    #a new event is present if the time distance exceeds one timestep
    emo5_pr_sliced['New event'] = (emo5_pr_sliced['delta t (h)'] > time_resolution).astype(int)
    #index the events
    emo5_pr_sliced['Event_index'] = np.cumsum(emo5_pr_sliced['New event'])
    #group by the event index (sum)
    emo5_events = emo5_pr_sliced.groupby('Event_index', as_index = False).sum()
    
    #get a start and end timestamp. EMO5 accumulates precip and the timestamp represents the 
    #end of the accumulation period
    #minus the time resolution. Given that precipitation is accumulated from n hours to timestamp
    start_t = emo5_pr_sliced.groupby('Event_index').first()['Timestamp'] - pd.Timedelta(hours = time_resolution)
    end_t = emo5_pr_sliced.groupby('Event_index').last()['Timestamp']
    emo5_events['Start timestamp'] = start_t
    emo5_events['End timestamp'] = end_t    
    #get the event duration in hours
    emo5_events['Event dur (h)'] = (emo5_events['End timestamp'] - emo5_events['Start timestamp']).astype('timedelta64[m]')/(60)
    emo5_events.index = start_t
    
    #get 2 masks: 1) the total precip in an event > 12.7, or 2) the precip in one timestep > 6.35 mm
    mask_1 = (emo5_events[col_name] >= 12.7).values
    mask_2 = (emo5_events[col_name] >= 6.35) & (emo5_events[col_name] < 12.7) & (emo5_events['Event dur (h)'] == time_resolution)
    #combine the masks and slice the potential events
    emo5_ei = emo5_events[mask_1 | mask_2].copy(deep = True)
    
    #get a dataframe of the relevant alpha and beta parameters for the EnS
    a_b_ens = pd.DataFrame()
    
    columns = []
    for i in list(alpha_m.columns):
        if 'Month' in i:
            columns.append(i)
    
    a_b_ens['Month'] = np.arange(1,13)
    a_b_ens['Alpha'] = alpha_m[alpha_m['EnS_name'] == EnS][columns].T.values
    a_b_ens['Beta'] = beta_m[alpha_m['EnS_name'] == EnS][columns].T.values
    
    #prioritise backfill and fill remaining months in a forward 
    a_b_ens = a_b_ens.fillna(method = 'bfill')
    a_b_ens = a_b_ens.fillna(method = 'ffill')
    

    #get the month of the event
    emo5_ei['Month'] = emo5_ei['Start timestamp'].dt.month
    #merge based on the month
    emo5_ei_m = emo5_ei.merge(a_b_ens, how = 'left', on = 'Month')
    emo5_ei_m['RE EMO'] =  emo5_ei_m['Alpha'] * emo5_ei_m[col_name] ** emo5_ei_m['Beta']
    emo5_ei_m['Station_Id'] = name
    emo5_ei_m['EnS_name'] = EnS
    emo5_ei_m['EnZ'] = emo5_ei_m['EnS_name'].str[:3]
    emo5_ei_m = emo5_ei_m.rename(columns = {col_name: 'Rainfall depth (mm)'})
    emo5_ei_m = emo5_ei_m[['Station_Id', 'EnS_name', 'EnZ', 'Start timestamp', 'End timestamp', 
                           'Event dur (h)','Rainfall depth (mm)', 'RE EMO', 'Alpha',
                           'Beta']]
    return emo5_ei_m
    
    
    

def match_emo5_variables(df_sim_compiled, emo5_pd, emo5_rg, emo5_tn, emo5_tx, 
                         merge_type = 'nearest', time_tolerance = 24, dataset = 'EMO'):
    """
    A function to match EI30 events with the other meterological variables 
    from EMO.

    Parameters
    ----------
    df_sim_compiled : pd.DataFrame
        DESCRIPTION.
    emo5_pd : pd.DataFrame
        DESCRIPTION.
    emo5_rg : pd.DataFrame
        DESCRIPTION.
    emo5_tn : pd.DataFrame
        DESCRIPTION.
    emo5_tx : pd.DataFrame
        DESCRIPTION.
    merge_type : String, optional
        The direction argument of pd.merge_asof. The default is 'nearest'.
    time_tolerance : TYPE, optional
        DESCRIPTION. The default is 24.
    dataset : TYPE, optional
        DESCRIPTION. The default is 'EMO'.

    Returns
    -------
    df_all : TYPE
        DESCRIPTION.

    """
    
    df_all = pd.DataFrame()
    skipped_gauges = []
    for st_id in df_sim_compiled['Station_Id'].unique():
        
        df_ss = df_sim_compiled[df_sim_compiled['Station_Id'] == st_id]
        col = 'Station_Id ' + str(st_id)
        try:
            all_emo5_st = pd.DataFrame({'Start timestamp': pd.to_datetime(emo5_pd['Date'], dayfirst = True), 'pd_EMO5': emo5_pd[col], 
                                        'rg_EMO5': emo5_rg[col], 'tn_EMO5': emo5_tn[col], 
                                        'tx_EMO5': emo5_tx[col]})
            
        except:
            skipped_gauges.append([st_id, 'no emo5 ts'])
        
        all_emo5_st = all_emo5_st[~ all_emo5_st['Start timestamp'].isnull()]
        
        #merge with EMO5 timeseries variables
        df_ss = pd.merge_asof(df_ss, all_emo5_st, on = 'Start timestamp',
                                       tolerance=pd.Timedelta(hours = time_tolerance), 
                                       direction = merge_type)
        
        df_all = df_all.append(df_ss)
    if dataset == 'EMO':    
        df_all = df_all.reset_index(drop = True)
    elif dataset == 'REDES':
        df_all.index = pd.to_datetime(pd.to_datetime(df_all['Start timestamp'], yearfirst = True))
        df_all.index.name = 'Date'
    return df_all


def mask_snow(df, ei30_cols):
    """
    A function to mask the rainfall erosivity events which were likely snowfall.

    Parameters
    ----------
    df : pd.DataFrame
        A long form dataframe of rainfall erosivity events.
    ei30_cols : List
        The column names to mask

    Returns
    -------
    df : pd.DataFrame
        A dataframe with masked events

    """
    
    for col in ei30_cols:
        df[col] = np.where(df['tx_EMO5'] <= 1, np.nan, df[col]) 
        
    return df



    
