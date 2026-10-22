#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:55:23 2026

@author: amounier
"""

import time
import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from matplotlib.dates import DateFormatter, MonthLocator
import numpy as np

from utils import France, Climat, City, get_warming_color

def open_TRACC_file(force=False):
    full_data = None
    full_data_name = 'tracc_data.csv'
    
    if full_data_name not in os.listdir(os.path.join('data','TRACC')) or force:
        for warming_level in tqdm.tqdm([2.,2.7,4.]):
            for heatwave in [False,True]:
                file_name = f"Données climatiques prospectives - France +{str(warming_level).replace('.','-').replace('-0','')}.csv"
                if heatwave:
                    file_name = file_name.replace('.csv',' - Vague de chaleur.csv')
                    
                data = pd.read_csv(os.path.join('data','TRACC',file_name),sep=';')
                data = data[['STATION','MOIS','JOUR','HEURE','T2m','Hur','Tr']]
                data = data.rename(columns={'MOIS':'month','JOUR':'day','HEURE':'hour'})
                data['year'] = [2026]*len(data)
                data['datetime'] = pd.to_datetime(data[['year','month','day','hour']])
                data = data[['datetime','STATION','T2m','Hur','Tr']].set_index(['datetime','STATION'])
                
                if heatwave:
                    heatwave_suffix = '_heatwave'
                else:
                    heatwave_suffix = ''
                data = data.rename(columns={'T2m':f'temperature_{warming_level}'+heatwave_suffix,
                                            'Hur':f'humidite_{warming_level}'+heatwave_suffix,
                                            'Tr' :f'rosee_{warming_level}'+heatwave_suffix})
                
                if full_data is None:
                    full_data = data
                else:
                    full_data = full_data.join(data)
                    
        full_data.to_csv(os.path.join('data','TRACC',full_data_name))
        
    full_data = pd.read_csv(os.path.join('data','TRACC',full_data_name))
    full_data['datetime'] = pd.to_datetime(full_data.datetime)
    full_data = full_data.set_index(['datetime','STATION'])
    return full_data


def open_MF_data(force=False):
    full_data = None
    full_data_name = 'observation_data.csv'
    
    station_dict = {'TRAPPES':'VILLACOUBLAY',
                    'NANCY':'NANCY-ESSEY',
                    'MACON':'MACON',
                    'RENNES':'RENNES GALLET',
                    'TOURS':'TOURS',
                    'AGEN':'AGEN-LA GARENNE',
                    'CARPENTRAS':'CARPENTRAS',
                    'MARIGNANE':'MARIGNANE'}
    
    if full_data_name not in os.listdir(os.path.join('data','MF')) or force:
        france = France()
        zcl_list = [Climat(e) for e in france.climats]
        
        for zcl in tqdm.tqdm(zcl_list):
            file_name = f'HOR_departement_{City(zcl.center_city).departement.code}_periode_2025-2026.parquet'
            
            data = pd.read_parquet(os.path.join('data','MF',file_name))
            data = data[data.NOM_USUEL==station_dict.get(zcl.center_city_caps)][['AAAAMMJJHH','T','TD','U']]
            data['STATION'] = [zcl.center_city_caps]*len(data)
            data['datetime'] = pd.to_datetime(data.AAAAMMJJHH,format='%Y%m%d%H')
            data = data[data.datetime.dt.year==2026]
            data = data[['datetime','STATION','T','U','TD']].set_index(['datetime','STATION'])
            data = data.rename(columns={'T':'temperature','U':'humidite','TD' :'rosee'})
            
            if full_data is None:
                full_data = data
            else:
                full_data = pd.concat([full_data,data])
                
        full_data.to_csv(os.path.join('data','MF',full_data_name))
    
    full_data = pd.read_csv(os.path.join('data','MF',full_data_name))
    full_data['datetime'] = pd.to_datetime(full_data.datetime)
    full_data = full_data.sort_values(by='datetime')
    full_data = full_data.set_index(['datetime','STATION'])
    return full_data
    

def comparison_stations(station_1='TRAPPES', station_2='TOUSSUS LE NOBLE'):
    zcl = Climat('H1a')
    file_name = f'HOR_departement_{City(zcl.center_city).departement.code}_periode_2025-2026.parquet'
    
    data = pd.read_parquet(os.path.join('data','MF',file_name))
    
    data_station_1 = data[data.NOM_USUEL==station_1]
    num_poste_station_1 = data_station_1.NUM_POSTE.values[0]
    data_station_1 = data_station_1[['AAAAMMJJHH','T']].rename(columns={'T':f'temperature_{station_1}'})
    data_station_1['datetime'] = pd.to_datetime(data_station_1.AAAAMMJJHH,format='%Y%m%d%H')
    data_station_1 = data_station_1[data_station_1.datetime.dt.year==2026]
    data_station_1 = data_station_1.sort_values(by='datetime')
    data_station_1 = data_station_1[['datetime',f'temperature_{station_1}']].set_index('datetime')
    
    if not data_station_1[data_station_1[f'temperature_{station_1}'].isna()].empty:
        begin = data_station_1[data_station_1[f'temperature_{station_1}'].isna()].index[0]
        end = data_station_1[data_station_1[f'temperature_{station_1}'].isna()].index[-1]
        print(f"Missing value for {station_1}: {begin.strftime('%d %B')} - {end.strftime('%d %B')}")
        
    data_station_2 = data[data.NOM_USUEL==station_2]
    num_poste_station_2 = data_station_2.NUM_POSTE.values[0]
    data_station_2 = data_station_2[['AAAAMMJJHH','T']].rename(columns={'T':f'temperature_{station_2}'})
    data_station_2['datetime'] = pd.to_datetime(data_station_2.AAAAMMJJHH,format='%Y%m%d%H')
    data_station_2 = data_station_2[data_station_2.datetime.dt.year==2026]
    data_station_2 = data_station_2.sort_values(by='datetime')
    data_station_2 = data_station_2[['datetime',f'temperature_{station_2}']].set_index('datetime')
    
    if not data_station_2[data_station_2[f'temperature_{station_2}'].isna()].empty:
        begin = data_station_2[data_station_2[f'temperature_{station_2}'].isna()].index[0]
        end = data_station_2[data_station_2[f'temperature_{station_2}'].isna()].index[-1]
        print(f"Missing value for {station_2}: {begin.strftime('%d %B')} - {end.strftime('%d %B')}")
        
    data = data_station_1.join(data_station_2,how='outer')
    
    data_no_nan = data.dropna()
    mae = metrics.mean_absolute_error(data_no_nan[f'temperature_{station_1}'],data_no_nan[f'temperature_{station_2}'])
    mse = metrics.mean_squared_error(data_no_nan[f'temperature_{station_1}'],data_no_nan[f'temperature_{station_2}'])
    rmse = np.sqrt(mse)
    
    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
    ax.plot([-10,45],[-10,45],color='k',)
    ax.plot(data[f'temperature_{station_1}'],data[f'temperature_{station_2}'],marker='.',ls='',alpha=0.17,
            label=f'RMSE = {rmse:.1f}°C\nMAE = {mae:.1f}°C')
    ax.legend()
    ax.set_ylim([-10,45])
    ax.set_xlim([-10,45])
    ax.set_xlabel(f'{station_1} ({num_poste_station_1})')
    ax.set_ylabel(f'{station_2} ({num_poste_station_2})')
    plt.savefig(os.path.join('output',f'{station_1.replace(' ','_')}_{station_2.replace(' ','_')}'), bbox_inches='tight')
    plt.show()
    
    return rmse, mae


def get_temperature_data(force=False):
    TRACC_data = open_TRACC_file(force=force)
    MF_data = open_MF_data(force=force)
    data = TRACC_data.join(MF_data)
    return data
    

#%% ===========================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()
    
    #%% Trou de données pour Trappes, remplacement par Villacoublay
    if False:
        station_1 = 'TRAPPES'
        stations_2 = ['ACHERES',
                      'LE PECQ',
                      'LE PERRAY',
                      'MAGNANVILLE',
                      'MAULE',
                      'PRUNAY-LE-TEMPLE_SAPC',
                      'ST-LEGER_SAPC',
                      'TOUSSUS LE NOBLE',
                      'VERSAILLES- INRAE',
                      'VILLACOUBLAY']
        for station_2 in stations_2:
            print(station_2)
            rmse, mae = comparison_stations(station_1,station_2)
            print(f'\tRMSE = {rmse:.1f}°C\n\t MAE = {mae:.1f}°C\n')
        
        
    #%% Graphes des temperatures annuelles TRACC (inutile finalement)
    if False:
        france = France()
        zcl_list = [Climat(c) for c in france.climats]
        
        TRACC_data = open_TRACC_file(force=False)
        MF_data = open_MF_data(force=False)
        
        warming_level = 2.
        
        for zcl in tqdm.tqdm(zcl_list):
            fig,axs = plt.subplots(3,1,figsize=(10,15),dpi=300,sharex=True,sharey=True)
            axs = axs.flatten()
            
            for idx,warming_level in enumerate([2.,2.7,4.][::-1]):
                ax = axs[idx]
                city = zcl.center_city_caps
                data = TRACC_data[TRACC_data.index.get_level_values('STATION')==city].reset_index().set_index('datetime')
                obs = MF_data[MF_data.index.get_level_values('STATION')==city].reset_index().set_index('datetime')
                
                ax.plot(data.index,
                        data[f'temperature_{warming_level}'],
                        label=f'TRACC (+{warming_level}°C)',color=get_warming_color(warming_level),lw=1)
                ax.fill_between(data.index,
                                data[f'temperature_{warming_level}_heatwave'],
                                data[f'temperature_{warming_level}'],
                                label='5% heatwave',color=get_warming_color(warming_level),alpha=0.37)
                ax.plot(obs.index,
                        obs['temperature'],
                        label='Observations',color='k',lw=0.67)
                
                
                ax.legend()
                
                if idx == 0:
                    ax.set_title(f'{zcl.code} ({zcl.center_city})')
                    ax.set_ylabel('Hourly temperature (°C)')
                    
                    ax.xaxis.set_major_locator(MonthLocator())
                    ax.xaxis.set_major_formatter(DateFormatter("%b"))
                    
                    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
            
            plt.subplots_adjust(hspace=0.05)
            plt.savefig(os.path.join('output',f'temperature_{zcl.code}.png'), bbox_inches='tight')
            plt.show()
            plt.close()


    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()