#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:55:59 2026

@author: amounier
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.dates import DateFormatter, MonthLocator

from utils import France, Climat, City, get_warming_color
from temperature import get_temperature_data


def detect_heatwaves_TRACC(temperature, S1 , S2, S3):
    temperature = temperature.sort_index()
    daily_temperature = temperature.resample("D").agg(["max", "min"]).dropna()
    daily_temperature['mean'] = ((daily_temperature["max"] + daily_temperature["min"]) / 2)
    
    heatwaves = []
    
    doy = 0
    while doy < len(daily_temperature):
        if daily_temperature['mean'].values[doy] <= S1:
            doy += 1
            continue

        start = doy
        days_below_s2 = 0
        j = doy + 1
        end = None
    
        while True:
            if j >= len(daily_temperature):
                end = len(daily_temperature) - 1
                break
            if daily_temperature['mean'].values[j] < S3:
                end = j - 1
                break
            if daily_temperature['mean'].values[j] < S2:
                days_below_s2 += 1
                if days_below_s2 > 2:
                    end = j - 1 
                    break
            else:
                days_below_s2 = 0
            j += 1
    
        heatwaves.append({
            "start": daily_temperature.index[start],
            "end": daily_temperature.index[end],
            "duration_days": end - start + 1,
            "max_temperature": daily_temperature['max'].values[start:end + 1].max(),
            "severity": (daily_temperature['mean']-S2).values[start:end + 1].sum()/(S1-S2),
        })
        doy = end + 1
    
    return pd.DataFrame(heatwaves, columns=["start", "end", "duration_days", "max_temperature", "severity"])


def detect_heatwaves_MF(temperature, Smin, Smax):
    temperature = temperature.sort_index()
    daily_temperature = temperature.resample("D").agg(["max", "min"]).dropna()
    daily_temperature['mean'] = ((daily_temperature["max"] + daily_temperature["min"]) / 2)
    
    heatwaves = []
    
    doy = 0
    while doy < len(daily_temperature):
        if daily_temperature['max'].values[doy] < Smax or daily_temperature['min'].values[doy] < Smin:
            doy += 1
            continue

        start = doy
        days_below_smax = 0
        days_below_smin = 0
        j = doy + 1
        end = None
    
        while True:
            if j >= len(daily_temperature):
                end = len(daily_temperature) - 1
                break
            if daily_temperature['max'].values[j] < Smax:
                days_below_smax += 1
                if days_below_smax > 3:
                    end = j - 1 
                    break
            if daily_temperature['min'].values[j] < Smin:
                days_below_smin += 1
                if days_below_smin > 3:
                    end = j - 1 
                    break
            else:
                days_below_smax = 0
                days_below_smin = 0
            j += 1
    
        heatwaves.append({
            "start": daily_temperature.index[start],
            "end": daily_temperature.index[end],
            "duration_days": end - start + 1,
            "max_temperature": daily_temperature['max'].values[start:end + 1].max(),
            "severity": (daily_temperature['mean']-Smin).values[start:end + 1].sum()/(Smax-Smin),
        })
        doy = end + 1
    
    return pd.DataFrame(heatwaves, columns=["start", "end", "duration_days", "max_temperature", "severity"])


def detect_heatwaves_humidex(data, Smin, Smax, threshold=30):
    # https://en.wikipedia.org/wiki/Humidex
    data = data.sort_index()
    T = data[data.columns[0]]
    R = data[data.columns[1]]
    data['humidex'] = T + 0.5555 * (6.11*np.exp(5417.7530*(1/273.16 - 1/(R+273.15))) - 10)
    data['heatwave'] = data.humidex>threshold
    data = data.dropna()
    
    daily_temperature = T.resample("D").agg(["max", "min"]).dropna()
    daily_temperature['mean'] = ((daily_temperature["max"] + daily_temperature["min"]) / 2)
    daily_data = data['heatwave'].resample("D").agg(["any"]).dropna()
    
    heatwaves = []
    
    idx = 0
    while idx < len(daily_data):
        if not daily_data['any'].values[idx]:
            idx += 1
            continue
        
        start = idx
        j = idx + 1
        flag = True
        while j < len(daily_data) and flag:
            if not daily_data['any'].values[j]:
                end = j - 1 
                flag = False
            else:
                j += 1
            
        idx = end +1
    
        heatwaves.append({
            "start": daily_data.index[start],
            "end": daily_data.index[end],
            "duration_days": end - start + 1,
            "max_temperature": daily_temperature['max'].values[start:end + 1].max(),
            "severity": (daily_temperature['mean']-Smin).values[start:end + 1].sum()/(Smax-Smin),
        })
    
    res = pd.DataFrame(heatwaves, columns=["start", "end", "duration_days", "max_temperature", "severity"])
    res = res[res.duration_days>1]
    res = res[res.severity>1]
        
    return res


def detect_heatwaves_heatindex(data, threshold=80):
    # https://en.wikipedia.org/wiki/Heat_index
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6
    
    data = data.sort_index()
    
    T = 9/5 * data.temperature + 32
    R = data.humidite
    
    data['heatindex'] = c1 + c2*T + c3*R + c4*T*R + c5*T**2 + c6*R**2 + c7*T**2*R + c8*T*R**2 + c9*T**2*R**2
    data['heatwave'] = data.heatindex>threshold
    data = data.dropna()
    
    daily_temperature = data['temperature'].resample("D").agg(["max", "min"]).dropna()
    daily_data = data['heatwave'].resample("D").agg(["any"]).dropna()
    
    heatwaves = []
    
    idx = 0
    while idx < len(daily_data):
        if not daily_data['any'].values[idx]:
            idx += 1
            continue
        
        start = idx
        j = idx + 1
        flag = True
        while j < len(daily_data) and flag:
            if not daily_data['any'].values[j]:
                end = j - 1 
                flag = False
            else:
                j += 1
            
        idx = end +1
    
        heatwaves.append({
            "start": daily_data.index[start],
            "end": daily_data.index[end],
            "duration_days": end - start + 1,
            "max_temperature": daily_temperature['max'].values[start:end + 1].max(),
            "severity": np.nan,
        })
    
    res = pd.DataFrame(heatwaves, columns=["start", "end", "duration_days", "max_temperature", "severity"])
    res = res[res.duration_days>1]
        
    return res




#%% ===========================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()
    
    import pandas as pd


    #%% identification graphique
    if False:
        data = get_temperature_data(force=False)
        # data.to_csv(os.path.join('output','temperature.csv'))
        
        for zcl in France().climats:
            fig,axs = plt.subplots(4,1,figsize=(10,12),dpi=300,sharex=True,sharey=True)
            axs = axs.flatten()
            
            min_ylim, max_ylim = 10,0
            for idx,warming in enumerate(['obs',2.0,2.7,4.0]):
                ax = axs[idx]
                color = get_warming_color(warming)
                
                temperature_variable = 'temperature'
                rosee_variable = 'rosee'
                if warming != 'obs':
                    temperature_variable += f'_{warming}'
                    rosee_variable += f'_{warming}' 
                    label = f'TRACC (+{warming}°C)'
                else:
                    label = 'Observations 2026'
                    
                data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
                
                # test = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds)
                test = detect_heatwaves_MF(data_zcl[temperature_variable], Smin=City(Climat(zcl).center_city).departement.Smin, Smax=City(Climat(zcl).center_city).departement.Smax)
                # test = detect_heatwaves_humidex(data_zcl[[temperature_variable,rosee_variable]], Smin=City(Climat(zcl).center_city).departement.Smin, Smax=City(Climat(zcl).center_city).departement.Smax)
                # test = detect_heatwaves_heatindex(data_zcl[['temperature','humidite']])
                
                ax.plot(data_zcl.index,data_zcl[temperature_variable],color=color,lw=1,label=label)
                
                # ylim = ax.get_ylim()
                min_ylim = min(data_zcl[temperature_variable].min()-2,min_ylim)
                max_ylim = max(data_zcl[temperature_variable].max()+2,max_ylim)
                
                for start,end in zip(test.start,test.end):
                    ax.fill_between([start,end],[60]*2,[-10]*2,color=color,alpha=0.2)
                
                print(temperature_variable)
                print(test)
                print()
                
                temperature_variable = 'temperature'
                rosee_variable = 'rosee'
                if warming != 'obs':
                    temperature_variable += f'_{warming}_heatwave'
                    rosee_variable += f'_{warming}_heatwave' 
                    
                    data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
                    
                    # test = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds)
                    test = detect_heatwaves_MF(data_zcl[temperature_variable], Smin=City(Climat(zcl).center_city).departement.Smin, Smax=City(Climat(zcl).center_city).departement.Smax)
                    # test = detect_heatwaves_humidex(data_zcl[[temperature_variable,rosee_variable]], Smin=City(Climat(zcl).center_city).departement.Smin, Smax=City(Climat(zcl).center_city).departement.Smax)
                    # test = detect_heatwaves_heatindex(data_zcl[['temperature','humidite']])
                    
                    ax.plot(data_zcl.index,data_zcl[temperature_variable],color='k',lw=0.5,zorder=0, label='5% heatwave')
                    
                    # ylim = ax.get_ylim()
                    min_ylim = min(data_zcl[temperature_variable].min()*0.95,min_ylim)
                    max_ylim = max(data_zcl[temperature_variable].max()*1.05,max_ylim)
                    
                    for start,end in zip(test.start,test.end):
                        ax.fill_between([start,end],[60]*2,[-10]*2,color='k',alpha=0.2,zorder=-1)
                        
                    print(temperature_variable)
                    print(test)
                    print()
                
                ax.legend()
                
                if idx == 0:
                    ax.set_title(f'{zcl} ({Climat(zcl).center_city})')
                    ax.set_ylabel('Hourly temperature (°C)')
                    ax.xaxis.set_major_locator(MonthLocator())
                    ax.xaxis.set_major_formatter(DateFormatter("%b"))
                    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
                    
                if idx == 3:
                    ax.set_ylim([min_ylim, max_ylim])
                
            plt.subplots_adjust(hspace=0.05)
            plt.savefig(os.path.join('output',f'heatwaves_{zcl}_MF.png'), bbox_inches='tight')
            plt.show()
            
    #%% Graphes de 2026 par rapport au niveaux de réchauffement
    if True:
        data = get_temperature_data(force=False)
        france = France()
        
        df = {'zcl':[],'warming':[],
              'duration_day_mean':[],'duration_day_std':[],
              'max_temperature_max':[],'max_temperature_mean':[],'max_temperature_std':[],
              'severity_mean':[],'severity_std':[],
              'number_events':[]}
        
        for zcl in tqdm.tqdm(france.climats):
            data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
            
            for warming in ['obs','2.0','2.0_heatwave','2.7','2.7_heatwave','4.0','4.0_heatwave']:
                temperature_variable = f'temperature_{warming}'.replace('_obs','')
                
                # test = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds)
                test = detect_heatwaves_MF(data_zcl[temperature_variable], Smin=City(Climat(zcl).center_city).departement.Smin, Smax=City(Climat(zcl).center_city).departement.Smax)
                
                if test.empty:
                    duration_day_mean = np.nan
                    duration_day_std = np.nan
                    max_temperature_mean = np.nan
                    max_temperature_std = np.nan
                    severity_mean = np.nan
                    severity_std = np.nan
                    number_events = 0 
                else:
                    test_mean = test.mean()
                    test_std = test.std()
                    test_max = test.max()
                    duration_day_mean = test_mean.loc['duration_days']
                    duration_day_std = test_std.loc['duration_days']
                    max_temperature_max = test_max.loc['max_temperature']
                    max_temperature_mean = test_mean.loc['max_temperature']
                    max_temperature_std = test_std.loc['max_temperature']
                    severity_mean = test_mean.loc['severity']
                    severity_std = test_std.loc['severity']
                    number_events = len(test)
                
                df["zcl"].append(zcl)
                df['warming'].append(warming)
                df['duration_day_mean'].append(duration_day_mean)
                df['duration_day_std'].append(duration_day_std)
                df['max_temperature_mean'].append(max_temperature_mean)
                df['max_temperature_max'].append(max_temperature_max)
                df['max_temperature_std'].append(max_temperature_std)
                df['severity_mean'].append(severity_mean)
                df['severity_std'].append(severity_std)
                df['number_events'].append(number_events)
                
        df = pd.DataFrame().from_dict(df)
        df.to_csv(os.path.join('output','heatwaves_MF.csv'),index=False)
        
        df = pd.read_csv(os.path.join('output','heatwaves_MF.csv'))
        print(df)
        
        fig,axs = plt.subplots(1,3,figsize=(16,5),dpi=300,sharex=True)
        axs = axs.flatten()
        
        ylabel_dict = {'duration_day_mean':'Mean heatwave duration (days)',
                       'max_temperature_max':'Max heatwave temperature (°C)',
                       'max_temperature_mean':'Mean of max heatwave temperature (°C)',
                       'severity_mean':'Mean heatwave severity (no unit)'}
        
        for idx_var, var in enumerate(['duration_day_mean','max_temperature_mean','severity_mean']):
            if var == 'max_temperature_max':
                bottom = 30
                top = 55
            if var == 'max_temperature_mean':
                bottom = 30
                top = 50
            elif var == 'duration_day_mean':
                bottom = 0
                top = 25
            elif 'severity_mean':
                bottom = 0
                top = 15
                
            ax = axs[idx_var]
            for idx,zcl in enumerate(france.climats):
                if idx == 0:
                    label_obs = '2026'
                else:
                    label_obs = ''
                # ax.errorbar(x=[idx],
                #             y=[df[(df.warming=='obs')&(df.zcl==zcl)].duration_day_mean.values[0]],
                #             # yerr=[df[(df.warming=='obs')&(df.zcl==zcl)].duration_day_std.values[0]],
                #             yerr=[0],
                #             marker='o',color=get_warming_color('obs'),capsize=3,mec='w',ls='',
                #             label=label_obs)
                ax.plot([idx-0.35,idx,idx+0.35],
                        [df[(df.warming=='obs')&(df.zcl==zcl)][var].values[0]]*3,lw=2,
                        color=get_warming_color('obs'),label=label_obs,zorder=5)
                for idx_warming,warming in enumerate(['2.0','2.7','4.0']):
                    if idx == 0:
                        label_warming = f'TRACC +{warming}°C'
                    else:
                        label_warming = ''
                    # ax.errorbar(x=[idx+(-1+idx_warming)*0.25],
                    #             y=[df[(df.warming==warming)&(df.zcl==zcl)].duration_day_mean.values[0]],
                    #             # yerr=[df[(df.warming==warming)&(df.zcl==zcl)].duration_day_std.values[0]],
                    #             yerr=[0],
                    #             marker='o',color=get_warming_color(float(warming)),capsize=3,mec='w',ls='',
                    #             label=label_warming)
                    ax.bar(x=[idx+(-1+idx_warming)*0.25],
                           height=[df[(df.warming==warming)&(df.zcl==zcl)][var].values[0]],
                           width=0.2,
                           color=get_warming_color(float(warming)),
                           label=label_warming)
                    # ax.errorbar(x=[idx+(-1+idx_warming)*0.25],
                    #             y=[df[(df.warming==f'{warming}_heatwave')&(df.zcl==zcl)].duration_day_mean.values[0]],
                    #             # yerr=[df[(df.warming==warming)&(df.zcl==zcl)].duration_day_std.values[0]],
                    #             yerr=[0],
                    #             marker='^',color=get_warming_color(float(warming)),capsize=3,mec='w',ls='',)
                    ax.plot([idx+(-1+idx_warming)*0.25],
                            [df[(df.warming==f'{warming}_heatwave')&(df.zcl==zcl)][var].values[0]],
                            marker='^',color=get_warming_color(float(warming)),mec='w',ls='',ms=8)
                    
                    if idx%2 == 1:
                        ax.fill_between([idx-0.5,idx+0.5],[top]*2,[bottom]*2,color='tab:grey',alpha=0.1,zorder=-2)
                    
            ax.plot([-1],[0],color='k',marker='^',label='5% heatwave',mec='w',ls='',ms=8)
            ax.set_xlim([-0.5,len(france.climats)-0.5])
            ax.set_ylabel(ylabel_dict.get(var))
            ax.set_xticks(list(range(len(france.climats))),france.climats)
            ax.set_ylim(bottom=bottom, top=top)
            ax.yaxis.get_major_locator().set_params(integer=True)
            
            if idx_var == 2:
                ax.legend()
        
        plt.subplots_adjust(wspace=0.15)
        plt.savefig(os.path.join('output','heatwave_comparison.png'), bbox_inches='tight')
        plt.show()
                
    
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()