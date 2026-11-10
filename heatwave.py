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

from utils import France, Climat, get_warming_color
from temperature import get_temperature_data


def detect_heatwaves_TRACC(temperature,S1,S2,S3):
    temperature = temperature.sort_index()
    daily_temperature = temperature.resample("D").agg(["max","min"]).dropna()
    daily_temperature['mean'] = ((daily_temperature["max"] + daily_temperature["min"]) / 2)
    
    start_end_heatwaves = []
    
    heatwave_flag = False
    for doy in range(len(daily_temperature)):
        if not heatwave_flag and daily_temperature['mean'].values[doy] < S2:
            continue
        
        if not heatwave_flag and daily_temperature['mean'].values[doy] >= S2:
            heatwave_flag = True
            nb_days_below_S2 = 0
            start = doy
            
        if heatwave_flag and daily_temperature['mean'].values[doy] >= S2:
            nb_days_below_S2 = 0
        
        if heatwave_flag and daily_temperature['mean'].values[doy] < S3:
            heatwave_flag = False
            end = doy - 1 - nb_days_below_S2
            start_end_heatwaves.append((start,end))
        
        if heatwave_flag and daily_temperature['mean'].values[doy] < S2:
            nb_days_below_S2 += 1
            if nb_days_below_S2 == 3:
                heatwave_flag = False
                end = doy - 3
                start_end_heatwaves.append((start,end))
    
    heatwaves = []
    for start,end in start_end_heatwaves:
        heatwaves.append({"start": daily_temperature.index[start],
                          "end": daily_temperature.index[end],
                          "duration_days": end - start + 1,
                          "max_mean_temperature": daily_temperature['mean'].values[start:end + 1].max(),
                          "max_temperature": daily_temperature['max'].values[start:end + 1].max(),
                          "severity": (daily_temperature['mean']-S2).values[start:end + 1].sum()/ (S1 - S2)})
        
    heatwaves = pd.DataFrame(heatwaves, columns=["start", "end", "duration_days","max_mean_temperature","max_temperature", "severity"])
    heatwaves = heatwaves[heatwaves.max_mean_temperature>=S1]
    heatwaves = heatwaves.reset_index()[["start", "end", "duration_days","max_temperature", 'max_mean_temperature',"severity"]]
    return heatwaves



#%% ===========================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()

    #%% identification graphique
    if True:
        data = get_temperature_data(force=False)
        data.to_csv(os.path.join('output','temperature.csv'))
        
        for zcl in France().climats:
            fig,axs = plt.subplots(5,1,figsize=(10,13),dpi=300,sharex=True,sharey=True)
            axs = axs.flatten()
            
            for idx,warming in enumerate([2003,2026,2.0,2.7,4.0]):
                ax = axs[idx]
                color = get_warming_color(warming)
                
                temperature_variable = f'temperature_{warming}'
                rosee_variable = f'rosee_{warming}'
                if warming in [2003,2026]:
                    label = f'Observations {warming}'
                else:
                    label = f'TRACC (+{warming}°C)'
                    
                data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
                
                min_ylim = data_zcl[[c for c in data_zcl.columns if c.startswith('temperature')]].min().min()
                max_ylim = data_zcl[[c for c in data_zcl.columns if c.startswith('temperature')]].max().max()
                
                heatwaves = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds_SAFRAN)
                
                ax.plot(data_zcl.index,data_zcl[temperature_variable],color=color,lw=1,label=label)
                
                for start,end in zip(heatwaves.start,heatwaves.end):
                    ax.fill_between([start,end],[max_ylim+2]*2,[min_ylim-2]*2,color=color,alpha=0.2)
                
                print(zcl, temperature_variable)
                print(heatwaves)
                print()
                
                temperature_variable = 'temperature'
                rosee_variable = 'rosee'
                if warming not in [2003,2026]:
                    temperature_variable += f'_{warming}_heatwave'
                    rosee_variable += f'_{warming}_heatwave' 
                    
                    data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
                    
                    heatwaves = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds_SAFRAN)
                    
                    ax.plot(data_zcl.index,data_zcl[temperature_variable],color='k',lw=0.5,zorder=0, label='5% heatwave')
                    
                    for start,end in zip(heatwaves.start,heatwaves.end):
                        ax.fill_between([start,end],[max_ylim+2]*2,[min_ylim-2]*2,color='k',alpha=0.2,zorder=-1)
                    
                    print(zcl,temperature_variable)
                    print(heatwaves)
                    print()
                
                ax.legend()
                
                if idx == 0:
                    ax.set_title(f'{zcl} ({Climat(zcl).center_city})')
                    ax.set_ylabel('Hourly temperature (°C)')
                    ax.xaxis.set_major_locator(MonthLocator())
                    ax.xaxis.set_major_formatter(DateFormatter("%b"))
                    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
                    
                if idx == 4:
                    ax.set_ylim([min_ylim, max_ylim])
                
            plt.subplots_adjust(hspace=0.05)
            plt.savefig(os.path.join('output',f'heatwaves_{zcl}.png'), bbox_inches='tight')
            plt.show()
            
    #%% Graphes de 2026 par rapport au niveaux de réchauffement
    if True:
        data = get_temperature_data(force=False)
        france = France()
        
        ref_heatwave_values = pd.read_csv(os.path.join('data','TRACC','caracteristiques_hw_stations_re2020.txt'))
        ref_heatwave_values['warming'] = [float(e.split('_')[1].replace('fr','')) for e in ref_heatwave_values['Unnamed: 0']]
        ref_heatwave_values['STATION'] = [e.split('_')[0] for e in ref_heatwave_values['Unnamed: 0']]
        ref_heatwave_values = ref_heatwave_values[['STATION','warming','HW_dur','T2m_max','Hw_Sev']].set_index(['STATION','warming'])
        print(ref_heatwave_values)
        
        df = {'zcl':[],'warming':[],
              'duration_day_mean':[],'duration_day_std':[],'duration_day_sum':[],'duration_day_max':[],
              'max_temperature_max':[],'max_temperature_mean':[],'max_temperature_std':[],
              'severity_mean':[],'severity_std':[],'severity_max':[],'severity_sum':[],
              'number_events':[]}
        
        for zcl in tqdm.tqdm(france.climats):
            data_zcl = data[data.index.get_level_values('STATION')==Climat(zcl).center_city_caps].reset_index().set_index('datetime')
            
            for warming in [2003,2026,'2.0','2.0_heatwave','2.7','2.7_heatwave','4.0','4.0_heatwave']:
                temperature_variable = f'temperature_{warming}'
                
                heatwaves = detect_heatwaves_TRACC(data_zcl[temperature_variable], *Climat(zcl).heatwave_thresholds_SAFRAN)
                
                if heatwaves.empty:
                    duration_day_mean = np.nan
                    duration_day_std = np.nan
                    duration_day_sum = np.nan
                    max_temperature_max = np.nan
                    max_temperature_mean = np.nan
                    max_temperature_std = np.nan
                    severity_mean = np.nan
                    severity_std = np.nan
                    number_events = 0 
                else:
                    test_mean = heatwaves.mean()
                    test_std = heatwaves.std()
                    test_max = heatwaves.max()
                    test_sum = heatwaves[['duration_days','severity']].sum()
                    duration_day_mean = test_mean.loc['duration_days']
                    duration_day_max = test_max.loc['duration_days']
                    duration_day_std = test_std.loc['duration_days']
                    duration_day_sum = test_sum.loc['duration_days']
                    max_temperature_max = test_max.loc['max_temperature']
                    max_temperature_mean = test_mean.loc['max_temperature']
                    max_temperature_std = test_std.loc['max_temperature']
                    severity_max = test_max.loc['severity']
                    severity_sum = test_sum.loc['severity']
                    severity_mean = test_mean.loc['severity']
                    severity_std = test_std.loc['severity']
                    number_events = len(heatwaves)
                
                df["zcl"].append(zcl)
                df['warming'].append(warming)
                df['duration_day_mean'].append(duration_day_mean)
                df['duration_day_std'].append(duration_day_std)
                df['duration_day_sum'].append(duration_day_sum)
                df['duration_day_max'].append(duration_day_max)
                df['max_temperature_mean'].append(max_temperature_mean)
                df['max_temperature_max'].append(max_temperature_max)
                df['max_temperature_std'].append(max_temperature_std)
                df['severity_mean'].append(severity_mean)
                df['severity_std'].append(severity_std)
                df['severity_max'].append(severity_max)
                df['severity_sum'].append(severity_sum)
                df['number_events'].append(number_events)
                
        df = pd.DataFrame().from_dict(df)
        df.to_csv(os.path.join('output','heatwaves.csv'),index=False)
        
        df = pd.read_csv(os.path.join('output','heatwaves.csv'))
        print(df)
        
        fig,axs = plt.subplots(1,3,figsize=(16,5),dpi=300,sharex=True)
        axs = axs.flatten()
        
        ylabel_dict = {'duration_day_mean':'Mean heatwave duration (days)',
                       'duration_day_sum':'Total heatwave duration (days)',
                       'duration_day_max':'Max heatwave duration (days)',
                       'max_temperature_max':'Max heatwave temperature (°C)',
                       'max_temperature_mean':'Mean of max heatwave temperature (°C)',
                       'severity_mean':'Mean heatwave severity (°C.°C$^{-1}$)',
                       'severity_max':'Max severity (°C.°C$^{-1}$)',
                       'severity_sum':'Total severity (°C.°C$^{-1}$)'}
        
        ylim_dict = {e:(max(0,round(df[e].min())-2),round(df[e].max())+2) for e in ylabel_dict.keys()}
        
        var_list = ['duration_day_max','max_temperature_max','severity_max']
        # var_list = ['duration_day_sum','max_temperature_mean','severity_mean']
        # var_list = ['duration_day_sum','max_temperature_mean','severity_sum']
        
        for idx_var, var in enumerate(var_list):
            bottom,top = ylim_dict.get(var,(0,100))
                
            ax = axs[idx_var]
            for idx,zcl in enumerate(france.climats):
                if idx == 0:
                    label_obs_2026 = '2026'
                    label_obs_2003 = '2003'
                else:
                    label_obs_2026 = ''
                    label_obs_2003 = ''
                    
                ax.plot([idx-0.35,idx,idx+0.35],
                        [df[(df.warming=='2003')&(df.zcl==zcl)][var].values[0]]*3,lw=2,
                        color=get_warming_color(2003),label=label_obs_2003,zorder=5)
                
                ax.plot([idx-0.35,idx,idx+0.35],
                        [df[(df.warming=='2026')&(df.zcl==zcl)][var].values[0]]*3,lw=2,
                        color=get_warming_color(2026),label=label_obs_2026,zorder=5)
                
                for idx_warming,warming in enumerate(['2.0','2.7','4.0']):
                    if idx == 0:
                        label_warming = f'TRACC +{warming}°C'
                    else:
                        label_warming = ''
                        
                    ax.bar(x=[idx+(-1+idx_warming)*0.25],
                           height=[df[(df.warming==warming)&(df.zcl==zcl)][var].values[0]],
                           width=0.2,
                           color=get_warming_color(float(warming)),
                           label=label_warming)
                    
                    ax.plot([idx+(-1+idx_warming)*0.25],
                            [df[(df.warming==f'{warming}_heatwave')&(df.zcl==zcl)][var].values[0]],
                            marker='^',color=get_warming_color(float(warming)),mec='w',ls='',ms=10)
                    
                    if var_list == ['duration_day_max','max_temperature_max','severity_max']:
                        ref_values = ref_heatwave_values.loc[(Climat(zcl).center_city_caps,float(warming))].values[idx_var]
                        ax.plot([idx+(-1+idx_warming)*0.25],
                                [ref_values],
                                marker='.',color='k',mec='w',ls='',ms=5,zorder=5)
                    
                    if idx%2 == 1:
                        ax.fill_between([idx-0.5,idx+0.5],[top]*2,[bottom]*2,color='tab:grey',alpha=0.1,zorder=-2)
                    
            ax.plot([-1],[0],color='k',marker='^',label='5% heatwave',mec='w',ls='',ms=8)
            ax.set_xlim([-0.5,len(france.climats)-0.5])
            ax.set_ylabel(ylabel_dict.get(var,var))
            ax.set_xticks(list(range(len(france.climats))),france.climats)
            ax.set_ylim(bottom=bottom, top=top)
            ax.yaxis.get_major_locator().set_params(integer=True)
            
            if idx_var == 2:
                ax.legend(loc='upper left')
        
        plt.subplots_adjust(wspace=0.17)
        plt.savefig(os.path.join('output',f"heatwave_comparison_{'_'.join(var_list)}.png"), bbox_inches='tight')
        plt.show()
                
    
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()