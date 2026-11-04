#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:30:49 2026

@author: amounier
"""

import time
import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


adm = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','communes-departement-region.csv'))
adm = adm.dropna(subset=['code_departement'])
adm['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in adm.code_departement]

geo = gpd.read_file(os.path.join('data','INSEE','decoupage_administratif','departements.geojson'))

zcl_df = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','zones_climatiques.csv'))
zcl_df['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in zcl_df.code_departement]

dict_code_dep_name_dep = {c:n for c,n in zip(adm.code_departement,adm.nom_departement)}
dict_name_dep_code_dep = {n:c for c,n in dict_code_dep_name_dep.items()}
dict_code_dep_geom_dep = {d:g for d,g in zip(geo.code,geo.geometry)}
dict_code_dep_code_zcl = {d:c for d,c in zip(zcl_df.code_departement,zcl_df.zone_climatique)}

prf = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','prefectures.csv'))
dict_name_dep_name_prf = {n:p for n,p in zip(prf.Département,prf.Préfecture)}

list_dep_code = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                 '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', 
                 '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', 
                 '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', 
                 '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', 
                 '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', 
                 '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', 
                 '88', '89', '90', '91', '92', '93', '94', '95']


def get_coordinates(city):
    coordinates_dict = {"Carpentras":(5.048768, 44.055404),
                        "Rennes":(-1.68002, 48.111339),
                        "Tours":(0.688927, 47.390047),
                        "Agen":(0.617611, 44.201583),
                        "Nancy":(6.18341, 48.693722),
                        "Mâcon":(4.832227, 46.303668),
                        "Trappes":(1.998836, 48.776096),
                        "Marignane":(5.214627, 43.416273)}
        
    longitude, latitude = coordinates_dict[city]
    return longitude, latitude

    
class City:
    def __init__(self,name):
        self.name = name
        self.coordinates = get_coordinates(self.name)
        self.departement_name = {v:k for k,v in dict_name_dep_name_prf.items()}.get(self.name)
        
        if self.departement_name is None:
            other_cities_dict = {'Trappes':'Yvelines',
                                 'Carpentras':'Vaucluse',
                                 'Marignane':'Bouches-du-Rhône'}
            self.departement_name = other_cities_dict.get(self.name)
            
        self.departement = Departement(dict_name_dep_code_dep.get(self.departement_name))
        
    def __str__(self):
        return self.name
    
    
class Departement:
    def __init__(self,dep_code):
        if type(dep_code) == int:
            self.code = "{:02d}".format(dep_code)
        elif type(dep_code) == str and len(dep_code) == 1:
            self.code = "{:02d}".format(int(dep_code))
        elif type(dep_code) == str and len(dep_code) == 2:
            if dep_code in ['2A','2B']:
                self.code = dep_code
            else:
                self.code = "{:02d}".format(int(dep_code))
        else:
            self.code = dict_name_dep_code_dep.get(dep_code)
            
        self.name = dict_code_dep_name_dep.get(self.code)
        self.codint = int(self.code.replace('A','01').replace('B','02'))
        self.geometry = dict_code_dep_geom_dep.get(self.code)
        self.climat = dict_code_dep_code_zcl.get(self.code)
        
        heatwave_thresholds = pd.read_csv(os.path.join('data','MF','seuils_canicules.csv'))
        self.Smin, self.Smax = heatwave_thresholds[heatwave_thresholds.dep==self.codint][['smin','smax']].values[0]
        
    def __str__(self):
        return '{} ({})'.format(self.name, self.code)
    
    
class Climat:
    def __init__(self,code):
        self.code = code
        self.departements = [Departement(k) for k,e in dict_code_dep_code_zcl.items() if e == self.code]
        self.geometry = unary_union([d.geometry for d in self.departements])
        self.codint = int(''.join([str({chr(ord('@')+n).lower():n for n in range(1,27)}.get(l.lower(),l)) for l in self.code]))
        # gestion du climat H3
        if self.codint<100:
            self.codint *= 10
        
        self.center_city = self.get_city()
        self.center_city_caps = self.center_city.upper().replace('Â','A')
        
        # valeurs SAFRAN
        dict_heatwave_thresholds_SAFRAN = {'H1a':[25.2,22.2,20.7],
                                           'H1b':[26.325,23.244,21.843],
                                           'H1c':[27.428,24.842,23.487],
                                           'H2a':[25.154,22.581,21.123],
                                           'H2b':[27.071,24.191,22.584],
                                           'H2c':[27.673,24.759,23.699],
                                           'H2d':[27.726,26.120,25.243 ], 
                                           'H3':[28.492,26.586,25.742]}
        # valeurs OBS
        dict_heatwave_thresholds_OBS = {'H1a':[26.4, 23.5, 21.7],
                                        'H1b':[25.2, 22.9, 21.6],
                                        'H1c':[27.3, 24.6, 23.4], 
                                        'H2a':[25.4, 22.8, 21.4],
                                        'H2b':[26.2, 23.5, 22.1],
                                        'H2c':[27.8, 25.2, 23.9],
                                        'H2d':[28.1, 26.8, 25.8],
                                        'H3':[28.8, 27.1, 26.2]}
        self.heatwave_thresholds_SAFRAN = dict_heatwave_thresholds_SAFRAN.get(self.code)
        self.heatwave_thresholds_OBS = dict_heatwave_thresholds_OBS.get(self.code)
        
        
        
    def __str__(self):
        return self.code
    
    def get_city(self, geographic=False):
        dict_center_city = {'H1a':'Trappes', 
                            'H1b':'Nancy', 
                            'H1c':'Mâcon', 
                            'H2a':'Rennes', 
                            'H2b':'Tours', 
                            'H2c':'Agen', 
                            'H2d':'Carpentras', 
                            'H3':'Marignane'}
        center_city = dict_center_city.get(self.code)
        return center_city


class France:
    def __init__(self):
        self.departements = [Departement(e) for e in list_dep_code]
        self.climats = sorted(list(set([e.climat for e in self.departements])))
        self.geometry = unary_union([Climat(d).geometry for d in self.climats])
    
    
def blank_national_map():
    def get_extent():
        extent = [-5, 9.8, 41.3, 51.3]
        return extent

    fig = plt.figure(figsize=(7,7), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(get_extent())
    
    ax.add_feature(cfeature.OCEAN, color='lightgrey',zorder=2)
    ax.add_feature(cfeature.LAND, color='w',zorder=1)
    ax.add_feature(cfeature.COASTLINE,zorder=5)
    ax.add_feature(cfeature.BORDERS,zorder=3)
    return fig,ax


def draw_climat_map(dict_zcl,figs_folder,cbar_min=0,cbar_max=1.,
                    automatic_cbar_values=False, cbar_label=None, no_cbar=False,
                    map_title=None,save=None, cmap=None,zcl_label=False,alpha=None,
                    add_city_points=None, add_legend=True,lw=None,figax=None,
                    border_color='k',point_color=None):
    
    if figax is not None:
        fig,ax = figax
    else:
        fig,ax = blank_national_map()
    
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    elif type(cmap) == np.ndarray:
        cmap = LinearSegmentedColormap.from_list('name', cmap, N=len(cmap))
    
    plotter = pd.DataFrame().from_dict({'climats':dict_zcl.keys(),'vals':dict_zcl.values()})
    plotter['geometry'] = [d.geometry for d in plotter.climats]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.vals.quantile(0.99)
        cbar_min = plotter.vals.quantile(0.01)
        cbar_extend = 'both'
    else:
        cbar_extend = 'neither'
    
    plotter['color'] = (plotter.vals-cbar_min)/(cbar_max-cbar_min)
    plotter['color'] = plotter['color'].apply(cmap)
    plotter['color'] = [(1.,1.,1.,1.) if e == (0.,0.,0.,0.) else e for e in plotter.color ]
    
    # print(plotter.color)
    plotter.plot(color=plotter.color, ax=ax, transform=ccrs.PlateCarree(),alpha=alpha)
    plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color=border_color,lw=lw)
    
    if type(zcl_label) == dict:
        for zcl,v in zcl_label.items():
            x = zcl.geometry.centroid.x
            y = zcl.geometry.centroid.y
            if zcl.code == 'H2a':
                x,y = (-2.91346, 48.05539)
            if zcl.code == 'H2c':
                x,y = (0.59459, 43.604462)
            if zcl.code == 'H3':
                x,y = (6.06993, 43.23685)
            ax.text(x, y, v, 
                    horizontalalignment='center', transform=ccrs.Geodetic(), zorder=20, color='k',
                    bbox=dict(facecolor='w', alpha=0.37))    
    elif zcl_label:
        for zcl in dict_zcl.keys():
            x = zcl.geometry.centroid.x
            y = zcl.geometry.centroid.y
            if zcl.code == 'H2a':
                x,y = (-2.91346, 48.05539)
            if zcl.code == 'H2c':
                x,y = (0.59459, 43.604462)
            if zcl.code == 'H3':
                x,y = (6.06993, 43.23685)
            ax.text(x, y, '{}'.format(zcl.code), 
                    horizontalalignment='center', transform=ccrs.Geodetic(), zorder=20, color='k',
                    bbox=dict(facecolor='k', alpha=0.1))
            
    if add_city_points is not None:
        if len(add_city_points) == len(dict_zcl.keys()):
            for i,city in enumerate(add_city_points):
                if point_color is None:
                    color = None
                    mec = 'k'
                    ms = 6
                elif point_color == 'same':
                    color = plotter['color'].iloc[i]
                    mec = 'w'
                    ms = 8
                else:
                    color = cmap(0.5)
                    mec = 'k'
                    ms = 6
                city = City(city)
                zcl = list(dict_zcl.keys())[i]
                ax.plot(city.coordinates[0],city.coordinates[1], 
                        transform=ccrs.PlateCarree(), color=color,ls='',ms=ms,
                        marker='o',label='{} : {}'.format(zcl.code, city.name),mec=mec,zorder=5)
        else:
            for i,city in enumerate(add_city_points):
                if len(add_city_points)>1:
                    color = None
                else:
                    color = cmap(0.5)
                city = City(city)
                ax.plot(city.coordinates[0],city.coordinates[1], 
                        transform=ccrs.PlateCarree(), color=color,ls='',
                        marker='o',label=city.name,mec='k',zorder=5)
        
    if not all(plotter.color==(1.,1.,1.,1.)) and not no_cbar:
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        norm = Normalize(vmin=cbar_min, vmax=cbar_max)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        
        cbar_label_var = cbar_label
        _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend=cbar_extend, extendfrac=0.02)
    
    if add_legend:
        ax.legend(ncol=3,loc='lower center')
        
    ax.set_title(map_title)
    if save is not None:
        os.makedirs(figs_folder,exist_ok=True)
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save)),bbox_inches='tight')
    return fig,ax



def get_warming_color(warming_level):
    color_dict = {2003:'tab:green', 
                  2026:'tab:blue',
                  2.0: (247,148,32),
                  2.7: (231,29,37),
                  4.0: (149,27,30),}
    for k in color_dict.keys():
        if k in [2003,2026]:
            continue
        v0,v1,v2 = color_dict.get(k)
        color_dict[k] = (v0/255,v1/255,v2/255)
    return color_dict.get(warming_level)
    
#%% ===========================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()
    
    #%% Carte des zones climatiques
    if True:
        france = France()
        
        zcl_dict = {Climat(e):None for e in france.climats}
        cmap = plt.get_cmap('viridis')
        
        draw_climat_map(zcl_dict,zcl_label=True, 
                        figs_folder='output', save='zcl_map',cmap=cmap,
                        no_cbar=True,alpha=0.6,point_color='else',
                        add_city_points=[Climat(c).center_city for c in france.climats],lw=0.7)
        
    #%% Récupération des départements (format MF https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires)
    if False:
        france = France()
        cities_list = [City(Climat(e).center_city) for e in france.climats]
        
        for city in cities_list:
            # print(f'HOR_departement_{city.departement.code}_periode_2025-2026')
            # print(f'HOR_departement_{city.departement.code}_periode_1990-1999')
            print(f'HOR_departement_{city.departement.code}_periode_2000-2009')
            
       
        # test = pd.read_parquet(os.path.join('data','MF','HOR_departement_13_periode_2025-2026.parquet'))

    #%% Formatage du tableau de seuils
    if True:
        dict_heatwave_thresholds_SAFRAN = {'H1a':[25.2,22.2,20.7],
                                           'H1b':[26.325,23.244,21.843],
                                           'H1c':[27.428,24.842,23.487],
                                           'H2a':[25.154,22.581,21.123],
                                           'H2b':[27.071,24.191,22.584],
                                           'H2c':[27.673,24.759,23.699],
                                           'H2d':[27.726,26.120,25.243 ], 
                                           'H3':[28.492,26.586,25.742]}
        
        text = ""
        text += '\\begin{tabular}{lccc|lccc}\n\t\\toprule\n'
        text += '\t& $S_1$ (\\SI{}{\\celsius}) & $S_2$ (\\SI{}{\\celsius}) & $S_3$ (\\SI{}{\\celsius})&  & $S_1$ (\\SI{}{\\celsius}) & $S_2$ (\\SI{}{\\celsius}) & $S_3$ (\\SI{}{\\celsius}) \\\\\n\t\\midrule\n'
        text += f"\tH1a & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1a')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1a')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1a')[2]}}}"
        text += f"& H2b & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2b')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2b')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2b')[2]}}} \\\\\n"
        text += f"\tH1b & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1b')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1b')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1b')[2]}}}"
        text += f"& H2c & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2c')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2c')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2c')[2]}}} \\\\\n"
        text += f"\tH1c & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1c')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1c')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H1c')[2]}}}"
        text += f"& H2d & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2d')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2d')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2d')[2]}}} \\\\\n"
        text += f"\tH2a & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2a')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2a')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H2a')[2]}}}"
        text += f"& H3 & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H3')[0]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H3')[1]}}} & \\num{{{dict_heatwave_thresholds_SAFRAN.get('H3')[2]}}} \\\\\n"
        text += '\t\\bottomrule\n\\end{tabular}'
        print(text)
        
        
    
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()