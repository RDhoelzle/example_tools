#!/usr/bin/env python3.9.13

#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__maintainer__ = 'Rob Hoelzle'
__script_name__ = 'pull_new_era5_data.py'
__version__ = '1.0.0'
__profiling__ = 'False'

###############################################################################
###############################################################################

## Import libraries

import cdsapi
import os
import math
import pandas as pd
import numpy as np
import xarray
from datetime import datetime, date
from copy import deepcopy

###############################################################################
###############################################################################

## Main

class New_ERA5:
    """
    Tools for downloading and transforming ERA5 data into dataframes and series
    """
    #init
    def __init__(self):
        #max items allowed by API
        self.max_items = 100000
        #observation parameters
        self.era_class = {
            'wind' :          [],
            'wind_v' :        ['100m_v_component_of_wind',    'v100', 'WIND_V',        1],
            'wind_u' :        ['100m_u_component_of_wind',    'u100', 'WIND_U',        1],
            'temperature' :   ['2m_temperature',              't2m',  'TEMPERATURE',   1],
            'dewpoint' :      ['2m_dewpoint_temperature',     'd2m',  'DEWPOINT',      1],
            'precipitation' : ['total_precipitation',         'tp',   'PRECIPITATION', 2],
            'irradiance' :    ['surface_net_solar_radiation', 'ssr',  'IRRADIANCE',    2]}
        #unit conversion constants
        self.beta = 17.625
        self.gamma = 243.04
    
    ###############################################################################
    
    #methods
    #pull_data
    def pull_data(self, observation: list, coordinates: list, year_i: int = 1979, year_f: int = date.today().year):
        """
        Downloads specificied weather data from specified
        coordinate and returns a dataframe in Brisbane time
        
        Args:
        ----------
        observation : list, ['data']
            Must be one of wind, wind_v, wind_u, temperature,
            precipitation, or irradiance
        coordinates : list, [-90-90, -180-180]
            Latitude and longitude as list
        year_i : int, 'YYYY'
            Earliest year to pull data from, default 1979
        year_f : int, 'YYYY'
            Final year to pull data from, default current year
            
        Returns
        ----------
        new_df : pandas.DataFrame
            Timestamped hourly weather data in Brisbane time
        
        Raises
        ----------
            ValueError: If given observation is not available
            ValueError: If latitude or longitude is out of range
            ValueError: If end year is greater than current year
            ValueError: If initial year is less than 1940 or greater than end year
        """
        #check that specified observation is in list
        if not all(item in list(self.era_class.keys()) for item in observation):
            raise ValueError(f"'{observation}' not available, please choose from:\n{list(self.era_class.keys())}")
        
        #round coordinates and verify within range
        coord = [round(l, 3) for l in coordinates]
        
        #check that coordinates exist on globe
        if (coord[0] <= -89.999) | (coord[0] >= 89.999):
            raise ValueError(f"Latitude {coord[0]} out of range, must be between -89.999 and 89.999")
        if (coord[1] <= -179.999) | (coord[1] >= 179.999):
            raise ValueError(f"Longitude {coord[1]} out of range, must be between -179.999 and 179.999")
        
        #check that end year is not greater than the current year
        if year_f > date.today().year:
            raise ValueError(f"Final year {year_f} must not be greater than the current year, {date.today().year}")
        
        #check that the start year greater than 1939 and not greater than the final year
        if year_i < 1940:
            raise ValueError(f"Initial year {year_i} must be 1940 or later")
        if year_i > year_f:
            raise ValueError(f"Initial year {year_i} cannot be greater than final year {year_f}")
        
        #format coordinates for API
        for i in range(0,4):
            if i == 0:
                coord[i] = round(coord[i] + 0.001, 3)
            elif i == 1:
                coord[i] = round(coord[i] - 0.001, 3)
            elif i == 2:
                coord.append(coord[0] - 0.002)
            else:
                coord.append(coord[1] + 0.002)
        
        #subselect observations by set
        #if wind is selected, edit to wind components
        if 'wind' in observation:
            wind = True
            observation.remove('wind')
            if 'wind_v' not in observation:
                observation.append('wind_v')
            if 'wind_u' not in observation:
                observation.append('wind_u')
        else:
            wind = False
        
        #get observation subset for set 1
        obs1 = {key: self.era_class[key] for key in observation if self.era_class[key][3] == 1}
        if len(obs1) == 0:
            set1 = False
        else:
            set1 = True
            #calculate set 1 year step size such that #observations < max_items
            step1 = math.floor((year_f - year_i)\
                    / ((len(obs1) * (date.today() - datetime(year_i, 1, 1).date()).days * 24)\
                    / self.max_items)) - 1
            #define first year range for set 1
            if step1 < 0:
                year_s1 = year_f
            else:
                year_s1 = year_f - step1
        
        #get observation subset for set 2
        obs2 = {key: self.era_class[key] for key in observation if self.era_class[key][3] == 2}
        if len(obs2) == 0:
            set2 = False
        else:
            set2 = True
            #calculate set 2 year step size such that #observations < max_items
            step2 = math.floor((year_f - year_i)\
                    / ((len(obs2) * (date.today() - datetime(year_i, 1, 1).date()).days * 24)\
                    / self.max_items)) - 1
            #define first year range for set 2
            if step2 < 0:
                year_s2 = year_f
            else:
                year_s2 = year_f - step2
        
        #pull data in steps
        #set1
        if set1:
            #get list of search names
            cols = [obs1[key][1] for key in list(obs1.keys())]
            cols.insert(0,'valid_time')
            
            #get initial end year for date range
            #then loop through steps while end year >= initial year
            year_e = deepcopy(year_f)
            while (year_e >= year_i):
                #maintain min of year_i
                if year_s1 < year_i:
                    year_s1 = year_i
                
                #get list of search years
                if year_s1 == year_e:
                    years = [year_e]
                else:
                    years = [str(i) for i in range(year_s1, year_e + 1)]
                
                #check for data
                #extract dataset
                if os.path.isfile(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib"):
                    print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib")
                    new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib", engine='cfgrib')
                #or download and extract
                else:
                    print(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib not found, downloading")
                    c = cdsapi.Client()
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'grib',
                            'variable': [obs1[key][0] for key in list(obs1.keys())],
                            'month': ['01','02','03','04','05','06',
                                    '07','08','09','10','11','12',],
                            'day': ['01','02','03','04','05','06','07',
                                    '08','09','10','11','12','13','14',
                                    '15','16','17','18','19','20','21',
                                    '22','23','24','25','26','27','28',
                                    '29','30','31',],
                            'time': ['00:00','01:00','02:00','03:00','04:00',
                                    '05:00','06:00','07:00','08:00','09:00',
                                    '10:00','11:00','12:00','13:00','14:00',
                                    '15:00','16:00','17:00','18:00','19:00',
                                    '20:00','21:00','22:00','23:00',],
                            'area': coord,
                            'year': years,
                        },
                        'download.grib')
                    #rename download.grip to descriptive filename
                    os.rename('download.grib', f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib")
                    
                    #extract dataset
                    print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib")
                    new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{year_s1}_{year_e}.grib", engine='cfgrib')
                
                #convert to dataframe and process data
                print("Converting to dataframe")
                new_xr = new_xr.to_dataframe()
                #rename columns to descriptive names
                new_xr = new_xr[cols].reset_index(drop=True).rename(columns={'valid_time':'SETTLEMENTDATE'})
                new_xr.rename(columns={obs1[key][1]: obs1[key][2] for key in obs1}, inplace=True)
                
                #convert units
                #wind from components
                if wind:
                    new_xr['WINDSPEED'] = np.sqrt((np.square(new_xr.WIND_U) + np.square(new_xr.WIND_V)))
                    new_xr['WINDDIR'] = (180 + (180/math.pi)*np.arctan2(new_xr.WIND_U, new_xr.WIND_V))
                    new_xr.drop(columns=['WIND_V','WIND_U'], inplace=True)
                
                #degrees celcius
                if 't2m' in cols:
                    new_xr.TEMPERATURE = new_xr.TEMPERATURE - 273.15
                if 'd2m' in cols:
                    new_xr.DEWPOINT = new_xr.DEWPOINT - 273.15
                
                #initial dataframe, or merge
                if year_e == year_f:
                    new_df = deepcopy(new_xr)
                else:
                    new_df = pd.concat([new_xr, new_df], axis=0).reset_index(drop=True)
                
                #move to next year span
                year_e = year_s1 - 1
                year_s1 = year_e - step1
        
        #set2
        if set2:
            #get list of search names
            cols = [obs2[key][1] for key in list(obs2.keys())]
            cols.insert(0,'valid_time')
            
            #get initial end year for date range
            #then loop through steps while end year >= initial year
            year_e = deepcopy(year_f)
            while (year_e >= year_i):
                #maintain min of year_i
                if year_s2 < year_i:
                    year_s2 = year_i
                
                #get list of search years
                if year_s2 == year_e:
                    years = [year_e]
                else:
                    years = [str(i) for i in range(year_s2, year_e + 1)]
                
                #check for data
                #extract dataset
                if os.path.isfile(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib"):
                    print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib")
                    new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib", engine='cfgrib')
                #or download and extract
                else:
                    print(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib not found, downloading")
                    c = cdsapi.Client()
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'grib',
                            'variable': [obs2[key][0] for key in list(obs2.keys())],
                            'month': ['01','02','03','04','05','06',
                                    '07','08','09','10','11','12',],
                            'day': ['01','02','03','04','05','06','07',
                                    '08','09','10','11','12','13','14',
                                    '15','16','17','18','19','20','21',
                                    '22','23','24','25','26','27','28',
                                    '29','30','31',],
                            'time': ['00:00','01:00','02:00','03:00','04:00',
                                    '05:00','06:00','07:00','08:00','09:00',
                                    '10:00','11:00','12:00','13:00','14:00',
                                    '15:00','16:00','17:00','18:00','19:00',
                                    '20:00','21:00','22:00','23:00',],
                            'area': coord,
                            'year': years,
                        },
                        'download.grib')
                    #rename download.grip to descriptive filename
                    os.rename('download.grib', f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib")
                    
                    #extract dataset
                    print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib")
                    new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{year_s2}_{year_e}.grib", engine='cfgrib')
                
                #convert to dataframe
                print("Converting to dataframe")
                new_xr = new_xr.to_dataframe()
                #rename columns to descriptive names
                new_xr = new_xr[cols].reset_index(drop=True).rename(columns={'valid_time':'SETTLEMENTDATE'})
                new_xr.rename(columns={obs2[key][1]: obs2[key][2] for key in obs2}, inplace=True)
                
                #convert units
                #irradiance watts
                if 'ssr' in cols:
                    new_xr.IRRADIANCE = new_xr.IRRADIANCE / 3600
                
                #initial dataframe, or merge
                if set1:
                    if year_e == year_f:
                        new_df = pd.merge(new_df, new_xr, how='left',
                                        left_on='SETTLEMENTDATE', right_on='SETTLEMENTDATE')
                    else:
                        new_df.loc[new_df.SETTLEMENTDATE.isin(new_xr.SETTLEMENTDATE.tolist()), [obs2[key][2] for key in obs2]] = \
                            new_xr[[obs2[key][2] for key in obs2]]
                elif year_e == year_f:
                    new_df = deepcopy(new_xr)
                else:
                    new_df = pd.concat([new_xr, new_df], axis=0).reset_index(drop=True)
                
                #move to next year span
                year_e = year_s2 - 1
                year_s2 = year_e - step2
        
        #convert to Brisbane time
        new_df['SETTLEMENTDATE'] = new_df.SETTLEMENTDATE.dt.tz_localize('utc').dt.tz_convert('Australia/Brisbane').dt.tz_localize(None)
        
        #return final dataframe
        return new_df
    
    ###############################################################################
    
    #update_data
    def update_data(self, observation: list, coordinates: list):
        """
        Downloads updated data from the last month for the
        specificied weather parameter from specified
        coordinate and returns a dataframe in Brisbane time
        
        Returned dataframe intended to be upserted to existing
        table
        
        Args:
        ----------
        observation : list, ['data']
            Must be one of wind, wind_v, wind_u, temperature,
            precipitation, or irradiance
        coordinates : list, [-90-90, -180-180]
            Latitude and longitude as list
        
        Returns:
        ----------
        new_df : pandas.DataFrame
            Timestamped hourly weather data in Brisbane time
        
        Raises:
        ----------
            ValueError: If given observation is not available
            ValueError: If latitude or longitude is out of range
        """
        
        #check that specified observation is in list
        if not all(item in list(self.era_class.keys()) for item in observation):
            raise ValueError(f"'{observation}' not available, please choose from:\n{list(self.era_class.keys())}")
        
        #round coordinates and verify within range
        coord = [round(l, 3) for l in coordinates]
        
        #check that coordinates exist on globe
        if (coord[0] <= -89.999) | (coord[0] >= 89.999):
            raise ValueError(f"Latitude {coord[0]} out of range, must be between -89.999 and 89.999")
        if (coord[1] <= -179.999) | (coord[1] >= 179.999):
            raise ValueError(f"Longitude {coord[1]} out of range, must be between -179.999 and 179.999")
        
        #format coordinates for API
        for i in range(0,4):
            if i == 0:
                coord[i] = round(coord[i] + 0.001, 3)
            elif i == 1:
                coord[i] = round(coord[i] - 0.001, 3)
            elif i == 2:
                coord.append(coord[0] - 0.002)
            else:
                coord.append(coord[1] + 0.002)
        
        #subselect observations by set
        #if wind is selected, edit to wind components
        if 'wind' in observation:
            wind = True
            observation.remove('wind')
            if 'wind_v' not in observation:
                observation.append('wind_v')
            if 'wind_u' not in observation:
                observation.append('wind_u')
        else:
            wind = False
        
        #get observation subsets
        obs1 = {key: self.era_class[key] for key in observation if self.era_class[key][3] == 1}
        if len(obs1) == 0:
            set1 = False
        else:
            set1 = True

        obs2 = {key: self.era_class[key] for key in observation if self.era_class[key][3] == 2}
        if len(obs2) == 0:
            set2 = False
        else:
            set2 = True
        
        #define year and month spans
        if date.today().month > 1:
            yr = [f"{date.today().year}"]
            mn = [f"{date.today().month - 1:02d}", f"{date.today().month:02d}"]
        else:
            yr = [f"{date.today().year - 1}", f"{date.today().year}"]
            mn = [f"{date.today().month - 1:02d}", f"{date.today().month:02d}"]
        
        #pull data in steps
        #set1
        if set1:
            #get list of search names
            cols = [obs1[key][1] for key in list(obs1.keys())]
            cols.insert(0,'valid_time')
            #check for data
            #extract dataset
            if os.path.isfile(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib"):
                print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib", engine='cfgrib')
            #or download and extract
            else:
                print(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib not found, downloading")
                c = cdsapi.Client()
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': [obs1[key][0] for key in list(obs1.keys())],
                        'month': mn,
                        'day': ['01','02','03','04','05','06','07',
                                '08','09','10','11','12','13','14',
                                '15','16','17','18','19','20','21',
                                '22','23','24','25','26','27','28',
                                '29','30','31',],
                        'time': ['00:00','01:00','02:00','03:00','04:00',
                                 '05:00','06:00','07:00','08:00','09:00',
                                 '10:00','11:00','12:00','13:00','14:00',
                                 '15:00','16:00','17:00','18:00','19:00',
                                 '20:00','21:00','22:00','23:00',],
                        'area': coord,
                        'year': yr,
                    },
                    'download.grib')
                #rename download.grib to descriptive filename
                os.rename('download.grib', f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                
                #extract dataset
                print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs1[key][1] for key in list(obs1.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib", engine='cfgrib')
            
            #convert to dataframe and process data
            print("Converting to dataframe")
            new_xr = new_xr.to_dataframe()
            #rename columns to descriptive names
            new_xr = new_xr[cols].reset_index(drop=True).rename(columns={'valid_time':'SETTLEMENTDATE'})
            new_xr.rename(columns={obs1[key][1]: obs1[key][2] for key in obs1}, inplace=True)
            
            #convert units
            #wind from components
            if wind:
                new_xr['WINDSPEED'] = np.sqrt((np.square(new_xr.WIND_U) + np.square(new_xr.WIND_V)))
                new_xr['WINDDIR'] = (180 + (180/math.pi)*np.arctan2(new_xr.WIND_U, new_xr.WIND_V))
                new_xr.drop(columns=['WIND_V','WIND_U'], inplace=True)
            
            #degrees celcius
            if 't2m' in cols:
                new_xr.TEMPERATURE = new_xr.TEMPERATURE - 273.15
            if 'd2m' in cols:
                new_xr.DEWPOINT = new_xr.DEWPOINT - 273.15
            
            #finalize data frame
            new_df = deepcopy(new_xr)
        
        #set2
        if set2:
            #get list of search names
            cols = [obs2[key][1] for key in list(obs2.keys())]
            cols.insert(0,'valid_time')
            #check for data
            #extract dataset
            if os.path.isfile(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib"):
                print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib", engine='cfgrib')
            #or download and extract
            else:
                print(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib not found, downloading")
                c = cdsapi.Client()
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': [obs2[key][0] for key in list(obs2.keys())],
                        'month': mn,
                        'day': ['01','02','03','04','05','06','07',
                                '08','09','10','11','12','13','14',
                                '15','16','17','18','19','20','21',
                                '22','23','24','25','26','27','28',
                                '29','30','31',],
                        'time': ['00:00','01:00','02:00','03:00','04:00',
                                 '05:00','06:00','07:00','08:00','09:00',
                                 '10:00','11:00','12:00','13:00','14:00',
                                 '15:00','16:00','17:00','18:00','19:00',
                                 '20:00','21:00','22:00','23:00',],
                        'area': coord,
                        'year': yr,
                    },
                    'download.grib')
                #rename download.grib to descriptive filename
                os.rename('download.grib', f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                
                #extract dataset
                print(f"Opening {round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib")
                new_xr = xarray.open_dataset(f"{round(coord[0] - 0.001, 3)}_x_{round(coord[1] + 0.001, 3)}_{'_'.join([obs2[key][1] for key in list(obs2.keys())])}_{date.today().strftime('%Y-%m-%d')}.grib", engine='cfgrib')
            
            #convert to dataframe and process data
            print("Converting to dataframe")
            new_xr = new_xr.to_dataframe()
            #rename columns to descriptive names
            new_xr = new_xr[cols].reset_index(drop=True).rename(columns={'valid_time':'SETTLEMENTDATE'})
            new_xr.rename(columns={obs2[key][1]: obs2[key][2] for key in obs2}, inplace=True)
            
            #convert units
            #irradiance watts
            if 'ssr' in cols:
                new_xr.IRRADIANCE = new_xr.IRRADIANCE / 3600
            
            #finalize dataframe
            if set1:
                new_df = pd.merge(new_df, new_xr, how='left',
                                  left_on='SETTLEMENTDATE', right_on='SETTLEMENTDATE')
            else:
                new_df = deepcopy(new_xr)
        
        #convert to Brisbane time
        new_df['SETTLEMENTDATE'] = new_df.SETTLEMENTDATE.dt.tz_localize('utc').dt.tz_convert('Australia/Brisbane').dt.tz_localize(None)
        
        #return final dataframe
        return new_df
    
    ###############################################################################
    
    #heat_index
    def heat_index(self, temp: pd.Series, dpt: pd.Series):
        """
        Calculates heat index from temperature and dewpoint temperature
        
        Args:
        ----------
        temp : series, 'data'
            Dataframe series of temperature data
        dpt : series, 'data'
            Dataframe series of dewpoint data
        
        Returns:
        ----------
        hidx : series, 'data'
            Dataframe series of heat index data
        
        Raises:
        ----------
            ValueError: If temperature and dewpoint series do not have the same length
        """
        #verify temperature and dewpoint series are the same length
        if len(temp) != len(dpt):
            raise ValueError(f"Temperature series length ({len(temp)}) must equal dewpoint series length ({len(dpt)})")
        
        #convert to numpy series
        temp = temp.to_numpy()
        dpt = dpt.to_numpy()
        
        #calculate relative humidity
        rh = 100 * (np.exp((self.beta * dpt)/(self.gamma + dpt))\
                    / np.exp((self.beta * temp)/(self.gamma + temp)))
        
        #calculate temp as F
        tf = ((9/5) * temp) + 32
        
        #calculate base heat index
        hf = np.where(tf >= 80,
                        -42.379 + 2.04901523*tf + 10.14333127*rh - .22475541*tf*rh - .00683783*(tf**2) - 0.05481717*(rh**2)\
                            + 0.00122874*rh*(tf**2) + 0.00085282*tf*(rh**2) - 0.00000199*(tf**2)*(rh**2),
                        0.5 * (tf + 61 + ((tf-68)*1.2) + (rh*0.094)))
        
        #adjust heat index if TF >= 80 and RH < 13
        #or if 80 <= TF <= 87 and RH > 85
        hf = np.where(((tf >= 80) & (rh < 13)),
                            hf - (((13-rh)/4) * np.sqrt((17 - abs(tf-95))/17)),
             np.where(((tf >= 80) & (tf < 87) & (rh > 85)),
                            hf + (((rh-85)/10) * ((87-tf)/5)),
                            hf))
        
        #adjust back to C and return
        hidx = (hf - 32) * (5/9)
        return hidx