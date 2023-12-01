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
__script_name__ = 'kml_2_geojson.py'
__version__ = '1.0.0'
__profiling__ = 'False'

###############################################################################
###############################################################################

## Import libraries

import os
from osgeo import ogr

###############################################################################
###############################################################################

## Main

class kml2geojson:
    """
    Converts KML file to GeoJson format
    """
    #init
    def __init__(self):
        #use KML driver
        self.drv = ogr.GetDriverByName('KML')
    
    ###############################################################################
    
    #methods
    #convert
    def convert(self, kml_file: str, geojson_file: str, name: str):
        """
        Converts KML file to GeoJson format
        
        Args:
        ----------
        kml_file : str
            Read path to KML file
        geojson_file : str
            Write path for GeoJson file
        name : str
            Tag for 'name' field in GeoJson file
        
        Returns:
        ----------
        NA, writes to geojson file
        
        Raises:
        ----------
        
        """
        #open files
        kml_ds = self.drv.Open(kml_file)
        fw = open(geojson_file, 'w')
        
        #first feature counter
        i=0
        
        #write leader lines
        fw.write('{\n')
        fw.write('"type": "FeatureCollection",\n')
        fw.write(f'"name": "{name}",\n')
        fw.write('"features": [\n')
        
        #loop through layers
        for kml_lyr in kml_ds:
            #write features
            for feat in kml_lyr:
                #if first feature, export line to json and increment counter
                if i==0:
                    i+=1
                    line = feat.ExportToJson()
                #for n>1 features, write previous line with ','
                #then export current line to json
                else:
                    fw.write(f'{line},\n')
                    line = feat.ExportToJson()
        
        #write final line, then write closing lines
        fw.write(f'{line}\n')
        fw.write(']\n')
        fw.write('}')
        
        #close geojson file
        fw.close()