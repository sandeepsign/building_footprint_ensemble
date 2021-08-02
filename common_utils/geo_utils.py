import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import torch
from collections import Counter 
from itertools import chain 
import time
import math
import matplotlib.path
import numpy as np
from collections import Counter 
from itertools import chain 
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import os
import cv2
from collections import Counter
from itertools import chain
import os
from io import BytesIO
import base64
import sys
from datetime import datetime 
from flask import Flask, render_template, escape, send_from_directory, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import numpy as np
import cv2
import urllib
import validators
import requests
import json
import pdb
from collections import OrderedDict 
from shapely.geometry import LineString
from google.cloud.sql.connector import connector
import google.protobuf.text_format
from collections import Counter
import math
import shapely.geometry
from shapely.geometry.polygon import Polygon
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
import pymysql

#Setup commons library
sys.path.append('../beans-ai-ml-cv-commons/')

from common_utils.logging_util import *
from common_utils.utils import *
from common_utils.download_file import *
from common_utils.basic import *
from common_utils.time_keeper import *
import traceback2 as traceback
from google.cloud import storage
from tempfile import TemporaryFile,NamedTemporaryFile

def get_center(points_list):
    ''' returns the center point by averaging all the points provided as list.
        Each point pt can be either of:
            - tuple of (lat,long) or (long,lat)
            - tuple of (row,col) or (col,row)
        '''
    xs,ys = [pt[0] for pt in points_list],[pt[1] for pt in points_list]
    return [sum(xs)/len(xs),sum(ys)/len(ys)]


def calculate_distance_between_coords(longlat1, longlat2):
    ''' Calculates the distance in meters between two points provided as (long,lat). 
        e.g.  (35.944142,-78.91827898)'''
    import geopy.distance
    return geopy.distance.geodesic(longlat1, longlat2).meters

def get_region_sizes_meter(nw,ne,se,sw):
    ''' Calculates the hight and w of the bounded area given as: nw,ne,se,sw
        Wigth and Height is calculated in meters.
    '''
    w = calculate_distance_between_coords(nw,ne)
    h = calculate_distance_between_coords(nw,sw)
    return w,h

def get_latlng_from_imagetilepoint(tile_center_lat, tile_center_lng, row, col, zoom=20, tile_size_w=512,tile_size_h=512):
    ''' Converts (row,col) location of image having center pixel with (tile_center_lat,tile_center_lng) 
        known, into (lat,lng)
        Params are:
            tile_center_lat, tile_center_lng: Known geo location of center pixel
            row, col: Candidate (row,col) pixel for which we want to find geo (lat,lng)
            zoom: Zoom level used, then this tile was read from good static api
            tile_size: size of image tile in pixel. 
            '''
    parallelMultiplier = math.cos(tile_center_lat * math.pi / 180.0)
    degreesPerPixelX = 360.0 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360.0 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = tile_center_lat - degreesPerPixelY * ( row - tile_size_h / 2.0)
    pointLng = tile_center_lng + degreesPerPixelX * ( col  - tile_size_w / 2.0)
    return (pointLat, pointLng)

def get_imagetilepoint_from_latlng(tile_center_lat, tile_center_lng, pointLat, pointLng, zoom=20, tile_size_w=512,tile_size_h=512):
    ''' Converts (pointLat, pointLng) location of image having center pixel with (tile_center_lat,tile_center_lng) 
        known, into (row,col)
        Params are:
            tile_center_lat, tile_center_lng: Known geo location of center pixel
        pointLat, pointLng: 
            zoom: Zoom level used, then this tile was read from good static api
            tile_size: size of image tile in pixel. 
        Returns:
            row, col: (row,col) of pixel for the geo (pointLat, pointLng) provided.
            '''
    parallelMultiplier = math.cos(tile_center_lat * math.pi / 180.0)
    degreesPerPixelX = 360.0 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360.0 / math.pow(2, zoom + 8) * parallelMultiplier
    row = (tile_center_lat - pointLat)/degreesPerPixelY + tile_size_h / 2.0 
    col = (pointLng - tile_center_lng)/degreesPerPixelX + tile_size_w / 2.0 
    return (round(row), round(col))

def polygon_colrow_to_latlong(polygon_cr,center,img_wh=[512,512],zoom_level=20):
    ''' Polygon is converted to be actual [lat,long] from [col,row] as image pixel  '''
    polygon_ll = []
    for cr in polygon_cr:
        polygon_ll.append(list(get_latlng_from_imagetilepoint(center[0],center[1],cr[1],cr[0], zoom=zoom_level, tile_size_w=img_wh[0],tile_size_h=img_wh[1])))
    return polygon_ll    

def polygon_latlong_to_colrow(polygon_ll,center,img_wh=[512,512],zoom_level=20):
    ''' Polygon is converted from actual [lat,long] to [col,row] as image pixel  '''
    polygon_cr = []
    for ll in polygon_ll:
        polygon_cr.append(list(get_imagetilepoint_from_latlng(center[0],center[1],ll[0],ll[1], zoom=zoom_level, tile_size_w=img_wh[0],tile_size_h=img_wh[1]))[::-1])
    return polygon_cr  

def convert_polygon_colrow_to_polygon_latlng(pl_cr,img,zoom_level=20,center_ll=None):
    ''' Takes are shapely polygon with col,row as their xy coords and converts it to 
        shapely polygon with lat,lng 
        params:
          pl_cr: shaply polygon with col,row as coords 
          img: image in which col,row are from top-left
          zoom_level: zoom level of sat image 
        returns:
          pl_ll: shapely polygon with lat,lng as coords'''
    coords_rc = list(zip(pl_cr.exterior.coords.xy[1].tolist(),pl_cr.exterior.coords.xy[0].tolist()))
    coords_ll = []
    for rc in coords_rc:
        coords_ll.append(get_latlng_from_imagetilepoint(center_ll[0], center_ll[1], rc[0],rc[1],tile_size_h=img.shape[0],tile_size_w=img.shape[1]))    
    pl_ll = convert_list2polygon(coords_ll)
    return pl_ll

def convert_polygon_latlng_to_polygon_colrow(pl_ll,img_wh,zoom_level=20):
    ''' Takes are shapely polygon with lat,lng  as their xy coords and converts it to 
        shapely polygon with col,row
        params:
          pl_ll: shaply polygon with lat,lng as coords 
          img_wh: image width and height
          zoom_level: zoom level of sat image 
        returns:
          pl_cr: shapely polygon with col,row as coords'''
    coords_ll = list(zip(pl_ll.exterior.coords.xy[1].tolist(),pl_ll.exterior.coords.xy[0].tolist()))
    coords_rc = []
    for ll in coords_ll:
        coords_rc.append(get_imagetilepoint_from_latlng(center_h[0], center_h[1], ll[0],ll[1],tile_size_h=img_wh[1],tile_size_w=img_wh[0]))    
    pl_cr = convert_list2polygon(coords_rc)
    return pl_cr

def shapely_point_to_xy(pt):
    ''' extracts point's coords value from shapely coords  as tuple pair '''
    return pt.coords.xy[0][0],pt.coords.xy[1][0]

def is_edge_polygon(img_h, plg_t):
    '''checks if shapely polygon is touching the edges of image '''
    img_h_ext0 = [[0,0],[0,-10],[img_h.shape[0]-1,-10], [img_h.shape[0]-1,0] ]
    img_h_ext1 = [[0,0],[-10,0],[-10, img_h.shape[1]-1], [0,img_h.shape[1]-1] ]
    img_h_ext2 = [[0,img_h.shape[1]-1],[0,img_h.shape[1]+10],[img_h.shape[0]-1,img_h.shape[1]+10], [img_h.shape[0]-1,img_h.shape[1]-1] ]
    img_h_ext3 = [[img_h.shape[0],0],[img_h.shape[0]-1,img_h.shape[1]-1],[img_h.shape[0]+10,img_h.shape[1]-1], [img_h.shape[0]-1,0] ]
    ext_plgns_cr = [img_h_ext0,img_h_ext1,img_h_ext2,img_h_ext3]
    ext_plgns = [convert_list2polygon(plg_cr) for plg_cr in ext_plgns_cr]    
    return np.any([plg_t.intersects(plg) for plg in ext_plgns ])

def latlong2xy( lat_deg, lon_deg, zoom=20):
    ''' convertst lat,long to (tile_x,tile_y) number. 
        these are unique number for each tile. 
        Any (lat,long) in tile returns the same tile number.'''
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** (zoom - 1)
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def xy2latlong( xtile, ytile, zoom=20):
    ''' Returns the center of (tile_x,tile_y) as (lat,long) '''
    xtile, ytile = xtile+0.5, ytile+0.5
    n = 2.0 ** (zoom - 1)
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def create_tiles_grid(xtile,ytile, grid_size=3):
    ''' Creates the tiles grid with (xtile,ytile) as center tile. Grid will be quare with hieght and width as grid_size given'''
    from itertools import product
    offset = grid_size//2    
    grid_list = list(product( list(range(ytile-offset,ytile+offset+1)), list(range(xtile-offset,xtile+offset+1)) ))
    grid_arr = np.array(grid_list).reshape(3,-1)
    grid_dict = {(r,c):(grid_arr[r][c*2+1],grid_arr[r][c*2]) for r in range(grid_size) for c in range(grid_size)}
    return grid_dict

def create_tiles_xy_grid_for_area(top_lat,left_long, bottom_lat,right_long,zoom=20):
    ''' Creates the indexed tiles grid for the area as (tile_x,tile_y). 
        Also, returns the grid_size_width,grid_size_height as number of tiles in width and height  '''
    from itertools import product
    topleft_x, topleft_y = latlong2xy(top_lat,left_long,zoom=zoom)
    bottomright_x, bottomright_y = latlong2xy(bottom_lat,right_long,zoom=zoom)    
    grid_size_width = bottomright_x-topleft_x+1
    grid_size_height = bottomright_y-topleft_y+1    
    grid_list = list(product( list(range(topleft_x, bottomright_x+1)), list(range(topleft_y, bottomright_y+1))))
    grid_dict = {(r,c):grid_list[c*grid_size_height+r] for r in range(grid_size_height) for c in range(grid_size_width)}
    return grid_dict,grid_size_width,grid_size_height

def convert_list2polygon(xy_list):
    ''' Takes a list of (x,y) and converts them in shapely polygon object.
        Note that x is column(or longitude) and y is row(or lattitude)
        This is reverse function of convert_polygon2list() '''
    if len(xy_list) > 2:
        return Polygon([(ll[0],ll[1]) for ll in xy_list])
    else:
        return None

def convert_polygon2list(shapely_polygon,reverse_order=False):
    ''' Takes a shapely polygon object and return list of (x,y) coords.
        Note that x is column(or longitude) and y is row(or lattitude)
        this is reverse function of convert_list2polygon() 
        reverse_order: if we need to reverse (x,y) to (y,x). needed for geojson as they have (long,lat) '''
    if shapely_polygon:
        xs,ys = shapely_polygon.exterior.coords.xy
    else:
        return None
    
    list_xy = list(zip(xs,ys))[:-1]
    
    if reverse_order:
        return [(xy[1],xy[0]) for xy in list_xy] #drop the repeated point in shapely polygons
    else:
        return list_xy

def polygon_list_to_boundingbox(polygon_cr):
    ''' takes a polygon as list of (col,row)/(lat,long) values and gives bounding box as:
        colmin, rowmin : it is top-left(xmin,ymin) of bounding box for polygon
        colmax, rowmax : it is bottom-right(xmax,ymax) of bounding box for polygon
        returns:
            (colmin, rowmin), (colmax, rowmax) : (top,left),(bottom,right)
            Note if it is list of (lat,long) then it will return as:
            (latmin,longmin), (latmax,longmax) : (bottom,left), (top,right)
            '''
    cr_arr = np.array(polygon_cr)
    return (np.min(cr_arr[:,0]),np.min(cr_arr[:,1])),(np.max(cr_arr[:,0]),np.max(cr_arr[:,1]))


def boundingbox_to_polyline_xy(bb_topleft, bb_bottomright):
    ''' take two tuples of (top,left) and (bottom,right). 
        and converts it to (top,left),(top,right), (bottom,right), (bottom, left) '''
    top,left = bb_topleft
    bottom,right = bb_bottomright
    return [(top,left),(top,right), (bottom,right), (bottom, left)]


def shapely_polygon_to_boundingbox(plg_shapely,reverse_order=False):
    ''' Calculates the shapely polygon's bounding box as:
      ((xmin,ymin),(xmax,ymax))'''
    plg_bounds = plg_shapely.bounds
    mins = min(plg_bounds[0],plg_bounds[2]),min(plg_bounds[1],plg_bounds[3])
    maxs = max(plg_bounds[0],plg_bounds[2]),max(plg_bounds[1],plg_bounds[3])
    
    if reverse_order:
        return mins[::-1],maxs[::-1]
    else:
        return mins,maxs

def estimate_tiles(top_left_ll, bottom_right_ll):
    ''' Takes top-left and bottom-right of the area as (lat,long)
        and calculates number of (x,y) tiles to be read for whole area'''
    tl_x,tl_y = latlong2xy(*top_left_ll)
    br_x,br_y = latlong2xy(*bottom_right_ll)   
    return (br_x-tl_x+1)*(br_y-tl_y+1)

def get_tile_name(tile_x,tile_y,zoom=20, maptype='satellite'):
    ''' constructs the tile's image name file '''        

    tile_name_prefix = 'sat'
    if maptype=='basemap':
        tile_name_prefix = 'base'
    
    TILE_NAME_TEMPLATE = f'tile_{tile_name_prefix}_z<zoom>_<tile_x>_<tile_y>.png'        

    return TILE_NAME_TEMPLATE.replace('<zoom>',str(zoom)).replace('<tile_x>',str(tile_x)).replace('<tile_y>',str(tile_y))

def is_overlapping(plg1,plg2, threshold=0.95):
    '''Checks if shapely polygon1 and polygon2 both are almost overlapping each other fully.
       it checks for IoU to be greater than threshold provided '''
    return plg1.intersection(plg2).area/plg1.union(plg2).area > threshold 

def find_overlapping_polygons(polygons_ll, threshold=0.95):
    ''' reduced the overlapping polygons by keep just one of the overlapping.
        extent of overlap is decided by threshold. 
        With higher threshold of 0.95, we can reduce poygon to unique non-overlapping polygons
        params:
           polygons_ll: dictionary of polygons with id as key and polygon as shapely object
           threshold: least extent to which Intersection-over-Union is considered overlapping
        returns:
            overlapping_groups: set of tuples, where each tuple of group of over-lapping polygon's ids '''
    overlapping_groups = set()
    #TODO: Change this n^2 approach to nearness based. check for transitive nearness
    for plgn1 in polygons_ll:
        dups = list()
        dups.append(plgn1)
        for plgn2 in polygons_ll:    
            if plgn1!=plgn2 and is_overlapping(polygons_ll[plgn1],polygons_ll[plgn2], threshold=threshold):
                dups.append(plgn2)
        overlapping_groups.add(tuple(sorted(list(dups))))
    return overlapping_groups

def xy2tilefn(tile_x, tile_y, maptype='sat',zoom=20):
    '''constructs the tile image file name by assuming convestion as:
        tile_{maptype}_z{zoom}_{tile_x}_{tile_y}.png
        e.g. tile_sat_z20_84377_203338.png
        maptype: either sat or base'''
    return f'tile_{maptype}_z{zoom}_{tile_x}_{tile_y}.png'

def tilefn2latlng(tile_fn):
    ''' finds the (lat,long) of the center of tile with given file name.
        e.g. tile_fn = tile_sat_z20_84429_203481.png 
        will return (37.32130128357806, -122.02686309814453)'''
    fn_base = tile_fn.split('/')[-1][:-4]
    tile_x,tile_y = fn_base.split('_')[-2],fn_base.split('_')[-1]
    return xy2latlong(int(tile_x),int(tile_y))
    
def find_representative_point(polygon):
    ''' Finds the represenative points of the passed polygon as shapely object '''
    return int(polygon.representative_point().x), int(polygon.representative_point().y)

def nearest_point_of_polygon(img, point, polygon):
    ''' Returns the nearest point on polygon's exterier from provided point '''
    from shapely.geometry import Polygon, Point, LinearRing
    pt = Point(point[0], point[1])
    pol_ext = LinearRing(polygon.exterior.coords)
    #set_trace()
    if pol_ext.contains(pt):
        return find_representative_point(pol_ext)
    d = pol_ext.project(pt)
    xy = pol_ext.interpolate(d)
    return int(xy.x), int(xy.y)

def nearest_points_for_polygons_pair(plg1,plg2):
    ''' Returns the nearest points on both polygon's exterier boundaries.
        Returns 2 points as:
        pt1: plg1's exterior point which nearest to plg2's exterior point.
        pt2: plg2's exterior point which nearest to plg1's exterior point.
        distance: between pt1 and pt2
        '''
    from shapely.geometry import Point, Polygon, LinearRing
    from shapely.ops import nearest_points    
    pt1,pt2 = nearest_points(plg1,plg2)
    return (int(pt1.coords.xy[0][0]),int(pt1.coords.xy[1][0])), (int(pt2.coords.xy[0][0]),int(pt2.coords.xy[1][0])), pt1.distance(pt2)


def create_horizontal_devides(topleft_ll, bottomright_ll, count=10):
    '''Creates count number of horizontal areas of equal size.
       params are:
           topleft_ll: (lat,long) of top-left of the candidate area
           bottomright_ll: (lat,long) of bottom-right of the candidate area
       returns:
           horizontal_devides: list of tuples of (topleft_ll, bottomright_ll) of count devides'''
    lat_diff = abs(topleft_ll[0]-bottomright_ll[0])

    tile_lat_height = 0.0005460535581249815 #each tile is around 61 meters/196 feet
    
    #If area if not big enough for atleast count devides, then do less number of devides
    if lat_diff < tile_lat_height*count :
        count = math.ceil(lat_diff/tile_lat_height)
        
    step_lat = (lat_diff)/count
    
    lats = []
    for i in range(count+1):
        lats.append(topleft_ll[0]-step_lat*i)

    tls = list(zip(lats[:-1], [topleft_ll[1]]*count))
    brs = list(zip(lats[1:], [bottomright_ll[1]]*count))
    horizontal_devides = list(zip(tls,brs))    
    return horizontal_devides

