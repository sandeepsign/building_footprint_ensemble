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

from colormath.color_objects import LabColor, sRGBColor 
from colormath.color_conversions import * 

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


def convert_img_float_to_uint(img_float):
    img_n = cv2.normalize(src=img_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_n.astype(np.uint8)


def find_interseting_fragment_in_image(colsrows_list, img):
    ''' Finds the parking segment which is within image passed.
        it assumes:
         - colsrows_list is polygon path 
         - whole parking fragment is withing image
         Example polygon Returned is:
             [(1344.0, 1160.0),(1284.0, 1163.0),(1279.0, 1195.0),(1414.0, 1191.0), (1385.0, 1163.0)]
         '''    
    img_center_row,img_center_col = np.ceil(img.shape[0]/2),np.ceil(img.shape[1]/2)
    if len(colsrows_list)<3:
        return None
    parking_polygon = convert_list2polygon([ (cr[0],cr[1]) for cr in colsrows_list])
    img_polygon = convert_list2polygon([(0.0,0.0),
                                        (0.0,float(img.shape[0]-1)), 
                                        (float(img.shape[1]-1),float(img.shape[0]-1)),
                                        (float(img.shape[1]-1),0.0)])
    
    plint = img_polygon.intersection(parking_polygon)
    
    if plint == None or (isinstance(plint, list)==False and plint.is_empty == True): #cases for invalid polygons etc.
        return None
    
    return list(zip(list(plint.exterior.coords.xy[0])[:-1],list(plint.exterior.coords.xy[1])[:-1]))

def crop_image(img, pts):
    ''' Crops img by using pts as polygon coords as list of [col,row].
        Returns minimum rectangular image with only inside polygon part as img,
        rest is black in rectagle i.e. empty/(0,0,0)'''
    height = img.shape[0]
    width = img.shape[1]
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    points = pts.reshape((1,-1,2))
    cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(img,img,mask = mask) #other than mask's ON location, rest is black now

    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    
    return cropped

def draw_line(img, p1, p2):
    ''' Draws line on the image object passed, from p1 to p2, points are in width,height format '''
    imgl = cv2.line(img,(int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),(0,0,255),5)
    return imgl

def annotate_boundingbox_on_image(canvas, box, color=(0, 255, 255), label=''):
    ''' draws rectangle for the bounding box(left,top,right,bottom).
        if canvas is not float32 then it will be converted into that.
        example value for box: [47, 140, 128, 296] (tl_x,tl_y,br_x,br_y) '''
    if canvas.dtype != np.float32:
        canvas = np.float32(canvas/255.0)
    cv2.rectangle(canvas, (box[0], box[1]),(box[2], box[3]), color, 2)  
    return canvas

def polygon_cr_to_boundingbox(polygon_cr):
    ''' takes a polygon as list of (col,row) values and gives bounding box as:
        colmin, rowmin : it is top-left(xmin,ymin) of bounding box for polygon
        colmax, rowmax : it is bottom-right(xmax,ymax) of bounding box for polygon
        returns:
            (colmin, rowmin), (colmax, rowmax)'''
    cr_arr = np.array(polygon_cr)
    return (np.min(cr_arr[:,0]),np.min(cr_arr[:,1])),(np.max(cr_arr[:,0]),np.max(cr_arr[:,1]))

def bbox_to_center(bbox_cr):
    ''' recived bbox in form of list of [ left, top, right, bottom]
        and return the mid point of bounding box'''
    left, top, right, bottom = tuple(bbox_cr)
    return (left+right)//2, (top+bottom)//2  

def image_folder_cleanup(folder_path):
    ''' scans the folder for all the .png files and
        checks if image has size 512 x 512. if not, deleted it.'''
    from pathlib import Path
    path_folder = Path(folder_path)
    pathlist = path_folder.glob('**/*.png')
    for p in path_folder.ls():
        img= plt.imread(p)
        if img.shape[:2]!=(512,512): #delete files which are not (512,512)
            p.unlink()
            print(f'deleted file {p.name}')
    #     if img.shape[2]==4: #Drop Alpha channel
    #         img = img[:,:,:3]
    #         plt.imsave(p, img)
    #         print(f'dropped alpha channel for {p.name} : {img.shape}')    
    
def get_surrounding_colors(img, pt, ignore_list):
    ''' find RGB triplets at location (pt[0],pt[1]) i.e. (r,c), within surrounding 3x3 area 
    ignores if surrounding color is in ignore_list '''
    r,c = pt[0],pt[1]

    rs = []
    if r > 1:
        rs.append(r-2)
    if r < img.shape[0]-2:
        rs.append(r+1)
    cs = []
    if c > 1:
        cs.append(c-2)
    if c < img.shape[1]-2:
        cs.append(c+2)
    
    coords = [(r,c) for r in rs for c in cs]
    
    clrs = []
    for co in coords:
        if tuple(img[co[0]][co[1]]) != (0, 255, 0):
            clrs.append(tuple(img[co[0]][co[1]]))
    
    return clrs

def show_colors_range(img, h_value=60):
    ''' Shows the mask of passed color on the img as mask, returns the 3-ch mask. 
        Color is to be provided as H value, default is green color as H=60
        img is normal RGB image, not hsv
        '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv[:,:,0], (h_value-10), (h_value+10))

    mask_3ch = np.copy(img)
    mask_3ch[:,:,0] = mask
    mask_3ch[:,:,1] = mask
    mask_3ch[:,:,2] = mask
    
    return mask_3ch

def visualize_images_list(imgs_list,cols=5,figsize=(20,20)):
    ''' Visualized list of images in single cell, we don't have to call plt.imshow() for each of them. '''
    rows = int(np.ceil(len(imgs_list)/cols))
    fig, axs = plt.subplots(rows,cols)
    fig.set_size_inches(figsize)
    r=0
    c=0
    for img in imgs_list:
        if len(axs.shape)==1:
            axs[c].imshow(img)
        else:
            axs[r][c].imshow(img)
        c+=1
        if c==cols:
            r+=1
            c=0

def get_random_color():
    ''' returns the randon RGB values '''
    import random
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return (r,g,b)


def annotate_text(anno_img, pointxy, text, color,thickness=1,size=1.0,font = cv2.FONT_HERSHEY_SIMPLEX):
    ''' Annotate passed text at location specfied in color passed. '''
    cv2.putText(anno_img,str(text), tuple(pointxy), font, size,  (255,255,255), thickness+2)
    cv2.putText(anno_img,str(text), tuple(pointxy), font, size,  color, thickness)
    return anno_img

def annotate_point_on_img(anno_img, point, color,thickness=4,clstr_n=0):
    ''' Annotates passed image with width,height coordinate location '''
    cv2.putText(anno_img,str(clstr_n), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0,0,0), thickness+2)
    cv2.putText(anno_img,str(clstr_n), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  color, thickness)
    return anno_img

def annotate_points_on_image(img, points,color=(0,255,255), cluster_id=0):
    ''' Annotates all the points passed in image passed '''
    for pt in points:  
        #annotate_cicle(img, tuple(pt),color=color)
        annotate_point_on_img(img,pt,color,thickness=4,clstr_n=cluster_id)
    return img

def annotate_cicle(img, centerxy, radius=20, color=(0,255,0), thickness=20):
    ''' Draws a circle at specified location using passed radius, color and thickness of line '''
    img = cv2.circle(img, centerxy, radius, color, thickness) 
    return img

def convert_polygon_to_xy(pl):
    ''' Pass a shapely polygon object and we will return all the (x,y) for boundary points '''
    # x,y = pl.boundary.xy
    x,y = pl.exterior.xy
    coords = []
    for t in zip(x,y):
        coords.append([int(t[0]),int(t[1])])
    return coords

def annotate_polygon_on_image(canvas, plg,color=(255,0,0),label=''):
    ''' Draws polygon on the canvas as base img using passed shapely polygon object.
        Polygon object is using col,row and coords xy
        returns the numpy array as updated image'''      
    
    coords = convert_polygon_to_xy(plg)
    plg_ctr = int(plg.representative_point().coords.xy[0][0]),int(plg.representative_point().coords.xy[1][0])
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1,1,2))
    canvas = cv2.polylines(canvas,[pts],True,color,1) 
    points_arr = np.array(coords)
    #polygon_center = (int(sum(points_arr[:,0])/points_arr.shape[0]),int(sum(points_arr[:,1])/points_arr.shape[0]))
    polygon_topleft = (int(np.min(points_arr[:,0])),int(np.min(points_arr[:,1])))
    if label:
        annotate_text(canvas, polygon_topleft,label, color,size=0.8, font=cv2.FONT_HERSHEY_SIMPLEX)        
    
    return canvas

def annotate_polygons_filled_on_image(canvas, polylines_xy):
    ''' Draws polygons on the canvas as base img using passed list of polygons coords  
     e.g. [[[5987, 89], [5888, 89], [5888, 0], [5987, 0], [5987, 89]],
          [[6127, 85], [6023, 82], [6026, 0], [6130, 2], [6127, 85]], ...
          returns the numpy array as updated image'''
    
    canvas = cv2.fillPoly(canvas,[np.array(pl,'int32') for pl in polylines_xy],(255,255,255))
    return canvas


def annotate_polygons_on_image(canvas, polylines_xy,color=(255,0,0),label='',label_location=(0,0),fill=False, alpha=0.25):
    ''' Draws polygons on the canvas as base img using passed list of polygons coords  
     e.g. [[[5987, 89], [5888, 89], [5888, 0], [5987, 0], [5987, 89]],
          [[6127, 85], [6023, 82], [6026, 0], [6130, 2], [6127, 85]], ...
          returns the numpy array as updated image'''
    for rxy in polylines_xy:
        pts = np.array(rxy, np.int32)
        pts = pts.reshape((-1,1,2))
        canvas = cv2.polylines(canvas,[pts],True,color,1) 
        
    if label:
        points_arr = np.array(polylines_xy)
        #polygon_center = (int(sum(points_arr[:,0])/points_arr.shape[0]),int(sum(points_arr[:,1])/points_arr.shape[0]))
        polygon_topleft = (int(np.min(points_arr[:,0])),int(np.min(points_arr[:,1])))
        annotate_text(canvas, polygon_topleft,label, color,size=2.0, font=cv2.FONT_HERSHEY_TRIPLEX)        
        
    return canvas


def shrink_or_swell_shapely_polygon(my_polygon, factor=0.10, swell=False):
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''
    from shapely import geometry

    #my_polygon = mask2poly['geometry'][120]

    factor = 0.10 #Shrink or swell by 10%
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*0.10

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance) #expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance) #shrink

    if my_polygon_resized.area == 0.0 or isinstance(my_polygon_resized, shapely.geometry.multipolygon.MultiPolygon):
        return my_polygon
    
    #visualize for debugging
    #x, y = my_polygon.exterior.xy
    #plt.plot(x,y)
    #x, y = my_polygon_resized.exterior.xy
    #plt.plot(x,y)
    ## to net let the image be distorted along the axis
    #plt.axis('equal')
    #plt.show()    
    
    return my_polygon_resized

def shapely_polygon_area(plg):
    ''' Calculates the area of shapely polygon in meters^2 '''
    from pyproj import Geod
    from shapely import wkt

    # specify a named ellipsoid
    geod = Geod(ellps="WGS84")

    area = abs(geod.geometry_area_perimeter(plg)[0])    
    
    return area

def shapely_point_to_xy(pt):
    ''' extracts point (col,row) value from shapely coords  '''
    return pt.coords.xy[0][0],pt.coords.xy[1][0]

def shapely_shape_to_point(shape):
    ''' Finds representative point for any shapely geopmetry as (col,row) or (lat,lng)  '''
    return list(shape.representative_point().coords.xy[0])[0],list(shape.representative_point().coords.xy[1])[0]

def shape_polygon_to_leftmost_point(plg):
    ''' finds the actual top-left point on polygon's exterior boundary, this is not bounding box top-left '''
    return sorted(convert_polygon_to_xy(plg), key=lambda xy: tuple(xy))[0]

def filter_bilateral(imgc,iterations=5):
    ''' Apply bilateral filter to img by running default params provided number of times. '''
    for i in range(0,iterations):
            imgc = cv2.bilateralFilter(imgc, 11, 27, 11) 
    #cv2.imwrite('mbfiltered.png',imgc)
    return imgc

def filter_median(imgc,iterations=5):
    ''' Apply median filter to img by running default params provided number of times. '''
    for i in range(0,iterations):
            imgc = cv2.medianBlur(imgc,3)
    #cv2.imwrite('mbfiltered.png',imgc)
    return imgc

def crop_all_polygons_exactly(img,mask,plgns):
    ''' Crops all the exact polygons(rotated) from the image passed and plgns is dict of all the polygon objects '''
    plgns_cropped = {}
    for pl_idx in plgns:
        xy = convert_polygon_to_xy(plgns[pl_idx].envelope)
        xy_arr = np.array(xy)
        xy_arr = np.where(xy_arr>=0, xy_arr, np.zeros_like(xy_arr)) 
        topleft = xy_arr[0]
        bottomright=xy_arr[2]
        cropped_plgn = img[topleft[1]:bottomright[1]+1, topleft[0]:bottomright[0]+1]
        cropped_mask = mask[topleft[1]:bottomright[1]+1, topleft[0]:bottomright[0]+1]
        exac_crop = cv2.bitwise_and(cropped_plgn,cropped_mask)
        plgns_cropped[pl_idx] = exac_crop

    return plgns_cropped

def distance_euclidean(l1,l2):
    return np.sqrt((l2[0]-l1[0])**2 + (l2[1]-l1[1])**2)


def convert_rgb_to_lab(color):
    ''' Converts RGB tripled value to LAB Color Space '''

    # sRGB Color
    color_rgb = sRGBColor(color[0], color[1], color[2])

    # Convert from sRGB to Lab Color Space
    color_lab = convert_color(color_rgb, LabColor)
    
    return color_lab

def calculate_lab_color_delta(color1, color2):
    
    # Convert from RGB to Lab Color Space
    color1_lab = convert_rgb_to_lab(color1)

    # Convert from RGB to Lab Color Space
    color2_lab = convert_rgb_to_lab(color2)

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab);

    #print("The difference between the 2 color = ", delta_e)
    return delta_e


def calculate_color_pair_differences(colors_pair1,colors_pair2):
    ''' Accepts 2 sets of colors, where each set is a pair of colors.
        And calculates color difference for each combination from list 1 and list 2. 
        Returns the delta of color pairs for all 4 compbinations from first list and second list'''
    
    color_deltas = []
    
    #00
    color_deltas.append(calculate_lab_color_delta(colors_pair1[0],colors_pair2[0]))
    #01
    color_deltas.append(calculate_lab_color_delta(colors_pair1[0],colors_pair2[1]))
    #10
    color_deltas.append(calculate_lab_color_delta(colors_pair1[1],colors_pair2[0]))
    #11
    color_deltas.append(calculate_lab_color_delta(colors_pair1[1],colors_pair2[1]))
    
    #print(color_deltas)
    
    return color_deltas

def color_pairs_match_check(colors_pair1,colors_pair2,color_delta_threshold=20.0):
    ''' checks if two color pairs probably are from similar roof '''
    #color_delta_threshold = 10.0
    clr_deltas = calculate_color_pair_differences(colors_pair1,colors_pair2)
    if len([d for d in clr_deltas if d < color_delta_threshold]) >= 2: #atleast 2 matching pairs.
        return True
    else:
        return False

def get_max_distance_between_polygon_pts(plg):
    import itertools
    coords_cr = convert_polygon_to_xy(plg)
    coords_cr = [tuple(xy) for xy in coords_cr]
    xy_pairs_nC2 = list(itertools.combinations(coords_cr,2))
    plg_dists = {xy_pair: distance_euclidean(xy_pair[0],xy_pair[1]) for xy_pair in xy_pairs_nC2}
    plg_dists = OrderedDict(sorted(plg_dists.items(), key=lambda t: -t[1]))
    plg_dia = list(plg_dists.values())[0]   
    return plg_dia

def find_most_dominant_colors(img_crop):
    ''' clusters the colors in 2 most frequent colors on the cropped image of roof.
        returns these cluster centers as colors as most frequent as first value 
        and less frequent as second value. 
        This also takes care of shaded very dark pixels by not counting them.
        '''
    ## Convert Cropped polygon to image without black(shaded area)
    
    cr = img_crop
    cr_flat = cr.reshape(-1,3)
    #drop zero pixels or near black pixels
    cr_flat = np.array([ t for t in cr_flat if convert_rgb_to_lab(t).lab_l>3000]) #Todo play with this value
    #b = cr_flat != [0,0,0]
    #pts_nonzero_mask = [np.any(p) for p in b]
    #pts_nonzero = cr_flat[pts_nonzero_mask]
    #pts_nonzero = pts_nonzero.reshape(-1,1,3) #This is image which is 1 pixel wide.
    #pts_nonzero = cr_flat.reshape(-1,1,3) #This is image which is 1 pixel wide
    if len(cr_flat) < 2:  
        pts_nonzero =  cr.reshape(-1,1,3) #if all pixels are dark, then use original crop and do now remove dark pixels for clustering
    else:
        pts_nonzero = cr_flat.reshape(-1,1,3)    
    
    #Calculate the 2 most frequent colorts by clustering them in 2 clusters
    #from __future__ import print_function
    import binascii
    import struct
    from PIL import Image
    #import numpy as np
    import scipy
    import scipy.misc
    import scipy.cluster

    NUM_CLUSTERS = 2

    ar = pts_nonzero
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    #print('finding clusters')
    try:
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    except:
        print(f'Exception thrown for {img_crop.shape}')
    #print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    #print('most frequent is %s (#%s)' % (peak, colour))    
    
    return [codes[np.argmax(counts)] , codes[np.argmin(counts)]]


def find_most_dominant_colors_for_all_plgns(exact_crops):
    ''' Find 2 most dominant colors for all the polygons crops.
        exact_crops is dictionary of pl_no : exactly cropped polygons'''
    mdc_plgns = {c:find_most_dominant_colors(exact_crops[c]) for c in exact_crops }
    mdc_plgns = {plno: [list(np.array(v, dtype=np.uint8))  for v in mdc_plgns[plno]] for plno in mdc_plgns}
    return mdc_plgns

def show_all_crops(crops,save=False):
    ''' Displays all the cropped images passed as dictionary of crops '''
    fig=plt.figure(figsize=(15, 15))
    columns = 5
    rows = np.ceil(len(crops)/5.0)
    for i in range(1, len(crops) +1):
        img_cropped = crops[i-1]
        if save:
            cv2.imwrite(f'plgn_{i}.png',img_cropped)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img_cropped)
    plt.show()

def show_dominant_color(img,polygons_mdc=None,plgn=0):
    ''' Find the 2 most dominant color in the image and displays image and 2 dominant colors '''  
    if polygons_mdc is None:
        mdc = find_most_dominant_colors(img) #most dominant color
    else:
        mdc = polygons_mdc[plgn]
        
    show_all_crops({0:img,1:[[ np.array(mdc[0],dtype=np.uint8),np.array(mdc[1],dtype=np.uint8) ]]})
    return mdc

def visualize_rgd_triplet(rgb_triplet):
    ''' Visualized the passed value of rgb triplet as list. 
    e.g. [[195, 151, 138]]'''
    plt.imshow([[rgb_triplet]*20]*20) 


