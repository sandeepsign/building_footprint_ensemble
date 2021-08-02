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

import sys
sys.path.append('../beans-ai-ml-cv-commons/')

from common_utils.logging_util import *
from common_utils.utils import *
from common_utils.download_file import *
from common_utils.basic import *
from common_utils.time_keeper import *
from common_utils.geo_utils import *
from common_utils.image_utils import *

import traceback2 as traceback
from google.cloud import storage
from tempfile import TemporaryFile,NamedTemporaryFile

GCP_KEY = 'AIzaSyDzgoaU7fIB4yux8TBjDUJxGa3hEAiQC1g'
SERVICE_ACCT_JSON = '/home/sandeep/Desktop/beans-home/environments/beans/beans-cloud-867403b137c5.json'
IMAGE_MAX_SIZE_TILES = 50000

PATH_IMGS = '/home/sandeep/GoogleDrive/beans-home/mount_shared_partition/images_home'
# PATH_IMGS_TILES = PATH_IMGS+'/tiles_xy'
PATH_IMGS_TEMP = PATH_IMGS+'/temp'

STORAGE_CLIENT = None




logger = setup_logging()

def create_sliding_windows(tiles_grid_xy,filter_size=4, stride=2):
    ''' Creates filtered areas of filter_size x filter_size with shift of stride given.
        Corner filtered areas as resize to match only area of interests'''
    tiles_grid_height, tiles_grid_width  = np.max(np.array(list(zip(*list(tiles_grid_xy.keys()))))[0,:])+1, \
        np.max(np.array(list(zip(*list(tiles_grid_xy.keys()))))[1,:])+1
    grid_centers = np.zeros((tiles_grid_height, tiles_grid_width,2))     
    for r,c in tiles_grid_xy:
        grid_centers[r,c] = xy2latlong(*tiles_grid_xy[r,c])
        
    offset_width = tiles_grid_width%filter_size
    offset_height = tiles_grid_height%filter_size
    
    from itertools import product

    top_lefts = list() #top left index of each filter placements

    filtered_areas = [] 
    for t in range(0, tiles_grid_height-filter_size+1, stride):
        for l in range(0,tiles_grid_width-filter_size+1,stride):
            top_lefts.append([t,l])
            filtered_areas.append(grid_centers[t:t+filter_size, l:l+filter_size])
        if offset_width:
            last_t,last_l = top_lefts[-1]
            top_lefts.append([last_t,last_l+stride])
            filtered_areas.append(grid_centers[last_t:last_t+filter_size, last_l+stride:])

    if offset_height:
        t = top_lefts[-1][0]+stride
        for l in range(0,tiles_grid_width-filter_size+1,stride):
            top_lefts.append([t,l])
            filtered_areas.append(grid_centers[t:t+filter_size, l:l+filter_size])

        if offset_width:
            last_t,last_l = top_lefts[-1]
            filtered_areas.append(grid_centers[last_t:last_t+filter_size, last_l+stride:])
    
    return filtered_areas


def create_area_image(fa,tile_size=512,maptype='basemap',dlm=None):
    ''' builds the image of the area with given set of centers as (lat,long)
        params are:
            fa: filtered area as 2d array of centers of tiled area
            dlm: tile download manager. If not provided it creates new one.
        returns:
            fa_canvas: Built image of all the tiles with given centers.
            top-left tiles: tile_x,tile_y'''
    if not dlm:
        dlm = TileDownloadManager(BUCKET_SAT_TILES,PATH_TILES_XY )
        
    area_rows, area_cols = fa.shape[:2]
    fa_canvas = np.zeros((area_rows*tile_size, area_cols*tile_size, 3), dtype=np.uint8)
    for r in range(fa.shape[0]):
        for c in range(fa[r].shape[0]):
            fa_tile,fa_tile_center,fa_tile_path =  dlm.get_map_tile(fa[r,c], maptype=maptype)
            fa_canvas[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = fa_tile  
    return fa_canvas, latlong2xy(*fa[0,0])

def create_image_from_filtered_areas(filtered_areas,tile_size=512,maptype='satellite',dlm=None, construct_image=True):
    ''' builds the image of all the areas with given as list of arrays of centers as (lat,long)
        params are:
            filtered_areas: list of filtered areas as each area is a 2d array of centers of tiled area
            dlm: tile download manager. If not provided it creates new one.
        returns:
            canvas: Built image of all the tiles in all the filtered areas given centers.
        '''
    if not dlm:
        dlm = TileDownloadManager(BUCKET_SAT_TILES,PATH_TILES_XY )

    fas_merged = np.concatenate([fa.reshape(-1,2) for fa in filtered_areas])
    fas_merged_xy = np.array(sorted(list(set([latlong2xy(*fa_ctr) for fa_ctr in fas_merged])), key=lambda xy:(xy[1],xy[0])))

    xmax, xmin = fas_merged_xy[:,0].max(), fas_merged_xy[:,0].min()
    ymax, ymin = fas_merged_xy[:,1].max(), fas_merged_xy[:,1].min()
    width = xmax-xmin+1
    height = ymax-ymin+1

    canvas_h = height*512
    canvas_w = width*512
    canvas = None
    if construct_image:        
        canvas = np.zeros((height*512,width*512,3),dtype=np.uint8)
        for xy in fas_merged_xy:
            xy_img = dlm.get_map_tile(xy2latlong(*xy), maptype=maptype)
            canvas[(xy[1]-ymin)*512:(xy[1]-ymin)*512+512,(xy[0]-xmin)*512:(xy[0]-xmin)*512+512] = xy_img[0]
    
    #Find center of this image
    topleft_ll = xy2latlong(xmin,ymin)
    bottomright_ll = xy2latlong(xmax,ymax)
    center_lat = bottomright_ll[0]+(topleft_ll[0]-bottomright_ll[0])/2
    center_long = topleft_ll[1]+(bottomright_ll[1]- topleft_ll[1])/2
    
    return canvas, (center_lat,center_long), (canvas_w, canvas_h)


#this download manager is attached to: 1 bucket in storage cloud and 1 local folder
class TileDownloadManager:
        
    def __init__(self, bucket_name=None, local_folder=None):
        self.bucket_name = bucket_name
        self.local_folder = local_folder
        self.client = storage.Client.from_service_account_json(SERVICE_ACCT_JSON)
        self.bucket = self.client.get_bucket(bucket_name)

    # def latlong2xy(self, lat_deg, lon_deg, zoom=20):
    #     ''' convertst lat,long to (tile_x,tile_y) number. 
    #         these are unique number for each tile. 
    #         Any (lat,long) in tile returns the same tile number.'''
    #     lat_rad = math.radians(lat_deg)
    #     n = 2.0 ** (zoom - 1)
    #     xtile = int((lon_deg + 180.0) / 360.0 * n)
    #     ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    #     return (xtile, ytile)

    # def xy2latlong(self, xtile, ytile, zoom=20):
    #     ''' Returns the center of (tile_x,tile_y) as (lat,long) '''
    #     xtile, ytile = xtile+0.5, ytile+0.5
    #     n = 2.0 ** (zoom - 1)
    #     lon_deg = xtile / n * 360.0 - 180.0
    #     lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    #     lat_deg = math.degrees(lat_rad)
    #     return (lat_deg, lon_deg)

    def create_tiles_grid(self, xtile,ytile, grid_size=3):
        ''' Creates the tiles grid with (xtile,ytile) as center of grid_size given '''
        from itertools import product
        offset = grid_size//2    
        grid_list = list(product( list(range(ytile-offset,ytile+offset+1)), list(range(xtile-offset,xtile+offset+1)) ))
        grid_arr = np.array(grid_list).reshape(3,-1)
        grid_dict = {(r,c):(grid_arr[r][c*2+1],grid_arr[r][c*2]) for r in range(grid_size) for c in range(grid_size)}
        return grid_dict

    def download_image_from_url(self, img_url):
        ''' Read image as numpy array from the url passed '''
        from skimage import io
        try:
            image_numpy = io.imread( img_url )
            return image_numpy
        except:
            return None


    def get_tile_name(self, tile_x,tile_y,zoom=20, maptype='satellite'):
        ''' constructs the tile's image name file '''        

        tile_name_prefix = 'sat'
        if maptype=='basemap':
            tile_name_prefix = 'base'
        
        TILE_NAME_TEMPLATE = f'tile_{tile_name_prefix}_z<zoom>_<tile_x>_<tile_y>.png'        

        return TILE_NAME_TEMPLATE.replace('<zoom>',str(zoom)).replace('<tile_x>',str(tile_x)).replace('<tile_y>',str(tile_y))

    def get_tile_bucket_path(self, tile_x,tile_y,  maptype='satellite' ,zoom=20):
        ''' constructs the bucket path using (tile_x,tile_y)'''
        tile_name = self.get_tile_name(tile_x,tile_y, maptype=maptype, zoom=zoom)
        blob_tile = self.bucket.blob(tile_name)
        if blob_tile.exists():
            return blob_tile.public_url
        else:
            return ''         


    def get_urls_for_files_in_bucket(self, folder_name, extensions=[]):
        ''' It tries to read all the file names in the folder inside bucket and returns their path 
            Note that:
            - folder_name does not include bucket_name
            e.g.
            folder_name = '0714cd4f-be45-463e-bea2-4a9f4087562a_27.904729647752177--82.50581840767953_27.9029103960991--82.51062358512971'

            - folder_name can be any prefix, not necessarily a folder. 
            e.g.
            folder_name = '0714cd4f-be45-463e-bea2-4a9f4087562a_27.904729647752177--82.50581840767953_27.9029103960991--82.51062358512971/0714cd4f-be45-463e-bea2-4a9f4087562a_27.904729647752177--82.50581840767953_27.9029103960991--82.51062358512971-'

            - extensions=['.png'] #list of extensions to be searched in folder, if empty, all extensions are provided.
        '''
        blobs= self.bucket.list_blobs(prefix=folder_name)
        file_paths = []
        for blob in blobs:
            if not extensions:
                file_paths.append(blob.name)
            elif blob.name[-4:].lower() in extensions:
                file_paths.append(blob.public_url)
        return file_paths

        
    # def convert_img_float_to_uint(self,img_float):
    #     img_n = cv2.normalize(src=img_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     return img_n.astype(np.uint8)

    def download_tile_xy(self, xtile, ytile, size='537x537',zoom=20, scaling=1,maptype='satellite'):
        ''' Download the sat view tile for the center location as: xtile,ytile
            e.g. 84191, 203259'''

        location = xy2latlong(xtile, ytile, zoom=zoom)

        tile_name_prefix = 'sat'
        if maptype=='basemap':
            tile_name_prefix = 'base'

        
        if maptype == 'satellite':
            url_template = f'https://maps.googleapis.com/maps/api/staticmap?center=location&size=tile_size&zoom={zoom}&scale=scaling&style=feature:all|element:labels|visibility:off&key={GCP_KEY}&maptype={maptype}'

        if maptype == 'hybrid':
            url_template = f'https://maps.googleapis.com/maps/api/staticmap?center=location&size=tile_size&zoom={zoom}&scale=scaling&style=feature:all|element:labels|visibility:off&key={GCP_KEY}&maptype={maptype}&style=feature:road|element:geometry.fill|color:0x00ff00'

        if maptype == 'basemap':
            url_template = f'https://maps.googleapis.com/maps/api/staticmap?center=location&size=tile_size&zoom={zoom}&scale=scaling&key={GCP_KEY}&style=feature:all|element:labels|visibility:off&style=feature:landscape.natural|element:geometry.fill|visibility:off&style=feature:all|element:geometry.stroke|visibility:off&style=feature:road|element:geometry.fill|color:0x00ff00'

        url=url_template.replace('location',f'{location[0]},{location[1]}').replace('tile_size',size).replace('zoom_level',str(zoom)).replace('scaling',str(scaling))
        
        if validators.url(url) :
            response = requests.get(url)
            img_file_name = f'{location[0]},{location[1]}'+'_sat.png' 
            Path(f'{PATH_IMGS_TILES}/').mkdir(parents=True, exist_ok=True)
            file_download_name =f'{PATH_IMGS_TILES}/tile_{tile_name_prefix}_z{zoom}_{xtile}_{ytile}.png'
            with open(file_download_name, 'wb') as file:
                file.write(response.content)
            logger.info(f'Downloaded image from url :  {url}')
        else:
            print('Not a valid URL')
            return None,None

        tile_np = convert_img_float_to_uint(plt.imread(file_download_name))
        return  tile_np[:,:,:3],file_download_name  


    def upload_tile_to_bucket(self, tile_name, img_np):
        ''' Writes the numpy image array to cloud bucket 
            Params are:
             - bucket object
             - folder name inside bucket
             - image as numpy array to be uploaded
            Returns:
             - URL in the bucket
        '''
        response_img_url = ''

        # Make sure we have default credentials created and use has admin permission, else we will get 403
        # Default credentials can be created by running following command in vm: 
        # gcloud auth application-default login

        with NamedTemporaryFile() as temp:
            #Extract name to the temp file
            cloud_path = f'{tile_name}'
            local_path = f'{self.local_folder}/{tile_name}'
            #Save image to local file
            cv2.imwrite(local_path, img_np[:,:,::-1])
            #Storing the local file inside the bucket
            blob_response = self.bucket.blob(cloud_path) 
            blob_response.upload_from_filename(local_path, content_type='image/png',timeout=500)
            self.make_google_storage_blob_public(blob_response)        
            response_img_url = blob_response.public_url
            logger.info(f'uploaded tile as {tile_name} at {response_img_url}.')
        return response_img_url

    def make_google_storage_blob_public(self,blob_object):
        ''' Makes the blob object for google storage publicly accessible. 
            Anyone with the link can access the the resource for the blob
            Params:
              blob_object : For the resource. 
            Returns: 
              True: If blob is made public.
              False: Otherwise. '''
        try:        
            acl = blob_object.acl
            acl.all().grant_read()
            acl.save()
            return True
        except:
            return False


    def get_map_tile(self, location,  tile_size=512 , zoom=20, maptype='satellite', scaling=1, local_download=True):
        ''' Gets the image tile containing location (lat,long).
        params:
            location: (lat,long) , this can be any point in tile to be read
            tile_size: size in pixels
            zoom: google maps api zoom level
            maptype: 'satellite' or hybrid' or 'basemap' 
            scalling: scale parameter for google maps api
        returns:
            img: numpy image array for tile.
            img_center: (lat,long) center of middle pixel of tile
            img_url: bucket url for tile image
        '''
        img = np.zeros((0,0))
        tile_x, tile_y = latlong2xy(location[0],location[1],zoom=zoom)
        img_center = xy2latlong(tile_x, tile_y, zoom=zoom)
        
        tile_fn = self.get_tile_name(tile_x, tile_y, maptype=maptype, zoom=zoom)
        tile_local_path = f'{self.local_folder}/{tile_fn}'
        #pdb.set_trace()
        #check if tile is available in local folder
        if Path(tile_local_path).exists():
            img = cv2.imread(tile_local_path)[:,:,::-1] #GBR to RGB
            logger.info(f'Found tile for ({tile_x},{tile_y}): {location} in local path f {tile_local_path}')
            return img[:,:,:3],img_center, tile_local_path
        
        #Check if tile is already in the bucket and if it can be downloaded
        tile_path = self.get_tile_bucket_path(tile_x, tile_y, maptype=maptype, zoom=zoom)
        #if found in bucket, get it from there
        if tile_path:
            #download tile from bucket and return as np image
            #print(f'from bucket : {tile_x, tile_y} ')
            img = self.download_image_from_url(tile_path)[:,:,:3]
            if local_download:
                plt.imsave(f'{self.local_folder}/{tile_path.split("/")[-1]}',img)
            if np.all(img):
                logger.info(f'Found tile for ({tile_x},{tile_y}): {location} in bucket path f {tile_path}')
                return img[:,:,:3],img_center,tile_path
        
        img_path_local= ''
        #if not in bucket, read from static maps api.
        try:
            #print(f'from static maps api : {tile_x, tile_y} ')
            img,img_path_local = self.download_tile_xy(tile_x, tile_y, size=f'{tile_size}x{tile_size+22}',zoom=zoom,maptype=maptype,scaling=scaling) 
        except:
            try:
                img,img_path_local = self.download_tile_xy(tile_x, tile_y, size=f'{tile_size}x{tile_size+32}',zoom=zoom,maptype=maptype,scaling=scaling) 
            except:
                img,img_path_local = self.download_tile_xy(tile_x, tile_y, size=f'{tile_size}x{tile_size+42}',zoom=zoom,maptype=maptype,scaling=scaling)

        if img.shape == (0, 0):
            logger.info(f'Not able to download image tile for location {location}')
            return None
        else:
            logger.info(f'downloaded tile for ({tile_x},{tile_y}): {location} from google maps static api.')
        
        #Fix image as needed
        if img.shape[0]-22 == tile_size:
            img =img[:-22,:]
        elif img.shape[0]-32 == tile_size:
            img =img[10:-32,:]
        else:
            img =img[20:-22,:]

        logger.info(f'map tile returned has shape : {img.shape}')
        
        #Upload read tile from static maps api tile to bucket with appropriate name
        img_url = self.upload_tile_to_bucket(tile_fn,img)

        #if we just want tiles to be stage in bucket, but do not want to keep download in local folder
        img_path_local = self.local_folder+'/'+tile_fn
        if not local_download:
            if os.path.exists(img_path_local):
                os.remove(img_path_local)
        
        return img[:,:,:3], img_center, img_url
        

    def collect_tiles_for_bounds(self, bounds, tile_size=512,zoom=20, maptype='satellite', local_download=True):
        ''' download all the image tiles for the area passed as bounds as one of ways:
            - Check if it is available in local folder.
            - Check if it is available in  google storage bucket for tiles 
            - or finally, download from google static image api
        Params are:
         - bounds as:  [(top,left),(bottom,right)]
         - maptype='satellite'(for plain satellite image no labels or roads) or 'basemap'(for simple map) or hybrid(with green roads)
         - if local_download=True, then tiles are downloaded into localf older as well. 
        Returns:
         - dictionary of all the tiles downloaded successfully as: 
            {(0,0): (tile_x,tile_y)....} here (0,0) is row-major index of tile, (tile_x,tile_y) of tile for this index
         - dictionary of all the tile centers (lat,long) with keys as (0,0) to (tiles_grid_height-1, tiles_grid_width-1)
         - dictionary of all the tiles and numpy arrays if local_download=True tiles_xy2images: 
         - Url in bucket for uploaded image: Not that url has actual top-left and bottom-right points coords
        '''

        top_lat,left_long =  bounds[0]
        bottom_lat,right_long = bounds[1]

        tiles_grid_xy,tiles_grid_width,tiles_grid_height = create_tiles_xy_grid_for_area(top_lat,left_long, bottom_lat,right_long, zoom= zoom)

        tiles_grid_centers = {tiles_grid_xy[(x,y)]:xy2latlong(*tiles_grid_xy[(x,y)])  for x,y in tiles_grid_xy}
        
        logger.info(f'Total number of tiles to be read are : {len(tiles_grid_centers)}, Estimated Cost: ${len(tiles_grid_centers)/1000*1.6}')

        if len(tiles_grid_centers) > IMAGE_MAX_SIZE_TILES:
            raise Exception(f'Too big of area for satellite image, more than {IMAGE_MAX_SIZE_TILES} tiles to be read!')

        # Download tiles - this just for training data, we never want to download lots of tiles to local machine
        tiles_xy2images = dict()
        tiles_xy2images = {(tile_x,tile_y):self.get_map_tile(tiles_grid_centers[(tile_x,tile_y)],  tile_size=tile_size,zoom=zoom, maptype=maptype ,local_download=local_download)[0]  for (tile_x,tile_y) in tiles_grid_centers}        

        logger.info(f'Image has been downloaded for : {bounds} , and it is uploaded to bucket : \n {self.bucket_name}')

        return tiles_grid_xy, tiles_grid_centers, tiles_xy2images

  
