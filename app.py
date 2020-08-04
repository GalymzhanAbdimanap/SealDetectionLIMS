#!flask/bin/python
"""
Detect stamp
Asynchronous implementation of seals (stamps) detection using Mask R-CNN model.
Implements API of detecting REQUEST_HOST_URL, calls API for saving result RESPONSE_API_URL.
Takes params and writes logs according to severity level to file or output.

Copyright (c) 2020 IDET.kz
Written by Galymzhan Abdymanap.
"""

# Address of this request dispatcher
REQUEST_HOST_URL = "127.0.0.1"

REQUEST_HOST_PORT = 8839

# Server address receiving the response with detected results.
RESPONSE_API_URL = "http://lims.llpcmg.kz/api/CheckFile/InsertResult"

# All logs will be saved in this directory. 
LOG_DIR_NAME = "logs"

# Model
WEIGHTS_DIR_NAME = "weights"
SEAL_WEIGHTS_FILE_NAME = "mask_rcnn_seal_0030.h5"


from flask import Flask, jsonify, abort, make_response,request, json, redirect, render_template
from flask_restplus import Api, Resource, fields

# Create async pool
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=3)

import multipart as mp

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Import Mask RCNN
# To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import tensorflow as tf
import requests

from pdf2image import convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

# Import support modules from this project
import parsers

#------------------------------------------------------------------------------
# Parse params and create logger
#------------------------------------------------------------------------------
import argparse
import logging

try:
    # Create parser 
    parser = argparse.ArgumentParser(description='Stamp (seal) detector script.')
    parser.add_argument('-f', dest = 'log_file_name', type = str, help = 'output log file name')
    parser.add_argument('-l', dest = 'log_level', type = str, \
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default = 'DEBUG', \
        help = 'logging level')
    # required = True, 

    parser.print_help()

    args = parser.parse_args()

    # Get int value by name
    log_level = logging._nameToLevel[args.log_level]

    # If log file name is None then logs will be printed to std out.
    # If not make full path name.
    if args.log_file_name:
        args.log_file_name = os.path.join(os.path.abspath(""), LOG_DIR_NAME, args.log_file_name)

    # Configure logger, 
    logging.basicConfig(level = log_level, filename = args.log_file_name, format='%(asctime)s :: %(name)s - %(levelname)s :: %(message)s')

    # Add additional logging handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(handler)

    lvl_name = logging.getLevelName(logging.getLogger().getEffectiveLevel())
    logging.info(f'Current logging level: {lvl_name}')
    
except Exception as ex:
    print(ex)
    logging.exception(ex)

#------------------------------------------------------------------------------
# Mask RCNN model
#------------------------------------------------------------------------------
graph = tf.get_default_graph()

ROOT_DIR = os.path.abspath("")

sys.path.append(ROOT_DIR)



MODEL_DIR = os.path.join(ROOT_DIR, WEIGHTS_DIR_NAME)

class SealConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "seal"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + seal
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

config = SealConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

logging.debug(f'MaskRCNN MODEL_DIR: {MODEL_DIR}')

# Or, load the last model you trained
weights_path = model.find_last()
#weights_path = os.path.abspath('G:\Stuff\Daniyar\Development\Python\detectStamp\logs\seal20200221T1748\mask_rcnn_seal_0030.h5')

# Load weights
logging.debug(f'Loading weights: {weights_path}')
model.load_weights(weights_path, by_name = True)

logging.info(f'MaskRCNN created, device: {DEVICE}')



#------------------------------------------------------------------------------
# Flask init
#------------------------------------------------------------------------------

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Seal Detection", 
		  description = "Asynchronous implementation of seals (stamps) detection using Mask R-CNN model.")

name_space = app.namespace('detectSeal', description='For seal detection')


#------------------------------------------------------------------------------
# Stamp detection
#------------------------------------------------------------------------------

def post_checking(masks, rois, max_inappropriate_cells_percent = 10): 
    """Returns True if number of inappropriate cells percent less than given maximum."""
    assert masks is not None
    assert rois is not None

    # Crop place where there is a mask
    masks = masks[rois[0]:rois[2],rois[1]:rois[3]]
    masks = masks.reshape(masks.shape[0], masks.shape[1], 1)

    # Create an ideal seal shape in the input matrix
    ideal_seal = np.zeros((masks.shape[0],masks.shape[1]),  dtype=bool)
    ideal_seal = ideal_seal.reshape(masks.shape[0],masks.shape[1],1)
    for i in range(len(ideal_seal)):
        for j in range(len(ideal_seal[i])):
            if np.sqrt(pow((len(masks[0])/2-i),2)+pow((len(masks[0])/2-j),2))<=len(masks[0])/2: # add with Pythagorean theorem
                ideal_seal[i][j]=True
                
    # Comparison input matrix and ideal seal matrix
    res = np.subtract(ideal_seal, masks, dtype=np.int)

    # Counting inappropriate cells in matrix between two matrix 
    count = 0
    for i, el in enumerate(res):
        for j in el:
            if j!=0:
                count+=1
    
    # Counting percent of inappropriate cells in matrix
    shape_arr = masks.shape[0] * masks.shape[1] * masks.shape[2]
    res_perc = count * 100 / shape_arr

    logging.debug(f'Percent of inappropriate cells in matrix: {res_perc}')

    return True if res_perc < max_inappropriate_cells_percent else False


#------------------------------------------------------------------------------

def detect_seal(image, min_probability = 0.95, min_width_height_ratio = 0.9):
    """ Detects stamp (seal) in given image.
        Returns True if detected with appropriate probability and width to height image ratio.
    """
    assert image is not None

    results_rois = []
    results_scores = []

    # Set verbose param for model if DEBUG or INFO severity is set
    verbose_param = logging.getLogger().getEffectiveLevel() < logging.WARNING
    
    # 
    results = model.detect([image], graph, verbose = verbose_param)
    logging.debug(results)

    # Parse detected result
    r = results[0] 
    rois = r['rois'] # format r['rois']=[y1,x1,y2,x2]
    scores = r['scores']
    masks = r['masks']
        
    for i, result in enumerate(scores):
        mask = masks[:,:,i]
        if result > min_probability and (rois[i][3]-rois[i][1])/(rois[i][2]-rois[i][0]) > min_width_height_ratio: # probability results AND width to height ratio
            if post_checking(mask, rois[i]):   
                results_rois.append(rois[i])
                results_scores.append(result)
                return True
            
    return False


#------------------------------------------------------------------------------

def async_file_processing(file_id, bytearr):
    """ Implements the body of async function for detecting stamps (seals) in given binary file with multy pages.
        Calls API by RESPONSE_API_URL to save the results.
    """
    assert file_id
    assert len(bytearr) > 0

    try:
        logging.debug(f'async_file_processing file_id={file_id}, bytearr={len(bytearr)}')

        # Get pages
	# For OS Windows / for the independence of the program from the built-in OS libraries, the poppler library is included in the program folder
        # pages = convert_from_bytes(bytearr, poppler_path='env/poppler-0.68.0/bin')
	# For OS Linux
        pages = convert_from_bytes(bytearr)
        res = None
        logging.debug(f'pages={len(pages)}')
        
        # Iterate by pages and detect stamp
        for i,page in enumerate(pages):
            logging.debug(f'ID={file_id}, page={i}')

            img_array = np.array(page)
            result = detect_seal(img_array)
            if result:
                res = True
                break
            else:
                res = False
        
        # Create JSON
        data = json.dumps({'ID':file_id, 'StampExist': res})
        logging.info(f'File processed: {data}')

        # Call API to save detected results
        r = requests.post(RESPONSE_API_URL, data = data, headers = {"content-type" : "application/json"})
        logging.info(f'API response: {r}')
        #print(r)
        # except requests.exceptions.RequestException as e:  # This is the correct syntax
        #     print(e)
        #return jsonify({'ID': file_id, 'StampExist': res})
    
    except Exception as ex:
        logging.exception(ex)

#------------------------------------------------------------------------------







#------------------------------------------------------------------------------

@name_space.route("/")
class MainClass(Resource):

	
	@app.expect(parsers.file_upload, parsers.id_of_file, validate=True)		
	def post(self):
		""" Implements API of POST processing for detecting stamps (seals).
		    Processing of each file goes asynchorously, function does not wait for result.
		"""
		logging.debug(f'Request: {request.method}')
		
		if request.method == "POST":
			if 'File' not in request.files:
				logging.warning(f'No file part in request.')
				return redirect(request.url)

			bin_file = request.files['File']
			file_id = request.form['ID']
			logging.info(f'POST ID={file_id}, filename={bin_file.filename}')

			if bin_file.filename == '':
				logging.warning(f'No selected file')
				return redirect(request.url)
		
			
			bytearr = bin_file.read()
			logging.debug(f'filesize: {len(bytearr)}')
		
			result = pool.apply_async(async_file_processing, args=(int(file_id), bytearr))
			logging.info(f'apply_async result:{result}')
				
		#async_processing(request)
		return "ok"



#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        # Set debug param if DEBUG or INFO severity is set
        is_debugging = logging.getLogger().getEffectiveLevel() < logging.WARNING
        flask_app.run(host = REQUEST_HOST_URL, port = REQUEST_HOST_PORT, threaded = True, debug = is_debugging)
        
    except Exception as ex:
        logging.exception(ex)


