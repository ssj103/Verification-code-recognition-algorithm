# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:37:31 2019

@author: s19776
"""

import tensorflow as tf
import os
import random
#import math
import sys
from PIL import Image
import numpy as np
 
## test data number
_NUM_TEST = 500
 
_RANDOM_SEED = 0
 
## dataset path
DATASET_DIR = "./captcha/images_9550_50/"
 
## tfrecord file exist path 
TFRECORD_DIR ="./captcha/"

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True
 
## get all verification images
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        ## get file path
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames
 
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
 
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
 
def image_to_tfexample(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image': bytes_feature(image_data),
      'label0': int64_feature(label0),
      'label1': int64_feature(label1),
      'label2': int64_feature(label2),
      'label3': int64_feature(label3),
    }))
 
## chage data to tfrecord
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']
 
    with tf.Session() as sess:
        ## define tfrecord file path and name
        output_filename = os.path.join(TFRECORD_DIR,split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()
                    
                    image_data = Image.open(filename)  
                    image_data = image_data.resize((224, 224))
                    image_data = np.array(image_data.convert('L'))
                    ## convert image to bytes
                    image_data = image_data.tobytes()              
 
                    ## get label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                                            
                    ## generate protocol data type
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                    
                except IOError as e:
                    print('Could not read:',filename)
                    print('Error:',e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()
 
if _dataset_exists(TFRECORD_DIR):
    print('tfcecord file is also exist')
else:
    photo_filenames = _get_filenames_and_classes(DATASET_DIR)
 
    ## Divide the data into training and test sets and shuffle
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames[:_NUM_TEST]
    #testing_filenames = photo_filenames[:_NUM_TEST]
 
    ## data convert
    _convert_dataset('train', training_filenames, DATASET_DIR)
    _convert_dataset('test', testing_filenames, DATASET_DIR)
    print('Done, generate tfcecord files')
	
	
	
