# -*- coding: utf-8 -*-

#import os
import tensorflow as tf 
#from PIL import Image
from nets import nets_factory
#import numpy as np
#import matplotlib.pyplot as plt  

## Clear the stack of the default graph and set the global map to the default map
tf.reset_default_graph()

# char num
CHAR_SET_LEN = 10
# height
IMAGE_HEIGHT = 60 
# width
IMAGE_WIDTH = 160  
# batch
BATCH_SIZE = 1

# tfrecord file path
#test one image with .tfrecord format
TFRECORD_FILE = "./captcha/test.tfrecords1" 

#.pd model file path
saved_model_dir = './export/1/' 

signature_key = 'predict_images'
input_key = 'images'
output_key0 = 'result0'
output_key1 = 'result1'
output_key2 = 'result2'
output_key3 = 'result3'

#read test iamge with .tfrecord format
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['image'], tf.uint8)
    image_raw = tf.reshape(image, [224, 224])
    image = tf.reshape(image, [224, 224])

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3

# get image data and label
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training = False)

with tf.Session() as sess:

    meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], saved_model_dir) 
    signature = meta_graph_def.signature_def
  
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name0 = signature[signature_key].outputs[output_key0].name 
    y_tensor_name1 = signature[signature_key].outputs[output_key1].name
    y_tensor_name2 = signature[signature_key].outputs[output_key2].name
    y_tensor_name3 = signature[signature_key].outputs[output_key3].name 
  
    ## print input and output tensor
    print("x_tensor_name = ", x_tensor_name)
    print("y_tensor_name0 = ",y_tensor_name0)
    print("y_tensor_name1 = ",y_tensor_name1)
    print("y_tensor_name2 = ",y_tensor_name2)
    print("y_tensor_name3 = ",y_tensor_name3)
  
    x_input = sess.graph.get_tensor_by_name(x_tensor_name)     
    predict0 = sess.graph.get_tensor_by_name(y_tensor_name0) 
    predict1 = sess.graph.get_tensor_by_name(y_tensor_name1)
    predict2 = sess.graph.get_tensor_by_name(y_tensor_name2)
    predict3 = sess.graph.get_tensor_by_name(y_tensor_name3)  

    # initialization
    sess.run(tf.global_variables_initializer())
    # Loading trained models
    saver = tf.train.Saver()   
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    for i in range(1): #one image run one time
        # Get a batch of data and tags
        b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                    image_raw_batch, 
                                                                    label_batch0, 
                                                                    label_batch1, 
                                                                    label_batch2, 
                                                                    label_batch3])
        # show image
        #img=Image.fromarray(b_image_raw[0],'L')
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()
        
        # print label
        print('label:',b_label0, b_label1 ,b_label2 ,b_label3)
        # predict label
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x_input: b_image})
        # print predict value
        print('predict:',Counter(label0).most_common(1),Counter(label1).most_common(1),Counter(label2).most_common(1),Counter(label3).most_common(1)) 
        print('predict:',label0,label1,label2,label3) 
