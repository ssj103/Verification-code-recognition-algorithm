# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:26:57 2019

@author: s19776
"""

import os
import tensorflow as tf 
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt  


tf.reset_default_graph()

## Different number of characters
CHAR_SET_LEN = 10
## image height
IMAGE_HEIGHT = 60 
## image width
IMAGE_WIDTH = 160  
## batch
BATCH_SIZE = 1

## tfrecord file save path 
TFRECORD_FILE = "./captcha/test.tfrecords"

## placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  

## Read data from tfrecord
def read_and_decode(filename):
    ## Generate a queue based on the file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    ## Return file name and file
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    ## Get image data
    image = tf.decode_raw(features['image'], tf.uint8)
    ## Grayscale image without preprocessing
    image_raw = tf.reshape(image, [224, 224])
    ## tf.train.shuffle_batch must ensure shape
    image = tf.reshape(image, [224, 224])
    ## Image preprocessing
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    ## get label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


## Get image data and tags
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

## Use shuffle_batch to randomly mess up
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

## Defining the network structure
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])

    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)
    
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])  
    predict0 = tf.argmax(predict0, 1)  

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])  
    predict1 = tf.argmax(predict1, 1)  

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])  
    predict2 = tf.argmax(predict2, 1)  

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])  
    predict3 = tf.argmax(predict3, 1)  


    sess.run(tf.global_variables_initializer())
    ## Loading trained models
    saver = tf.train.Saver()   
    saver.restore(sess, tf.train.latest_checkpoint('./captcha/models/'))  

    ## Create a coordinator, manage threads
    coord = tf.train.Coordinator()
    ## Start QueueRunner, the file name queue has been queued
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        ## Get a batch of data and tags
        b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                    image_raw_batch, 
                                                                    label_batch0, 
                                                                    label_batch1, 
                                                                    label_batch2, 
                                                                    label_batch3])
        ## display image
        img=Image.fromarray(b_image_raw[0],'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        ## print label 
        print('label:',b_label0, b_label1 ,b_label2 ,b_label3)
        ## predict
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x: b_image})
        ## print predict value 
        print('predict:',label0,label1,label2,label3) 
                
    ## close thread
    coord.request_stop()
    coord.join(threads)


''''
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt 
 
CAPTCHA_LEN = 4
 
MODEL_SAVE_PATH = './captcha/models/'
TEST_IMAGE_PATH = './captcha/test_images/'
 
def get_image_data_and_name(fileName, filePath=TEST_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #转为灰度图
    img = img.convert("L")       
    image_array = np.array(img)    
    image_data = image_array.flatten()/255
    image_name = fileName[0:CAPTCHA_LEN]
    return image_data, image_name
 
def digitalStr2Array(digitalStr):
    digitalList = []
    for c in digitalStr:
        digitalList.append(ord(c) - ord('0'))
    return np.array(digitalList)
 
def model_test():
    nameList = []
    for pathName in os.listdir(TEST_IMAGE_PATH):
        nameList.append(pathName.split('/')[-1])
    totalNumber = len(nameList)
    #加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + "crack_captcha.model-4500.meta")
    graph = tf.get_default_graph()
    #从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        count = 0
        for fileName in nameList:
            img_data, img_name = get_image_data_and_name(fileName, TEST_IMAGE_PATH)
            predict = sess.run(predict_max_idx, feed_dict={input_holder:[img_data], keep_prob_holder : 1.0})            
            filePathName = TEST_IMAGE_PATH + fileName
            print(filePathName)
            img = Image.open(filePathName)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            predictValue = np.squeeze(predict)
            rightValue = digitalStr2Array(img_name)
            if np.array_equal(predictValue, rightValue):
                result = '正确'
                count += 1
            else: 
                result = '错误'            
            print('label:{}, predict:{}，test_result：{}'.format(rightValue, predictValue, result))
            print('\n')
            
        print('accuracy: %.2f%%(%d/%d)' % (count*100/totalNumber, count, totalNumber))
 
if __name__ == '__main__':
    model_test()
'''