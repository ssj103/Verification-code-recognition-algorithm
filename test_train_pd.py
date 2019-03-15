# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:13:18 2019

@author: s19776
"""

import os
import tensorflow as tf 
#from PIL import Image
from nets import nets_factory
import shutil
#import numpy as np

tf.app.flags.DEFINE_string('export_dir', 'export', 'Working directory.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('model_dir', 'models', 'model directory.')
FLAGS = tf.app.flags.FLAGS

## reset_default_graph 
tf.reset_default_graph()

## number length
CHAR_SET_LEN = 10
## image height
IMAGE_HEIGHT = 60 
## image width
IMAGE_WIDTH = 160  
## batch size
BATCH_SIZE = 25
## tfrecord file path
TFRECORD_FILE = "./captcha/train.tfrecords"

## placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  
y0 = tf.placeholder(tf.float32, [None]) 
y1 = tf.placeholder(tf.float32, [None]) 
y2 = tf.placeholder(tf.float32, [None]) 
y3 = tf.placeholder(tf.float32, [None])

## learning rate
lr = tf.Variable(0.003, dtype=tf.float32)

# from tfrecord read data
def read_and_decode(filename):
    ## generate a queue based on the file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    ## return file name and file
    _, serialized_example = reader.read(filename_queue) 
    
    ## tf.FixedLenFeature: analyze fixed-length input features feature-related configuration
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    ## get image data
    image = tf.decode_raw(features['image'], tf.uint8)
    ## tf.train.shuffle_batch must ensure shape
    image = tf.reshape(image, [224, 224])
    ## image preprocessing 
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    
    ## get label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, label0, label1, label2, label3

## get data and label
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

## use shuffle_batch randomly disrupted data 
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

## define network structure 
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes = CHAR_SET_LEN,
    weight_decay = 0.0005,
    is_training = True)
 
## GPU resource use limit
config = tf.ConfigProto(allow_soft_placement=True)
## GPU limit usage 80%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
## gpu resource
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.Session() as sess:
     
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])   
    
    # output value
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)
	
    ## label convert one-hot format
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    
    ## compute loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0,labels=one_hot_labels0)) 
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1,labels=one_hot_labels1)) 
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2,labels=one_hot_labels2)) 
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3,labels=one_hot_labels3)) 
	
    ## compute total loss
    total_loss = (loss0+loss1+loss2+loss3)/4.0
	
    ## optimize total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss) 
    
    ## compute accuracy
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
    
    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
    
    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
    
    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32)) 
    
    ## save model -cpkt
    saver = tf.train.Saver()
    ## initial
    sess.run(tf.global_variables_initializer())
	
	## .pd model save	
    #serialized_tf_example = tf.placeholder(tf.string)
    export_path_base = FLAGS.export_dir
    export_path = os.path.join(
		tf.compat.as_bytes(export_path_base),
		tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
  
	## check file if exist,if exist, then delete and build again
    if os.path.exists(export_path):
        shutil.rmtree(export_path)  		
    builder = tf.saved_model.builder.SavedModelBuilder(export_path) # step 1
    
    ## create Coordinator, management thread
    coord = tf.train.Coordinator()
    ## start QueueRunner, the file name queue has been queued
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10001):
        ## get batch data and label
        b_image, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        ## optimize model
        sess.run(optimizer, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3})  

        ## every 20 time compute loss and accuracy  
        if i % 20 == 0:  
            ## every 2000 time decrease learning rate
            if i%2000 == 0:
                sess.run(tf.assign(lr, lr/3))
            acc0,acc1,acc2,acc3,loss_ = sess.run([accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
                                                                                                                y0: b_label0,
                                                                                                                y1: b_label1,
                                                                                                                y2: b_label2,
                                                                                                                y3: b_label3})    
    
            learning_rate = sess.run(lr)
            print ("Iteration:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (i,loss_,acc0,acc1,acc2,acc3,learning_rate))
             
            ## save model 
            ## if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90: 
            if i==10000:
                
                # save cpkt model file
                saver.save(sess, "./models/crack_captcha.model", global_step=i)                
                ## restore cpkt model file
                #model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
                #saver.restore(sess, model_file)             
                
				## save .pd model file
                tensor_info_x  = tf.saved_model.utils.build_tensor_info(x)
                tensor_info_y0 = tf.saved_model.utils.build_tensor_info(y0)
                tensor_info_y1 = tf.saved_model.utils.build_tensor_info(y1)
                tensor_info_y2 = tf.saved_model.utils.build_tensor_info(y2)
                tensor_info_y3 = tf.saved_model.utils.build_tensor_info(y3)

				## images and label is tensor name.
                prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
										inputs={'images': tensor_info_x},
										outputs={'label0': tensor_info_y0, 'label1':tensor_info_y1, 'label2':tensor_info_y2, 'label3':tensor_info_y3},
										method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 
										
				## Import graph and variables information 
				## signature_def_map: a dict, save model need parameters. 			
                builder.add_meta_graph_and_variables(sess, ["serve"], 
													signature_def_map={'predict_images':prediction_signature},
													main_op = tf.tables_initializer()) # step 2
                
                builder.save() #step 3
                print('Done exporting!')
                
                break 	
                
    ## Notify other threads to close
    coord.request_stop()
    ## This function can only be returned after all other threads are closed.
    coord.join(threads)
    
    
    
    
    
    
    