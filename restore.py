#!/usr/bin/env python

import h5py
#import os
#import re
#import sys
#from datetime import datetime
#import os.path
import time
import math
import numpy
#import tensorflow.python.platform
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

NUM_LABELS = 16000

class HDF5Reader:
  
    file_list = None
    curr_file = None
    curr_file_index = 0
    curr_data = None
    curr_label = None
    curr_index = 0
    curr_vdata = None
    curr_vlabel = None
    
    def readFiles(self,filename):
        self.file_list = [line.strip() for line in open(filename)]
        assert len(self.file_list)>0
        
    def __loadHDF5(self,filename):
        self.curr_file = h5py.File(self.file_list[0], 'r')
        self.curr_data = self.curr_file['/data']
        self.curr_label = self.curr_file['/label']
        self.curr_vdata = self.curr_file['/vdata']
        self.curr_vlabel = self.curr_file['/vlabel']
        self.curr_index = 0
        self.curr_indexv = 0
        
    def reset(self):
        self.curr_file = None
        
    def computeTotalNumberOfSamples(self):
        total = 0
        for f in self.file_list:
            total += h5py.File(f,'r')['/data'].shape[1]
        return total

    def get_labels(self, y1, NUM_LABELS):
        l = len(y1)
        y = [[]] * l
        for i in range(l):
            y[i] = self.get_label_vector(y1[i], NUM_LABELS)
        y = numpy.array(y);
        return y
    
    def get_label_vector(self, i, NUM_LABELS):
        label = [0] * NUM_LABELS
        label[int(i-1)] = 1
        return label
    
    def next_batch_val(self, batchSize):
        assert len(self.file_list)>0
        if self.curr_file == None:
            self.curr_file_index = 0
            self.__loadHDF5(self.file_list[self.curr_file_index])
        
        X = self.curr_vdata[:,self.curr_indexv:self.curr_indexv+batchSize]
        Y = self.get_labels(numpy.transpose(self.curr_vlabel[:, self.curr_indexv:self.curr_indexv+batchSize]),NUM_LABELS)

        assert X.shape[1] == Y.shape[0]
        
        if X.shape[1] == batchSize:
            self.curr_indexv += batchSize
        else:
            #self.curr_file_index = (self.curr_file_index+1) % len(self.file_list)
            #self.__loadHDF5(self.file_list[self.curr_file_index])
            self.curr_indexv = 0
            samples_missing = batchSize - X.shape[1]
            if(X.shape[1] == 0):
                X = self.curr_vdata[:,self.curr_indexv:self.curr_indexv+batchSize]
                Y = self.get_labels(numpy.transpose(self.curr_vlabel[:, self.curr_indexv:self.curr_indexv+batchSize]),NUM_LABELS)
            else:               
                X = numpy.hstack([X,self.curr_vdata[:, 0:samples_missing]])
                Y = numpy.vstack([Y,self.get_labels(numpy.transpose(self.curr_vlabel[:, 0:samples_missing]),NUM_LABELS)])
            self.curr_indexv = samples_missing    
        X = numpy.transpose(X)
        X[ X<0 ] = 0
        #Y = nympy.transpose(Y)
        X = numpy.reshape(X, (X.shape[0], 32,32,3))
        #X_val = numpy.reshape(X_val, (X_val.shape[0], 32,32,3))
        return X,Y
        
    def next_batch(self,batchSize):
        assert len(self.file_list)>0
        if self.curr_file == None:
            self.curr_file_index = 0
            self.__loadHDF5(self.file_list[self.curr_file_index])
        
        X = self.curr_data[:,self.curr_index:self.curr_index+batchSize]
        #Y = self.curr_label[self.curr_index:self.curr_index+batchSize]
        Y = self.get_labels(numpy.transpose(self.curr_label[:, self.curr_index:self.curr_index+batchSize]),NUM_LABELS)

        assert X.shape[1] == Y.shape[0]
        
        if X.shape[1] == batchSize:
            self.curr_index += batchSize
        else:
            self.curr_file_index = (self.curr_file_index+1) % len(self.file_list)
            self.__loadHDF5(self.file_list[self.curr_file_index])
            samples_missing = batchSize - X.shape[1]
            if(X.shape[1] == 0):
                X = self.curr_data[:,self.curr_index:self.curr_index+batchSize]
                Y = self.get_labels(numpy.transpose(self.curr_label[:, self.curr_index:self.curr_index+batchSize]),NUM_LABELS)
            else:
                X = numpy.hstack([X,self.curr_data[:, 0:samples_missing]])
                Y = numpy.vstack([Y,self.get_labels(numpy.transpose(self.curr_label[:, 0:samples_missing]), NUM_LABELS)])
            self.curr_index = samples_missing    
        X = numpy.transpose(X)
        X[ X<0 ] = 0
        #Y = nympy.transpose(Y)
        """X_val = X[batchSize -10:batchSize, :]
        Y_val = Y[batchSize -10:batchSize, :]
        X = X[0:batchSize -10, :]
        Y = Y[0:batchSize -10, :]
        print 'xshape=' , X.shape
        print 'yshape=' , Y.shape
        print 'xvalshape=' , X_val.shape
        print 'yvalshape=' , Y_val.shape"""
        X = numpy.reshape(X, (X.shape[0], 32,32,3))
        #X_val = numpy.reshape(X_val, (X_val.shape[0], 32,32,3))
        return X,Y
            

def activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def xavier_conv_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/(shape[0]*shape[1]* shape[2])))

def xavier_fc_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/shape[0]))

def conv2d_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        # Variables created here will be named "name/weights", "name/biases".
        weights = tf.Variable(xavier_conv_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[3]]), name="biases")
        conv = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')    
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        layer = tf.nn.relu(bias, name=name)
    activation_summary(layer)
    return layer

def fc_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        weights = tf.Variable(xavier_fc_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[1]]), name="biases")
        input_flat = tf.reshape(input_data, [-1,shape[0]])
        layer = tf.nn.relu(tf.nn.xw_plus_b(input_flat, weights, biases, name=name))
    activation_summary(layer)
    return layer

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def build_graph(data, keep_prob):
    data_shape = data.get_shape().as_list();
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",data,[5, 5, NUM_CHANNELS, 64])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    conv1 = conv2d_layer("conv1",pool0,[5, 5, 64, 64])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([128, NUM_LABELS])
    b_fc1 = bias_variable([NUM_LABELS])

    y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
     
    return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])




def main():

    reader = HDF5Reader()
    reader.readFiles("train_files.txt")
    print "Total number of samples: ", reader.computeTotalNumberOfSamples()

    BATCH_SIZE = 100
    nr_epochs = 500
    batches_per_epoch = reader.computeTotalNumberOfSamples()/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    X,Y = reader.next_batch(BATCH_SIZE)
    
    
    #imsize = 32
    #imgs = numpy.zeros((2,imsize,imsize,3))
    #img = cv2.imread('Mikolajczyk/graffiti/img1.ppm')[0:300, 0:300]
    #img = cv2.resize(img, (imsize,imsize), interpolation = cv2.INTER_CUBIC)
    #imgs[0] = img
    #img = cv2.imread('Mikolajczyk/graffiti/img1.ppm')[0:110, 0:110]
    #img = cv2.resize(img, (imsize,imsize), interpolation = cv2.INTER_CUBIC)
    #imgs[1] = img
    #imgs = imgs/255
    
    
    # Create input/output placeholder variables for the graph (to be filled manually with data)
    # Placeholders MUST be filled for each session.run()
    #net_x = tf.placeholder("float", imgs.shape, name="in_x")
    #net_y = tf.placeholder("float", [1, NUM_LABELS], name="in_y")
    net_x = tf.placeholder("float", X.shape, name="in_x")
    net_y = tf.placeholder("float", Y.shape, name="in_y")
    
    
    
    # Build the graph that computes predictions and assert that network output is compatible
    data_shape = net_x.get_shape().as_list();
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",net_x,[5, 5, NUM_CHANNELS, 64])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    pool_shape = pool0.get_shape().as_list()
    r = pool_shape[1]/4
    pool0_44 = tf.nn.max_pool(conv0, ksize=[1, r, r, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0_44')
    conv1 = conv2d_layer("conv1",pool0,[5, 5, 64, 128])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    pool_shape = pool1.get_shape().as_list()
    r = pool_shape[1]/4
    pool1_44 = tf.nn.max_pool(conv0, ksize=[1, r, r, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0_44')   
    conv2 = conv2d_layer("conv2",pool1,[5, 5, 128, 64])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", conv2, [fc0_inputdim, 512])   
    
    fc0_drop = tf.nn.dropout(fc0, 0.5)
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([512, NUM_LABELS])
    b_fc1 = bias_variable([NUM_LABELS])

    #y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
    output=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(net_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Create initialization "op" and run it with our session 
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    
    # Create a saver and a summary op based on the tf-collection
    saver = tf.train.Saver(tf.all_variables())
    
            
    saver.restore(sess, 'model_1600_32.ckpt')   # Load previously trained weights
    acc, p0,p044, p1,p144,f0 = sess.run([accuracy, pool0,pool0_44, pool1, pool1_44, fc0], feed_dict={net_x:X, net_y:Y})
    
    print 'distance f0', numpy.linalg.norm(f0[1]-f0[0])
    print 'distance p0', numpy.linalg.norm(p0[1]-p0[0])
    print 'distance p044', numpy.linalg.norm(p044[1]-p044[0])
    print 'distance p1', numpy.linalg.norm(p1[1]-p1[0])
    print 'distance p144', numpy.linalg.norm(p144[1]-p144[0])
    print 'accuracy: ', acc
    
    
    
    """reader.reset()
    for epoch in range(nr_epochs):
            print "Starting epoch ", epoch
            for batch in range(batches_per_epoch):
                X,Y= reader.next_batch(BATCH_SIZE)
            
                start_time = time.time()
                #res, absy_, logy_, mult_,err = sess.run([output,absy, logy, mult, cross_entropy], feed_dict={net_x:X, net_y: Y})
                error,summary = sess.run([cross_entropy, summary_op], feed_dict={net_x:X, net_y: Y})
                duration = time.time() - start_time
                print "Batch:", batch ,"  Loss: ", error, "   Duration (sec): ", duration

                summary_writer.add_summary(summary, batch)
                
                Xv, Yv = reader.next_batch_val(BATCH_SIZE)
                acc = sess.run(accuracy, feed_dict={net_x:Xv, net_y: Yv})
                print 'test accuracy = ', acc
                # Save the model checkpoint periodically.
                if batch % 100 == 0:
                    save_path = saver.save(sess, "model.ckpt")
                    print "Model saved in file: ",save_path"""



    print 'done'
if __name__ == "__main__":
    main()


