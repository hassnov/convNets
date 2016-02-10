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
#import cv2

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

def build_graph_orig(data, keep_prob):
    data_shape = data.get_shape().as_list();
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",data,[5, 5, NUM_CHANNELS, 64])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    conv1 = conv2d_layer("conv1",pool0,[5, 5, 64, 128])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')   
    conv2 = conv2d_layer("conv2",pool1,[5, 5, 128, 64])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", conv2, [fc0_inputdim, 512])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([512, NUM_LABELS])
    b_fc1 = bias_variable([NUM_LABELS])

    y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
     
    return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])

def train(total_loss, global_step,num_batches_per_epoch):
    
    MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    LR_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INIT_RATE = 0.1  # Initial learning rate.
    
    # Variables that affect learning rate.
    decay_steps = int(num_batches_per_epoch)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    # Generate moving averages of all losses and associated summaries.
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summmary to all individual losses, the total loss, and averaged ones
    for l in losses + [total_loss]:
        # Rename each loss as '(raw)' and rename the moving average loss as original
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l)) 
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op



def main():

    reader = HDF5Reader()
    reader.readFiles("train_files.txt")
    print "Total number of samples: ", reader.computeTotalNumberOfSamples()
    
    INIT_RATE = 0.001
    LR_DECAY_FACTOR = 0.1
    
    BATCH_SIZE = 100
    nr_epochs = 500
    batches_per_epoch = reader.computeTotalNumberOfSamples()/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    X,Y = reader.next_batch(BATCH_SIZE)
    

    #cv2.imwrite("sample0.png",training[0]*255)
    #cv2.imwrite("mask0.png",labels[0]*255)
    #cv2.waitKey()
    
    with tf.Graph().as_default() as graph:

        # Create input/output placeholder variables for the graph (to be filled manually with data)
        # Placeholders MUST be filled for each session.run()
        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder("float", Y.shape, name="in_y")
        
        
        #lr = tf.placeholder(tf.float32)
        # Build the graph that computes predictions and assert that network output is compatible
        
        output = build_graph_orig(net_x, 0.5)
        print 'output shape: ',output.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        assert (output.get_shape().as_list() == net_y.get_shape().as_list() )
        
        # Add l2 loss to loss collection and create final loss tensor as the sum of all losses
        #tf.add_to_collection('losses', tf.nn.l2_loss(tf.div((output-net_y),BATCH_SIZE),name='l2_loss'))
        #total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        global_step = tf.Variable(0, trainable=False)
        
        #compute cross entropy loss
        #tf.reshape(net_y, [net_y.get_shape().as_list()[1],net_y.get_shape().as_list()[0]])
        
        
        logy = tf.log(output + 1e-10)
        mult = net_y*logy
        cross_entropy = -tf.reduce_sum(mult)
        #cross_entropy = -tf.reduce_sum(output*tf.log(net_y))
        
        #tf.reshape(net_y, [net_y.get_shape().as_list()[1],net_y.get_shape().as_list()[0]])
        
        decay_steps = int(batches_per_epoch)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
        tf.scalar_summary('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.GradientDescentOptimizer(0.000001)
        train_op = opt.minimize(cross_entropy, global_step=global_step)
        
        
    
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(net_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        #train_op = train(total_loss, global_step, batches_per_epoch)
    
        #opt = tf.train.AdamOptimizer(0.001)
        #opt = tf.train.GradientDescentOptimizer(0.000001)
        #train_op = opt.apply_gradients(opt.compute_gradients(total_loss), global_step=global_step)

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('.', graph_def=sess.graph_def)
                
        saver.restore(sess, 'model.ckpt')   # Load previously trained weights
                
        reader.reset()
        for epoch in range(nr_epochs):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    X,Y= reader.next_batch(BATCH_SIZE)
                
                    start_time = time.time()
                    #res, absy_, logy_, mult_,err = sess.run([output,absy, logy, mult, cross_entropy], feed_dict={net_x:X, net_y: Y})
                    _, error,summary = sess.run([train_op, cross_entropy, summary_op], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Batch:", batch ,"  Loss: ", error, "   Duration (sec): ", duration

                    summary_writer.add_summary(summary, batch)
                    
                    #Xv, Yv = reader.next_batch_val(BATCH_SIZE)
                    #acc = sess.run(accuracy, feed_dict={net_x:Xv, net_y: Yv})
                    #print "Batch:", batch ,"  Loss: ", error, " Test accuracy: ", acc, "   Duration (sec): ", duration
                    # Save the model checkpoint periodically.
                    if batch % 100 == 0:
                        save_path = saver.save(sess, "model.ckpt")
                        print "Model saved in file: ",save_path




    print 'done'
if __name__ == "__main__":
    main()


