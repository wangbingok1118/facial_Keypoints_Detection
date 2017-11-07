# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
import dataProcess
import config

def weight_variable(shape, sdev=0.1):
    """ randomly initialize a weight variable w/given shape """
    initial = tf.truncated_normal(shape, stddev=sdev)
    return tf.Variable(initial)
def bias_variable(shape, constant=0.1):
    """ initialize bias to constant vector w/given shape """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)

cnnModelParam={
    'conv1_weight':weight_variable(shape=[config.filter_size,config.filter_size,config.image_channels,16]),
    'conv1_bias':bias_variable(shape=[16]),
    'conv2_weight':weight_variable(shape=[config.filter_size,config.filter_size,16,32]),
    'conv2_bias':bias_variable(shape=[32]),
    'conv3_weight':weight_variable(shape=[config.filter_size,config.filter_size,32,32]),
    'conv3_bias':bias_variable(shape=[32]),
    'fc1_weight':weight_variable(shape=[(config.image_size//8)*(config.image_size//8)*32,1024]),
    'fc1_bias':bias_variable(shape=[1024]),
    'fc2_weight':weight_variable(shape=[1024,512]),
    'fc2_bias':bias_variable(shape=[512]),
    'fc3_weight':weight_variable(shape=[512,config.label_numbers]),
    'fc3_bias':bias_variable(shape=[config.label_numbers])
}




def cnnModel(data=None,train=False):
    """

    :param data: input data
    :param train:
    :return:
    """
    global cnnModelParam
    conv1_weight = cnnModelParam.get('conv1_weight')
    conv1_bias = cnnModelParam.get('conv1_bias')
    conv1 = tf.nn.conv2d(input=data,filter=conv1_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv1_add_bias = tf.nn.bias_add(value=conv1,bias=conv1_bias,data_format='NHWC')
    conv1_relu = tf.nn.relu(conv1_add_bias)
    pool1 = tf.nn.max_pool(value=conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    conv2_weight = cnnModelParam.get('conv2_weight')
    conv2_bias = cnnModelParam.get('conv2_bias')
    conv2 = tf.nn.conv2d(input=pool1,filter=conv2_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv2_add_bias = tf.nn.bias_add(value=conv2,bias=conv2_bias,data_format='NHWC')
    conv2_relu = tf.nn.relu(conv2_add_bias)
    pool2 = tf.nn.max_pool(value=conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    conv3_weight = cnnModelParam.get('conv3_weight')
    conv3_bias = cnnModelParam.get('conv3_bias')
    conv3 = tf.nn.conv2d(input=pool2,filter=conv3_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv3_add_bias = tf.nn.bias_add(value=conv3,bias=conv3_bias,data_format='NHWC')
    conv3_relu = tf.nn.relu(conv3_add_bias)
    pool3 = tf.nn.max_pool(value=conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool3_shape_list = pool3.get_shape().as_list()
    reshaped_pool3 = tf.reshape(pool3,shape=[pool3_shape_list[0],pool3_shape_list[1]*pool3_shape_list[2]*pool3_shape_list[3]])

    fc1_weight = cnnModelParam.get('fc1_weight')
    fc1_bias = cnnModelParam.get('fc1_bias')
    fc1 = tf.matmul(reshaped_pool3,fc1_weight)
    fc1_add_bias = tf.add(fc1,fc1_bias)
    fc1_relu = tf.nn.relu(fc1_add_bias)
    if train:
        fc1_relu = tf.nn.dropout(fc1_relu,keep_prob=config.dropout_rate)

    fc2_weight = cnnModelParam.get('fc2_weight')
    fc2_bias = cnnModelParam.get('fc2_bias')
    fc2 = tf.matmul(fc1_relu,fc2_weight)
    fc2_add_bias = tf.add(fc2,fc2_bias)
    fc2_relu = tf.nn.relu(fc2_add_bias)
    if train:
        fc2_relu = tf.nn.dropout(fc2_relu,keep_prob=config.dropout_rate)

    fc3_weight = cnnModelParam.get('fc3_weight')
    fc3_bias = cnnModelParam.get('fc3_bias')
    fc3 = tf.add(tf.matmul(fc2_relu,fc3_weight),fc3_bias)
    return fc3

    pass


def cnnModel_old(data=None,train=False):
    """

    :param data: input data
    :param train:
    :return:
    """
    conv1_weight = weight_variable(shape=[config.filter_size,config.filter_size,config.image_channels,16])
    conv1_bias = bias_variable(shape=[16])
    conv1 = tf.nn.conv2d(input=data,filter=conv1_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv1_add_bias = tf.nn.bias_add(value=conv1,bias=conv1_bias,data_format='NHWC')
    conv1_relu = tf.nn.relu(conv1_add_bias)
    pool1 = tf.nn.max_pool(value=conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    conv2_weight = weight_variable(shape=[config.filter_size,config.filter_size,16,32])
    conv2_bias = bias_variable(shape=[32])
    conv2 = tf.nn.conv2d(input=pool1,filter=conv2_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv2_add_bias = tf.nn.bias_add(value=conv2,bias=conv2_bias,data_format='NHWC')
    conv2_relu = tf.nn.relu(conv2_add_bias)
    pool2 = tf.nn.max_pool(value=conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    conv3_weight = weight_variable(shape=[config.filter_size,config.filter_size,32,32])
    conv3_bias = bias_variable(shape=[32])
    conv3 = tf.nn.conv2d(input=pool2,filter=conv3_weight,strides=[1,1,1,1],padding='SAME',data_format='NHWC')
    conv3_add_bias = tf.nn.bias_add(value=conv3,bias=conv3_bias,data_format='NHWC')
    conv3_relu = tf.nn.relu(conv3_add_bias)
    pool3 = tf.nn.max_pool(value=conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool3_shape_list = pool3.get_shape().as_list()
    reshaped_pool3 = tf.reshape(pool3,shape=[pool3_shape_list[0],pool3_shape_list[1]*pool3_shape_list[2]*pool3_shape_list[3]])

    fc1_weight = weight_variable(shape=[pool3_shape_list[1]*pool3_shape_list[2]*pool3_shape_list[3],1024])
    fc1_bias = bias_variable(shape=[1024])
    fc1 = tf.matmul(reshaped_pool3,fc1_weight)
    fc1_add_bias = tf.add(fc1,fc1_bias)
    fc1_relu = tf.nn.relu(fc1_add_bias)
    if train:
        fc1_relu = tf.nn.dropout(fc1_relu,keep_prob=config.dropout_rate)

    fc2_weight = weight_variable(shape=[1024,512])
    fc2_bias = bias_variable(shape=[512])
    fc2 = tf.matmul(fc1_relu,fc2_weight)
    fc2_add_bias = tf.add(fc2,fc2_bias)
    fc2_relu = tf.nn.relu(fc2_add_bias)
    if train:
        fc2_relu = tf.nn.dropout(fc2_relu,keep_prob=config.dropout_rate)

    fc3_weight = weight_variable(shape=[512,config.label_numbers])
    fc3_bias = bias_variable(shape=[config.label_numbers])
    fc3 = tf.add(tf.matmul(fc2_relu,fc3_weight),fc3_bias)
    return fc3

    pass


def error_measure(predictions, labels):
    """ calculate sum squared error of predictions """
    return np.sum(np.power(predictions - labels, 2)) / (2.0 * predictions.shape[0])


def eval_model_batchs(data=None,sess=None,eval_predict_op=None,eval_node_placehold=None): # debug ***
    eval_x_node = eval_node_placehold
    eval_prediction = eval_predict_op
    # evaluate model use validation data
    validation_data_size = data.shape[0]

    if validation_data_size < config.mini_batch_size:
        raise ValueError("batch size for evals larger than dataset size: %d" % validation_data_size)
    validation_predict = None
    validation_predict = np.ndarray(shape=(validation_data_size, config.label_numbers), dtype=np.float32)

    for begin in range(0, validation_data_size, config.mini_batch_size):
        end = begin + config.mini_batch_size
        # get next batch from begin index to end index
        if end <= validation_data_size:
            validation_predict[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_x_node: data[begin:end, ...]})
        else:  # if end index is past the end of the data, fit input to batch size required for feed_dict
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_x_node: data[-config.mini_batch_size:, ...]})
            validation_predict[begin:, :] = batch_predictions[begin - validation_data_size:, :]
    return validation_predict
    pass


def trainModelAndSaveModel(trainDataX=None,trainDataY=None,valiDataX=None,valiDataY=None):
    print("constructing tensorflow cnn  model and train it :::")
    train_x_node = tf.placeholder(tf.float32, shape=(config.mini_batch_size, config.image_size, config.image_size, config.image_channels))
    train_y_node = tf.placeholder(tf.float32, shape=(config.mini_batch_size, config.label_numbers))

    eval_x_node = tf.placeholder(tf.float32, shape=(config.mini_batch_size, config.image_size, config.image_size, config.image_channels))

    train_predict = cnnModel(data=train_x_node, train=True)
    eval_predict = cnnModel(data=eval_x_node, train=False)
    train_predict_y_square = tf.square(train_predict-train_y_node)
    train_loss_sum = tf.reduce_sum(train_predict_y_square, axis=1)
    train_loss = tf.reduce_mean(train_loss_sum)
    train_step = tf.train.AdamOptimizer().minimize(train_loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("start train cnn Model")
        current_epoch = 0
        while current_epoch < config.train_epcohs:
            # new train ecoch , so shuffled
            shuffled_indices = np.arange(trainDataX.shape[0])
            np.random.shuffle(shuffled_indices)
            train_data = trainDataX[shuffled_indices]
            train_labels = trainDataY[shuffled_indices]
            train_loss_result=None
            for batch_index in range(trainDataX.shape[0] // config.mini_batch_size):
                offset = batch_index * config.mini_batch_size
                train_batch_data_x = train_data[offset:offset+config.mini_batch_size,...]
                train_batch_data_y = train_labels[offset:offset+config.mini_batch_size,...]
                train_feed_dict = {train_x_node: train_batch_data_x, train_y_node: train_batch_data_y}
                _,train_loss_result = sess.run([train_step,train_loss],feed_dict=train_feed_dict)
                pass
            validata_predict = eval_model_batchs(data=valiDataX,sess=sess,eval_predict_op=eval_predict,eval_node_placehold=eval_x_node)
            validate_error = error_measure(predictions=validata_predict,labels=valiDataY)
            print("Epoch %d, train loss %.8f, validate loss %.8f"%(current_epoch,train_loss_result,validate_error))
            current_epoch += 1
        pass
    pass



def train():
    # load train data
    print('loading training file')
    originalTrainX,originalTrainY = dataProcess.loadData(config.trainFilePath,test=False)
    print(originalTrainX.shape)
    # load test data
    print('loading test file')
    testX,_ = dataProcess.loadData(config.testFilePath,test=True) # test data only have x value
    trainX,validationX,trainY,validationY = sklearn.model_selection.train_test_split(originalTrainX,originalTrainY,test_size=config.validataionSize)
    trainX_shape = trainX.shape[0]
    validationX_shape = validationX.shape[0]
    testX_shape = testX.shape[0]
    print("trainX's shape is ",trainX_shape)
    print("validationX's shape is ",validationX_shape)
    print("testX's shape is ",testX_shape)
    trainModelAndSaveModel(trainDataX=trainX,trainDataY=trainY,valiDataX=validationX,valiDataY=validationY)
    pass
def test():
    pass



def main():
    train()
    pass


if __name__ == '__main__':
    main()