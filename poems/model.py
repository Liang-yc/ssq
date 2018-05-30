# -*- coding: utf-8 -*-
# file: model.py
# author: JinTian
# time: 07/03/2017 3:07 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import tensorflow as tf
import numpy as np


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              input_num=128,output_num=1,learning_rate=0.01,use_cnn=True):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}


    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
        cell = cell_fun(rnn_size)
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
        cell = cell_fun(rnn_size)
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
        cell = cell_fun(rnn_size, state_is_tuple=True)
        
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)
    if use_cnn:
        with tf.name_scope('fc1'):
            fc1_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([input_num, 128],
                                    # mean=1.0,
                                    stddev=0.1,
                                    dtype=tf.float32))
            fc1_biases = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32))
            fc1 = tf.nn.relu(tf.matmul(tf.to_float(input_data), fc1_weights) + fc1_biases)
        with tf.name_scope('fc2'):
            fc2_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([128, 256],
                                    # mean=1.0,
                                    stddev=0.1,
                                    dtype=tf.float32))
            fc2_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
            fc2 = tf.nn.relu(tf.matmul(tf.to_float(fc1), fc2_weights) + fc2_biases)
        with tf.name_scope('fc3'):
            fc3_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([256, 512],
                                    # mean=1.0,
                                    stddev=0.1,
                                    dtype=tf.float32))
            fc3_biases = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32))
            fc3 = tf.nn.relu(tf.matmul(tf.to_float(fc2), fc3_weights) + fc3_biases)
        with tf.name_scope('fc4'):
            fc4_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([512, 128*output_num],
                                    # mean=1.0,
                                    stddev=0.1,
                                    dtype=tf.float32))
            fc4_biases = tf.Variable(tf.constant(1.0, shape=[128*output_num], dtype=tf.float32))
            fc4 = tf.nn.relu(tf.matmul(fc3, fc4_weights) + fc4_biases)


        # [batch_size, ?, rnn_size] = [64, ?, 128]
        embedded=tf.reshape(fc4,[batch_size,-1,128])
        outputs, last_state = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)
    else:
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
                [vocab_size + 1, rnn_size], -1.0, 1.0))
            inputs = tf.nn.embedding_lookup(embedding, input_data)
    # embedded = tf.reshape(input_data, [batch_size, -1, 128])
    # outputs, last_state = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)
    # outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases)+
                tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases)+
                tf.nn.l2_loss(fc4_weights) + tf.nn.l2_loss(fc4_biases))
        total_loss+=5e-4 * regularizers
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
        prediction = tf.nn.softmax(logits)
        end_points['prediction'] = prediction
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points

