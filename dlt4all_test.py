# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
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
from poems.model import rnn_model
from poems.resnet import *
from poems.poems import process_poems
import numpy as np
from ssq_data import *
start_token = 'B'
end_token = 'E'
model_dir = './model4all/'
corpus_file = './data/poems.txt'


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem():
    batch_size = 1
    print('## loading model from %s' % model_dir)
    input_data = tf.compat.v1.placeholder(tf.float32, [1, 1,7,1])
    logits = inference(input_data, 1, reuse=False,output_num=128)

    # print(tf.shape(input_data))
    output_targets = tf.compat.v1.placeholder(tf.int32, [1, None])
    end_points = rnn_model(model='lstm', input_data=logits, output_data=output_targets, vocab_size=35+12,
                           output_num=7,
                           rnn_size=128, num_layers=7, batch_size=1, learning_rate=0.01)

    # input_data = tf.placeholder(tf.int32, [batch_size, None])
    #
    # end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=33,
    #                        rnn_size=128, num_layers=7, batch_size=1, learning_rate=0.01)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./dlt_model/')
        saver.restore(sess, checkpoint)
        # saver.restore(sess, "E:/workplace/tensorflow_poems-master/model4red/poems-150000")
        ssqdata = get_dlt_data(random_order=False,use_resnet=True)
        # x = np.array([list(map(word_int_map.get, start_token))])
        x=[ssqdata[len(ssqdata)-1]]
        print("input: %s"%(x+np.asarray([[[[1],[1],[1],[1],[1],[-34],[-34]]]])))
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem_=np.argmax(np.array(predict),axis=1)
        sorted_result = np.argsort(np.array(predict), axis=1)
        results=poem_+np.asarray([1,1,1,1,1,-34,-34])
        print(sorted_result)
        print("output: %s"%results)
        return poem_

def gen_blue():
    batch_size = 1
    print('## loading model from %s' % model_dir)
    input_data = tf.compat.v1.placeholder(tf.float32, [1, 1,7,1])
    logits = inference(input_data, 10, reuse=False,output_num=128)

    # print(tf.shape(input_data))
    output_targets = tf.compat.v1.placeholder(tf.int32, [1, None])
    end_points = rnn_model(model='lstm', input_data=logits, output_data=output_targets, vocab_size=33,
                           output_num=1,
                           rnn_size=128, num_layers=3, batch_size=1, learning_rate=0.01)

    # input_data = tf.placeholder(tf.int32, [batch_size, None])
    #
    # end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=33,
    #                        rnn_size=128, num_layers=7, batch_size=1, learning_rate=0.01)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./model4all/')
        saver.restore(sess, checkpoint)
        # saver.restore(sess, "E:/workplace/tensorflow_poems-master/model4blue/poems-84201")
        ssqdata = get_exl_data(random_order=False,use_resnet=True)
        # x = np.array([list(map(word_int_map.get, start_token))])
        x=[ssqdata[len(ssqdata)-1]]
        print("input: %s"%(x))
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem_=np.argmax(np.array(predict),axis=1)
        sorted_result=np.argsort(np.array(predict),axis=1)
        results=poem_+np.ones(1)
        print(results)
        print(sorted_result)
        return poem_


if __name__ == '__main__':
    # begin_char = input('## please input the first character:')
    # poem=gen_blue()#13
    poem = gen_poem()
    # pretty_print_poem(poem_=poem)
