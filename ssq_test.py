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
from poems.poems import process_poems
import numpy as np
from ssq_data import *
start_token = 'B'
end_token = 'E'
model_dir = './model/'
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

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=33,
                           rnn_size=128, num_layers=7, batch_size=1, learning_rate=0.01)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)
        ssqdata = get_exl_data()
        # x = np.array([list(map(word_int_map.get, start_token))])
        x=[ssqdata[len(ssqdata)-1]]
        print("input: %s"%(x+np.ones(7)))
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem_=np.argmax(np.array(predict),axis=1)
        results=poem_+np.ones(7)
        print(results)
        return poem_



if __name__ == '__main__':
    # begin_char = input('## please input the first character:')
    poem = gen_poem()
    # pretty_print_poem(poem_=poem)