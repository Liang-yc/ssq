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
import os
import numpy as np
import tensorflow as tf
from poems.model import rnn_model
from poems.resnet import *
from poems.poems import process_poems, generate_batch
from ssq_data import *
# for Windows10：OSError: raw write() returned invalid length 96 (should have been between 0 and 48)
import win_unicode_console
win_unicode_console.enable()
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 2214, 'batch size.')
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.compat.v1.app.flags.DEFINE_string('model_dir', os.path.abspath('./dlt_model'), 'model save path.')
tf.compat.v1.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems.')
tf.compat.v1.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.compat.v1.app.flags.DEFINE_integer('epochs', 500000, 'train how many epochs.')

FLAGS = tf.compat.v1.app.flags.FLAGS


def run_training():
    # if not os.path.exists(FLAGS.model_dir):
    #     os.makedirs(FLAGS.model_dir)
    #
    # poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    # ssqdata=get_exl_data(random_order=False,use_resnet=True)
    # # print(ssqdata[len(ssqdata)-1])
    # batches_inputs=ssqdata[0:(len(ssqdata)-1)]
    # ssqdata=get_exl_data(random_order=False,use_resnet=False)
    ssqdata=get_dlt_data(random_order=False,use_resnet=True)
    # print(ssqdata[len(ssqdata)-1])
    batches_inputs=ssqdata[0:(len(ssqdata)-1)]
    ssqdata=get_dlt_data(random_order=False,use_resnet=False)
    batches_outputs = ssqdata[1:(len(ssqdata))]
    FLAGS.batch_size=len(batches_inputs)
    # print(np.shape(batches_outputs))
    # data=batches_outputs[1:7]
    # print(len(data))
    del ssqdata
    input_data = tf.compat.v1.placeholder(tf.float32, [FLAGS.batch_size, 1,7,1])
    logits = inference(input_data, 1, reuse=False,output_num=128)

    # print(tf.shape(input_data))
    output_targets = tf.compat.v1.placeholder(tf.int32, [FLAGS.batch_size, None])
    end_points = rnn_model(model='lstm', input_data=logits, output_data=output_targets, vocab_size=35+12,output_num=7,
                           rnn_size=128, num_layers=7, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)
    # end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
    #     vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        # saver.restore(sess, "D:/tensorflow_poems-master/model4all/poems-207261")
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                # n_chunk = len(poems_vector) // FLAGS.batch_size
                # n_chunk = len(batches_inputs) // FLAGS.batch_size
                n_chunk=math.ceil(len(batches_inputs) / FLAGS.batch_size)
                for batch in range(n_chunk):
                    left=(batch+1)*FLAGS.batch_size-len(batches_inputs)
                    if left<0:
                        inputdata=batches_inputs[(batch*FLAGS.batch_size):((batch+1)*FLAGS.batch_size)]
                        outputdata=batches_outputs[(batch*FLAGS.batch_size):((batch+1)*FLAGS.batch_size)]
                    else:
                        # temp=batches_inputs[batch*FLAGS.batch_size:len(batches_inputs) ]
                        # temp.extend(batches_inputs[0:left])
                        inputdata=batches_inputs[len(batches_inputs)-FLAGS.batch_size:len(batches_inputs)]
                        # temp=batches_outputs[batch*FLAGS.batch_size:len(batches_inputs) ]
                        # temp.extend(batches_outputs[0:left])
                        outputdata=batches_outputs[len(batches_outputs)-FLAGS.batch_size:len(batches_outputs)]
                    # print(len(inputdata))
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: inputdata, output_targets: outputdata})
                    # ], feed_dict={input_data: batches_inputs, output_targets: batches_outputs})
                    n += 1
                if epoch % 1000 == 0:
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                if epoch % 50000 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))
        saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.compat.v1.app.run()