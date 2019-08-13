# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Tanel Peet. All Rights Reserved.
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
# ==============================================================================
#
# Modifications Copyright 2017-2018 Arm Inc. All Rights Reserved.
# Adapted from freeze.py to run quantized inference on train/val/test dataset on the
# trained model in the form of checkpoint
#
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np

import tensorflow as tf
import input_data
import quant_models as models
import quant_act_max

import helper

def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count,
                           model_architecture, model_size_info):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    model_size_info: Model dimensions : different lengths for different models
  """

  act_max = FLAGS.act_max
  for maxium in act_max:
    if maxium == 0:
      print('Calling quant_act_max.py to get best act_max')
      quant_act_max.FLAGS = FLAGS
      act_max = quant_act_max.get_best_act_max(act_max)

  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

  label_count = model_settings['label_count']
  fingerprint_size = model_settings['fingerprint_size']

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits = models.create_model(
      fingerprint_input,
      model_settings,
      model_architecture,
      model_size_info,
      act_max,
      is_training=False)

  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)


  # Quantize weights to 8-bits using (min,max) and write to file
  f = open('weights.h','wb')
  f.close()

  if model_architecture == "ds_cnn":
    num_layers = model_size_info[0]
    helper.write_ds_cnn_c_file('ds_cnn.c', num_layers)

    ds_cnn_h_fname = "ds_cnn.h"
    weights_h_fname = "ds_cnn_weights.h"

    f = open(ds_cnn_h_fname,'wb')
    f.close()

    with open(ds_cnn_h_fname, 'a') as f:
      helper.write_ds_cnn_h_beginning(f, wanted_words, sample_rate,
          clip_duration_ms, window_size_ms, window_stride_ms,
          dct_coefficient_count, model_size_info, act_max)

    # Quantize weights to 8-bits using (min,max) and write to file
    f = open(weights_h_fname, 'wb')
    f.close()

    total_layers = len(act_max)
    layer_no = 1
    weights_dec_bits = 0

  for v in tf.trainable_variables():
    var_name = str(v.name)
    var_values = sess.run(v)
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    dec_bits = 7-int_bits
    # convert to [-128,128) or int8
    var_values = np.round(var_values*2**dec_bits)
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    with open('weights.h','a') as f:
      f.write('#define '+var_name+' {')
    if(len(var_values.shape)>2): #convolution layer weights
      transposed_wts = np.transpose(var_values,(3,0,1,2))
    else: #fully connected layer weights or biases of any layer
      transposed_wts = np.transpose(var_values)
    with open('weights.h','a') as f:
      transposed_wts.tofile(f,sep=", ",format="%d")
      f.write('}\n')
    # convert back original range but quantized to 8-bits or 256 levels
    var_values = var_values/(2**dec_bits)
    # update the weights in tensorflow graph for quantizing the activations
    var_values = sess.run(tf.assign(v,var_values))
    print(var_name+' number of wts/bias: '+str(var_values.shape)+\
            ' dec bits: '+str(dec_bits)+\
            ' max: ('+str(var_values.max())+','+str(max_value)+')'+\
            ' min: ('+str(var_values.min())+','+str(min_value)+')')

    if model_architecture == "ds_cnn":
      conv_layer_no = layer_no // 2 + 1

      wt_or_bias = 'BIAS'
      if 'weights' in var_name:
        wt_or_bias = 'WT'

      with open(weights_h_fname, 'a') as f:
        if conv_layer_no == 1:
          f.write('#define CONV1_{} {{'.format(wt_or_bias))
        elif conv_layer_no <= num_layers:
          if layer_no % 2 == 0:
            f.write('#define CONV{}_DS_{} {{'.format(conv_layer_no, wt_or_bias))
          else:
            f.write('#define CONV{}_PW_{} {{'.format(conv_layer_no, wt_or_bias))
        else:
          f.write('#define FINAL_FC_{} {{'.format(wt_or_bias))

        transposed_wts.tofile(f, sep=", ", format="%d")
        f.write('}\n')

      if 'weights' in var_name:
        weights_dec_bits = dec_bits

      if 'biases' in var_name:
        # if averege pool layer, go to the next one
        if layer_no == total_layers - 2:
          layer_no += 1

        input_dec_bits = 7 - np.log2(act_max[layer_no - 1])
        output_dec_bits = 7 - np.log2(act_max[layer_no])
        weights_x_input_dec_bits = input_dec_bits + weights_dec_bits
        bias_lshift = int(weights_x_input_dec_bits - dec_bits)
        output_rshift = int(weights_x_input_dec_bits - output_dec_bits)
        print("Layer no: {} | Bias Lshift: {} | Output Rshift: {}\n".format(layer_no, bias_lshift, output_rshift))
        with open('ds_cnn.h', 'a') as f:
          if conv_layer_no == 1:
            f.write("#define CONV1_BIAS_LSHIFT {}\n".format(bias_lshift))
            f.write("#define CONV1_OUT_RSHIFT {}\n".format(output_rshift))
          elif conv_layer_no <= num_layers:
            if layer_no % 2 == 0:
              f.write("#define CONV{}_DS_BIAS_LSHIFT {}\n".format(conv_layer_no, bias_lshift))
              f.write("#define CONV{}_DS_OUT_RSHIFT {}\n".format(conv_layer_no, output_rshift))
            else:
              f.write("#define CONV{}_PW_BIAS_LSHIFT {}\n".format(conv_layer_no, bias_lshift))
              f.write("#define CONV{}_PW_OUT_RSHIFT {}\n".format(conv_layer_no, output_rshift))
          else:
            f.write("#define FINAL_FC_BIAS_LSHIFT {}\n".format(bias_lshift))
            f.write("#define FINAL_FC_OUT_RSHIFT {}\n".format(output_rshift))

        layer_no += 1

  if model_architecture == "ds_cnn":
    input_dec_bits = 7 - np.log2(act_max[len(act_max) - 3])
    output_dec_bits = 7 - np.log2(act_max[len(act_max) - 2])

    if input_dec_bits > output_dec_bits:
      output_dec_bits = input_dec_bits

    with open(ds_cnn_h_fname, 'a') as f:
      f.write("#define AVG_POOL_OUT_LSHIFT {}\n\n".format(int(output_dec_bits - input_dec_bits)))
      helper.write_ds_cnn_h_end(f, num_layers)

def main(_):

  # Create the model, load weights from checkpoint and dump
  run_quant_inference(FLAGS.wanted_words, FLAGS.sample_rate,
                      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
                      FLAGS.model_architecture, FLAGS.model_size_info)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--validation_dir',
      type=str,
      default=None,
      help="""\
      Where to get the speech validation data.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--training_percentage',
      type=int,
      default=80,
      help='What percentage of wavs to use as a training set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--act_max',
      type=float,
      nargs="+",
      default=[128,128,128],
      help='activations max')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
