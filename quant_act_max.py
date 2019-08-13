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

def run_quant_inference(act_max):
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

  wanted_words = FLAGS.wanted_words
  sample_rate = FLAGS.sample_rate
  clip_duration_ms = FLAGS.clip_duration_ms
  window_size_ms = FLAGS.window_size_ms
  window_stride_ms = FLAGS.window_stride_ms
  dct_coefficient_count = FLAGS.dct_coefficient_count
  model_architecture = FLAGS.model_architecture
  model_size_info = FLAGS.model_size_info

  total_layers = len(act_max)
  layer_no = 1
  weights_dec_bits = 0

  tf.reset_default_graph()
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

  if FLAGS.validation_dir is None:
    FLAGS.validation_dir = FLAGS.data_dir
  validation_audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.validation_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','),
      100, 0, model_settings)

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


  # Quantize weights to 8-bits using (min,max)

  for v in tf.trainable_variables():
    var_name = str(v.name)
    var_values = sess.run(v)
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    dec_bits = 7-int_bits
    # convert to [-128,128) or int8
    var_values = np.round(var_values*2**dec_bits)
    # convert back original range but quantized to 8-bits or 256 levels
    var_values = var_values/(2**dec_bits)
    # update the weights in tensorflow graph for quantizing the activations
    var_values = sess.run(tf.assign(v,var_values))

    if 'weights' in var_name:
      weights_dec_bits = dec_bits

    if 'biases' in var_name:
      # if averege pool layer, go to the next one
      if layer_no == total_layers - 2:
        layer_no += 1

      if act_max[layer_no] != 0 and act_max[layer_no - 1] != 0:
        input_dec_bits = 7 - np.log2(act_max[layer_no - 1])
        output_dec_bits = 7 - np.log2(act_max[layer_no])
        weights_x_input_dec_bits = input_dec_bits + weights_dec_bits
        bias_lshift = int(weights_x_input_dec_bits - dec_bits)
        output_rshift = int(weights_x_input_dec_bits - output_dec_bits)
        if bias_lshift < 0 or output_rshift < 0:
          print("CMSIS-5 NN doesn't support negative shift now!")
          tf.reset_default_graph()
          sess.close()
          return -1

      layer_no += 1

  # validation set
  set_size = validation_audio_processor.set_size('validation')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    validation_fingerprints, validation_ground_truth = (
        validation_audio_processor.get_data(FLAGS.batch_size, i,
                                            model_settings, 0.0,
                                            0.0, 0, 'validation', sess))
    validation_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: validation_fingerprints,
            ground_truth_input: validation_ground_truth,
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (validation_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                  (total_accuracy * 100, set_size))

  tf.reset_default_graph()
  sess.close()
  return total_accuracy

def get_best_act_max(act_max):
  num_layers = FLAGS.model_size_info[0]
  act_max_unlimit = [0] * (num_layers * 2 + 2)

  if act_max is None:
    act_max = act_max_unlimit

  for start_layer in range(len(act_max) - 1, -1, -1):
    if act_max[start_layer] > 0:
      break

  print("Start with - Layer: {}".format(start_layer))
  print("act_max: {}".format(act_max))

  accuracy = run_quant_inference(act_max_unlimit)
  print("Without quantization. Accuracy: {}".format(accuracy))

  for act_layer in range(start_layer, len(act_max), 1):
    best = 0
    best_act_max = 0

    for i in range(8):
      maximum = 2**i
      if act_max[act_layer] > maximum:
        continue

      act_max[act_layer] = maximum

      print("Training next - Layer: {} | Max: {}".format(act_layer, maximum))
      print("act_max: {}".format(act_max))
      accuracy = run_quant_inference(act_max)
      print("Accuracy: {}\n".format(accuracy))

      if best < accuracy:
        best = accuracy
        best_act_max = maximum

    act_max[act_layer] = best_act_max
    print("Best act_max for layer {}: {}".format(act_layer, act_max))

  print("Best act_max: {}".format(act_max))
  return act_max

def main(_):
  get_best_act_max(FLAGS.act_max)

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
      default=None,
      help='activations max')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
