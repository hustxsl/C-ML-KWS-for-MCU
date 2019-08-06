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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf

import joblib
import pandas as pd
import train

from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, fmin, Trials
from timeit import default_timer as timer
import csv
import helper

ITERATION = 0
BEST_ACCURACY = 0

def parse_model_size_info(msi):
  num_layers = msi['num_layers']

  model_size_info = [num_layers]
  for i in range(num_layers):
    layer = msi['layers'][i]

    conv_feat = layer['num_features']

    if 'kernel_time' in layer.keys():
      conv_kt = layer['kernel_time']
    else:
      conv_kt = 3

    if 'kernel_freq' in layer.keys():
      conv_kf = layer['kernel_freq']
    else:
      conv_kf = 3

    conv_st = 2 if i == 0 else 1
    conv_sf = 2 if i == 0 else 1

    model_size_info.append(int(conv_feat))
    model_size_info.append(int(conv_kt))
    model_size_info.append(int(conv_kf))
    model_size_info.append(int(conv_st))
    model_size_info.append(int(conv_sf))

  return model_size_info


def objective(parameters):
  """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
  global ITERATION
  global BEST_ACCURACY

  work_dir = os.path.join(FLAGS.hyperopt_dir, 'work-{}'.format(ITERATION))
  FLAGS.train_dir = os.path.join(work_dir, 'training')
  FLAGS.summaries_dir = os.path.join(work_dir, 'retrain_logs')

  model_size_info = parameters['model_size_info']
  FLAGS.model_size_info = parse_model_size_info(model_size_info)

  FLAGS.window_size_ms = parameters['window_size_ms']
  FLAGS.window_stride_ms = round(FLAGS.window_size_ms * parameters['window_stride_coeff'])
  FLAGS.dct_coefficient_count = parameters['dct_coefficient_count']

  start = timer()

  print()
  print(parameters)
  sys.stdout.flush()

  # Hyperopt tends to use same hyperparameters several times
  df = pd.read_csv(FLAGS.trials_file)
  if str(FLAGS) in df['flags'].tolist():
    print("There is already trial with these experiments!")
    loss = df[df.flags == str(FLAGS)]['loss'].tolist()[0]
    best_val_acc = df[df.flags == str(FLAGS)]['best_val_acc'].tolist()[0]
    num_params = df[df.flags == str(FLAGS)]['num_params'].tolist()[0]
    return {'loss': loss, 'flags': FLAGS, 'iteration': ITERATION,
            'acc': best_val_acc, "num_params": num_params,
            'train_time': 0, 'status': STATUS_OK}

  train.FLAGS = FLAGS
  best_val_acc, num_params = train.train(False)

  loss = 1 - best_val_acc
  run_time = timer() - start

  with open(FLAGS.trials_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([ITERATION, loss, best_val_acc, run_time, num_params, FLAGS])

  return {'loss': loss, 'flags': FLAGS, 'iteration': ITERATION,
          'acc': best_val_acc, "num_params": num_params,
          'train_time': run_time, 'status': STATUS_OK}


def run_trials(parameters):
  global ITERATION
  global BEST_ACCURACY

  if not os.path.exists(FLAGS.hyperopt_dir):
    os.makedirs(FLAGS.hyperopt_dir)

  if not os.path.exists(FLAGS.trials_file):
    # Write the headers to the file
    with open(FLAGS.trials_file, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['ITERATION', 'loss', 'best_val_acc', 'run_time', 'num_params', 'flags'])

  trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
  max_trials = 2  # initial max_trials. put something small to not have to wait

  iterations = [int(x.split('-')[1].split('.')[0]) for x in os.listdir(FLAGS.hyperopt_dir) if x.endswith(".hyperopt")]
  if len(iterations) == 0:
    trials = Trials()
    print("Created new Trials object")
  else:
    ITERATION = max(iterations)
    trials_fname = os.path.join(FLAGS.hyperopt_dir, "trial-{}.hyperopt".format(ITERATION))
    try:  # try to load an already saved trials object, and increase the max
      trials = joblib.load(trials_fname)
      max_trials = len(trials.trials) + trials_step
      print("\nSuccessfully loaded! Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # if failed to load, try again with previous file
      ITERATION -= 1
      trials = joblib.load(trials_fname)
      max_trials = len(trials.trials) + trials_step
      print("\nFailed to load! Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))

    BEST_ACCURACY = max([d['result']['acc'] for d in
                         trials.trials if d['result']['status'] == 'ok'])

  # Keep track of evals
  best = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=max_trials, trials=trials, rstate=np.random.RandomState(35))

  BEST_ACCURACY = max([d['result']['acc'] for d in
                       trials.trials if d['result']['status'] == 'ok'])

  try:
    # save the trials object
    joblib.dump(trials, os.path.join(FLAGS.hyperopt_dir, "trial-{}.hyperopt".format(ITERATION + 1)))
  except:
    print("Failed to save trials file")


def main(_):
  space = {
    'window_size_ms': hp.choice('window_size_ms', [32, 64, 128]),
    'window_stride_coeff': hp.quniform('window_stride_coeff', 0.25, 0.75, 0.25),
    'dct_coefficient_count': hp.choice('dct_coefficient_count', [5, 10, 20, 40]),
    'model_size_info': hp.choice('model_size_info',
        [
          {
            'num_layers': 3,
            'layers': [
              {
                'num_features': hp.quniform('feat_31', 16, 64, 16),
                'kernel_time': hp.quniform('kt-3', 5, 20, 5),
                'kernel_freq': hp.quniform('kf-3', 2, 8, 2),
              },
              { 'num_features': hp.quniform('feat_32', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_33', 16, 64, 16) },
            ]
          },
          {
            'num_layers': 4,
            'layers': [
              {
                'num_features': hp.quniform('feat_41', 16, 64, 16),
                'kernel_time': hp.quniform('kt-4', 5, 20, 5),
                'kernel_freq': hp.quniform('kf-4', 2, 8, 2),
              },
              { 'num_features': hp.quniform('feat_42', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_43', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_44', 16, 64, 16) },
            ]
          },
          {
            'num_layers': 5,
            'layers': [
              {
                'num_features': hp.quniform('feat_51', 16, 64, 16),
                'kernel_time': hp.quniform('kt-5', 5, 20, 5),
                'kernel_freq': hp.quniform('kf-5', 2, 8, 2),
              },
              { 'num_features': hp.quniform('feat_52', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_53', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_54', 16, 64, 16) },
              { 'num_features': hp.quniform('feat_55', 16, 64, 16) },
            ]
          },
      ]),
  }

  FLAGS.trials_file = os.path.join(FLAGS.hyperopt_dir, FLAGS.trials_file)
  FLAGS.start_checkpoint = ''

  # skip testing
  FLAGS.testing_percentage = 0

  # loop indefinitely and stop whenever you like
  try:
    while True:
      run_trials(space)
  except KeyboardInterrupt:
    pass

  print("Best accuracy: {} | Trials: {}".format(BEST_ACCURACY, FLAGS.trials_file))

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
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
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
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
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
      '--how_many_training_steps',
      type=str,
      default='300,300,300',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=100,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.0005,0.0001,0.00002',
      help='How large a learning rate to use when training.')
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
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--hyperopt_dir',
      type=str,
      default='/tmp/hyperopt',
      help='Where to save hyper parameters optimize files for TensorBoard.')
  parser.add_argument(
      '--trials_file',
      type=str,
      default='trials.cvs',
      help='Where to save trials file for TensorBoard.')

  FLAGS, unparsed = parser.parse_known_args()

  # Below args would be force overrided later for train.py
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
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
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')

  # Parse it again to generate train flags
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
