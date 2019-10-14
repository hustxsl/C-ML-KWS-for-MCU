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

import math
import cmath
import numpy as np
import os

def write_ds_cnn_license(f):
  f.write('/*\n')
  f.write(' * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.\n')
  f.write(' *\n')
  f.write(' * SPDX-License-Identifier: Apache-2.0\n')
  f.write(' *\n')
  f.write(' * Licensed under the Apache License, Version 2.0 (the License); you may\n')
  f.write(' * not use this file except in compliance with the License.\n')
  f.write(' * You may obtain a copy of the License at\n')
  f.write(' *\n')
  f.write(' * www.apache.org/licenses/LICENSE-2.0\n')
  f.write(' *\n')
  f.write(' * Unless required by applicable law or agreed to in writing, software\n')
  f.write(' * distributed under the License is distributed on an AS IS BASIS, WITHOUT\n')
  f.write(' * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n')
  f.write(' * See the License for the specific language governing permissions and\n')
  f.write(' * limitations under the License.\n')
  f.write(' */\n\n')

def write_ds_cnn_h_beginning(f, wanted_words, sample_rate, clip_duration_ms,
                             window_size_ms, window_stride_ms, dct_coefficient_count,
                             model_size_info, act_max):
  write_ds_cnn_license(f)

  f.write("#ifndef __DS_CNN_H__\n")
  f.write("#define __DS_CNN_H__\n\n")
  f.write('#include "nn.h"\n')
  f.write('#include "ds_cnn_weights.h"\n')
  f.write('#include "local_NN.h"\n')
  f.write('#include "arm_math.h"\n\n')
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

  input_x = dct_coefficient_count
  input_y = spectrogram_length

  f.write("#define SAMP_FREQ {}\n".format(sample_rate))
  f.write("#define MFCC_DEC_BITS {}\n".format(int(7 - np.log2(act_max[0]))))
  f.write("#define FRAME_SHIFT_MS {}\n".format(int(window_stride_ms)))
  f.write("#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))\n")
  f.write("#define NUM_FRAMES {}\n".format(spectrogram_length))
  f.write("#define NUM_MFCC_COEFFS {}\n".format(dct_coefficient_count))
  f.write("#define FRAME_LEN_MS {}\n".format(int(window_size_ms)))
  f.write("#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))\n\n")

  f.write("#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)\n")
  f.write("#define OUT_DIM {}\n\n".format(int(len(wanted_words.split(',')) + 2)))

  num_layers = model_size_info[0]
  i = 1
  for layer_no in range(1, num_layers + 1):
    f.write("#define CONV{}_OUT_CH {}\n".format(layer_no, model_size_info[i]))
    i += 1
    ky = model_size_info[i]
    i += 1
    kx = model_size_info[i]
    i += 1
    sy = model_size_info[i]
    i += 1
    sx = model_size_info[i]
    out_x = math.ceil(float(input_x) / float(sx))
    out_y = math.ceil(float(input_y) / float(sy))
    pad_x = max((out_x - 1) * sx + kx - input_x, 0) // 2
    pad_y = max((out_y - 1) * sy + ky - input_y, 0) // 2
    if layer_no == 1:
      f.write("#define CONV1_IN_X NUM_MFCC_COEFFS\n")
      f.write("#define CONV1_IN_Y NUM_FRAMES\n")
      f.write("#define CONV{}_KX {}\n".format(layer_no, kx))
      f.write("#define CONV{}_KY {}\n".format(layer_no, ky))
      f.write("#define CONV{}_SX {}\n".format(layer_no, sx))
      f.write("#define CONV{}_SY {}\n".format(layer_no, sy))
      f.write("#define CONV{}_PX {}\n".format(layer_no, pad_x))
      f.write("#define CONV{}_PY {}\n".format(layer_no, pad_y))
      f.write("#define CONV{}_OUT_X {}\n".format(layer_no, int(out_x)))
      f.write("#define CONV{}_OUT_Y {}\n".format(layer_no, int(out_y)))

    else:
      f.write("#define CONV{1}_IN_X CONV{0}_OUT_X\n".format(layer_no - 1, layer_no))
      f.write("#define CONV{1}_IN_Y CONV{0}_OUT_Y\n".format(layer_no - 1, layer_no))
      f.write("#define CONV{}_DS_KX {}\n".format(layer_no, kx))
      f.write("#define CONV{}_DS_KY {}\n".format(layer_no, ky))
      f.write("#define CONV{}_DS_SX {}\n".format(layer_no, sx))
      f.write("#define CONV{}_DS_SY {}\n".format(layer_no, sy))
      f.write("#define CONV{}_DS_PX {}\n".format(layer_no, int(pad_x)))
      f.write("#define CONV{}_DS_PY {}\n".format(layer_no, int(pad_y)))
      f.write("#define CONV{0}_OUT_X {1}\n".format(layer_no, int(out_x)))
      f.write("#define CONV{0}_OUT_Y {1}\n".format(layer_no, int(out_y)))

    i += 1
    f.write("\n")
    input_x = out_x
    input_y = out_y

def write_ds_cnn_h_end(f, num_layers):
  f.write(
    '#define SCRATCH_BUFFER_SIZE (2*2*CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY + 2*CONV2_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y)\n\n')
  f.write('#endif\n')

def write_ds_cnn_c_file(fname, num_layers):
  f = open(fname, 'wb')
  f.close()
  with open(fname, 'a') as f:
    write_ds_cnn_license(f)

    f.write('#include "ds_cnn.h"\n')
    f.write('#include "stdlib.h"\n\n')

    f.write('static int frame_len;\n')
    f.write('static int frame_shift;\n')
    f.write('static int num_mfcc_features;\n')
    f.write('static int num_frames;\n')
    f.write('static int num_out_classes;\n')
    f.write('static int in_dec_bits;\n\n')
    f.write('static q7_t* scratch_pad;\n')
    f.write('static q7_t* col_buffer;\n')
    f.write('static q7_t* buffer1;\n')
    f.write('static q7_t* buffer2;\n\n')

    for layer_no in range(0, num_layers):
      if layer_no == 0:
        f.write("static const q7_t conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY]=CONV1_WT;\n")
        f.write("static const q7_t conv1_bias[CONV1_OUT_CH]=CONV1_BIAS;\n")
      else:
        f.write(
          "static const q7_t conv{1}_ds_wt[CONV{0}_OUT_CH*CONV{1}_DS_KX*CONV{1}_DS_KY]=CONV{1}_DS_WT;\n".format(layer_no, layer_no + 1))
        f.write("static const q7_t conv{1}_ds_bias[CONV{0}_OUT_CH]=CONV{1}_DS_BIAS;\n".format(layer_no, layer_no + 1))
        f.write(
          "static const q7_t conv{1}_pw_wt[CONV{1}_OUT_CH*CONV{0}_OUT_CH]=CONV{1}_PW_WT;\n".format(layer_no, layer_no + 1))
        f.write("static const q7_t conv{0}_pw_bias[CONV{0}_OUT_CH]=CONV{0}_PW_BIAS;\n".format(layer_no + 1))

    f.write("static const q7_t final_fc_wt[CONV{0}_OUT_CH*OUT_DIM]=FINAL_FC_WT;\n".format(num_layers))
    f.write("static const q7_t final_fc_bias[OUT_DIM]=FINAL_FC_BIAS;\n\n")

    f.write('int nn_get_frame_len() {\n')
    f.write('  return frame_len;\n')
    f.write('}\n\n')
    f.write('int nn_get_frame_shift() {\n')
    f.write('  return frame_shift;\n')
    f.write('}\n\n')
    f.write('int nn_get_num_mfcc_features() {\n')
    f.write('  return num_mfcc_features;\n')
    f.write('}\n\n')
    f.write('int nn_get_num_frames() {\n')
    f.write('  return num_frames;\n')
    f.write('}\n\n')
    f.write('int nn_get_num_out_classes() {\n')
    f.write('  return num_out_classes;\n')
    f.write('}\n\n')
    f.write('int nn_get_in_dec_bits() {\n')
    f.write('  return in_dec_bits;\n')
    f.write('}\n\n')

    f.write("void nn_init()\n")
    f.write("{\n")
    f.write('  scratch_pad = malloc(sizeof(q7_t) * SCRATCH_BUFFER_SIZE);\n')
    f.write("  buffer1 = scratch_pad;\n")
    f.write("  buffer2 = buffer1 + (CONV1_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y);\n")
    f.write("  col_buffer = buffer2 + (CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y);\n")
    f.write("  frame_len = FRAME_LEN;\n")
    f.write("  frame_shift = FRAME_SHIFT;\n")
    f.write("  num_mfcc_features = NUM_MFCC_COEFFS;\n")
    f.write("  num_frames = NUM_FRAMES;\n")
    f.write("  num_out_classes = OUT_DIM;\n")
    f.write("  in_dec_bits = MFCC_DEC_BITS;\n")
    f.write("}\n\n")

    f.write('void nn_deinit()\n')
    f.write('{\n')
    f.write('  free(scratch_pad);\n')
    f.write('}\n\n')

    f.write("void nn_run_nn(q7_t* in_data, q7_t* out_data)\n")
    f.write("{\n")
    for layer_no in range(0, num_layers):
      if layer_no == 0:
        f.write("  //CONV1 : regular convolution\n")
        f.write(
          "  arm_convolve_HWC_q7_basic_nonsquare(in_data, CONV1_IN_X, CONV1_IN_Y, 1, conv1_wt, CONV1_OUT_CH, CONV1_KX, CONV1_KY, CONV1_PX, CONV1_PY, CONV1_SX, CONV1_SY, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_X, CONV1_OUT_Y, (q15_t*)col_buffer, NULL);\n")
        f.write("  arm_relu_q7(buffer1,CONV1_OUT_X*CONV1_OUT_Y*CONV1_OUT_CH);\n\n")
      else:
        f.write("  //CONV{} : DS + PW conv\n".format(layer_no + 1))
        f.write("  //Depthwise separable conv (batch norm params folded into conv wts/bias)\n")
        f.write(
          "  arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV{1}_IN_X,CONV{1}_IN_Y,CONV{0}_OUT_CH,conv{1}_ds_wt,CONV{0}_OUT_CH,CONV{1}_DS_KX,CONV{1}_DS_KY,CONV{1}_DS_PX,CONV{1}_DS_PY,CONV{1}_DS_SX,CONV{1}_DS_SY,conv{1}_ds_bias,CONV{1}_DS_BIAS_LSHIFT,CONV{1}_DS_OUT_RSHIFT,buffer2,CONV{1}_OUT_X,CONV{1}_OUT_Y,(q15_t*)col_buffer, NULL);\n".format(layer_no, layer_no + 1))
        f.write("  arm_relu_q7(buffer2,CONV{0}_OUT_X*CONV{0}_OUT_Y*CONV{0}_OUT_CH);\n".format(layer_no + 1))

        f.write("  //Pointwise conv\n")
        f.write(
          "  arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV{1}_OUT_X, CONV{1}_OUT_Y, CONV{0}_OUT_CH, conv{1}_pw_wt, CONV{1}_OUT_CH, 1, 1, 0, 0, 1, 1, conv{1}_pw_bias, CONV{1}_PW_BIAS_LSHIFT, CONV{1}_PW_OUT_RSHIFT, buffer1, CONV{1}_OUT_X, CONV{1}_OUT_Y, (q15_t*)col_buffer, NULL);\n".format(layer_no, layer_no + 1))
        f.write(
          "  arm_relu_q7(buffer1,CONV{0}_OUT_X*CONV{0}_OUT_Y*CONV{0}_OUT_CH);\n\n".format(layer_no + 1))

    f.write("  //Average pool\n")
    f.write(
      "  arm_avepool_q7_HWC_nonsquare (buffer1,CONV{0}_OUT_X,CONV{0}_OUT_Y,CONV{0}_OUT_CH,CONV{0}_OUT_X,CONV{0}_OUT_Y,0,0,1,1,1,1,NULL,buffer2, AVG_POOL_OUT_LSHIFT);\n".format(num_layers))
    f.write(
      "  arm_fully_connected_q7(buffer2, final_fc_wt, CONV{0}_OUT_CH, OUT_DIM, FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT, final_fc_bias, out_data, (q15_t*)col_buffer);\n".format(num_layers))
    f.write("}\n")
