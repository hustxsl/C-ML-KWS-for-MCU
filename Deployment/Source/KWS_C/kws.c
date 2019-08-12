/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: Keyword spotting example code using MFCC feature extraction
 * and neural network.
 */

#include "kws.h"

int16_t* kws_audio_buffer;
q7_t *kws_output;
q7_t *kws_predictions;
q7_t *kws_averaged_output;
int kws_num_frames;
int kws_frame_len;
int kws_frame_shift;
int kws_num_out_classes;
int kws_audio_block_size;
int kws_audio_buffer_size;

static q7_t *mfcc_buffer;
static int num_mfcc_features;

static int recording_win;
static int sliding_window_len;

static void kws_deinit()
{
  free(mfcc_buffer);
  free(kws_output);
  free(kws_predictions);
  free(kws_averaged_output);
  mfcc_deinit();
}

static void kws_init()
{
  num_mfcc_features = nn_get_num_mfcc_features();
  kws_num_frames = nn_get_num_frames();
  kws_frame_len = nn_get_frame_len();
  kws_frame_shift = nn_get_frame_shift();
  int mfcc_dec_bits = nn_get_in_dec_bits();
  kws_num_out_classes = nn_get_num_out_classes();
  mfcc_init(num_mfcc_features, kws_frame_len, mfcc_dec_bits);
  mfcc_buffer = malloc(sizeof(q7_t) * kws_num_frames*num_mfcc_features);
  kws_output = malloc(sizeof(q7_t) * kws_num_out_classes);
  kws_averaged_output = malloc(sizeof(q7_t) * kws_num_out_classes);
  kws_predictions = malloc(sizeof(q7_t) * sliding_window_len*kws_num_out_classes);
  kws_audio_block_size = recording_win*kws_frame_shift;
  kws_audio_buffer_size = kws_audio_block_size + kws_frame_len - kws_frame_shift;
}

void kws_nn_init(int record_win, int sliding_win_len)
{
  nn_init();
  recording_win = record_win;
  sliding_window_len = sliding_win_len;
  kws_init();
}

void kws_nn_init_with_buffer(int16_t* audio_data_buffer)
{
  nn_init();
  kws_audio_buffer = audio_data_buffer;
  recording_win = nn_get_num_frames();
  sliding_window_len = 1;
  kws_init();
}

void kws_nn_deinit()
{
  kws_deinit();
  nn_deinit();
}

void kws_extract_features()
{
  if(kws_num_frames>recording_win) {
    //move old features left
    memmove(mfcc_buffer,mfcc_buffer+(recording_win*num_mfcc_features),(kws_num_frames-recording_win)*num_mfcc_features);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (kws_num_frames-recording_win)*num_mfcc_features;
  for (uint16_t f = 0; f < recording_win; f++) {
    mfcc_compute(kws_audio_buffer+(f*kws_frame_shift),&mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += num_mfcc_features;
  }
}

void kws_classify()
{
  nn_run_nn(mfcc_buffer, kws_output);
  // Softmax
  arm_softmax_q7(kws_output,kws_num_out_classes,kws_output);
}

int kws_get_top_class(q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<kws_num_out_classes;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }
  }
  return max_ind;
}

void kws_average_predictions()
{
  // shift the old kws_predictions left
  arm_copy_q7((q7_t *)(kws_predictions+kws_num_out_classes), (q7_t *)kws_predictions, (sliding_window_len-1)*kws_num_out_classes);
  // add new kws_predictions at the end
  arm_copy_q7((q7_t *)kws_output, (q7_t *)(kws_predictions+(sliding_window_len-1)*kws_num_out_classes), kws_num_out_classes);
  //compute averages
  int sum;
  for(int j=0;j<kws_num_out_classes;j++) {
    sum=0;
    for(int i=0;i<sliding_window_len;i++)
      sum += kws_predictions[i*kws_num_out_classes+j];
    kws_averaged_output[j] = (q7_t)(sum/sliding_window_len);
  }
}

