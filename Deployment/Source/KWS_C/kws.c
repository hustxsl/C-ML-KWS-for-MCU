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

#define FRAME_TAIL (kws_frame_len - kws_frame_shift)

static q7_t *mfcc_buffer;
static int num_mfcc_features;

static int recording_win;
static int sliding_window_len;

#define KWS_STARTED 1
#define KWS_STANDBY 2
#define KWS_SLEEPING 3
#define KWS_STOPPED 4
static int kws_state;
static int kws_silence_count;
static int kws_silence_threshold;

//caching tail buffer
static int16_t *kws_audio_cache_buffer;
static int kws_audio_cache_size;

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
  kws_audio_buffer_size = kws_audio_block_size + FRAME_TAIL;

  kws_state = KWS_STANDBY;
  kws_silence_count = 0;

  //consider 1s silence as inactive
  kws_silence_threshold = SAMP_FREQ / kws_frame_shift;

  kws_audio_cache_buffer = malloc(sizeof(int16_t) * kws_frame_len * 2);
  kws_audio_cache_size = 0;
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

void kws_enable(int enable)
{
  if (enable && kws_state == KWS_STOPPED) {
    kws_state = KWS_STANDBY;
    //printf("%s, %d enabled!\n", __func__, __LINE__);
  } else if (!enable && kws_state != KWS_STOPPED) {
    kws_state = KWS_STOPPED;
    //printf("%s, %d disabled!\n", __func__, __LINE__);
  }
}

static void kws_extract_features_with_frames(int16_t *frames, int num_frames)
{
  q7_t *mfcc_buffer_head;
  int silence;

  if (kws_state == KWS_STOPPED)
    return;

  if (kws_state == KWS_SLEEPING) {
    //put pending features to the end of buffer
    mfcc_buffer_head = mfcc_buffer+(kws_num_frames-1)*num_mfcc_features;
  } else {
    //move old features left
    memmove(mfcc_buffer,mfcc_buffer+num_frames*num_mfcc_features,sizeof(q7_t)*(kws_num_frames-num_frames)*num_mfcc_features);
    //compute features only for the new audio
    mfcc_buffer_head = mfcc_buffer+(kws_num_frames-num_frames)*num_mfcc_features;
  }

  while (num_frames--) {
    silence = mfcc_compute(frames,mfcc_buffer_head);
    frames += kws_frame_shift;

    if (silence && kws_state != KWS_SLEEPING)
      kws_silence_count++;
    else
      kws_silence_count = 0;

    //handle sleep
    if (kws_silence_count >= kws_num_frames) {
      //printf("%s, %d sleeping!\n", __func__, __LINE__);
      kws_state = KWS_SLEEPING;
      break;
    }

    //handle active
    if (kws_state != KWS_STARTED && !silence) {
      //printf("%s, %d starting!\n", __func__, __LINE__);

      //fill the old features with unknown
      memset(mfcc_buffer,0,sizeof(q7_t)*(kws_num_frames-1)*num_mfcc_features);

      kws_state = KWS_STARTED;
      break;
    }

    //handle inactive
    if (kws_state == KWS_STARTED && kws_silence_count >= kws_silence_threshold) {
      //printf("%s, %d standby!\n", __func__, __LINE__);
      kws_state = KWS_STANDBY;
    }

    //forward mfcc buffer
    if (kws_state != KWS_SLEEPING)
      mfcc_buffer_head += num_mfcc_features;
  }

  //handle remaining frames
  if (num_frames > 0)
    kws_extract_features_with_frames(frames, num_frames);
}

void kws_extract_features_with_buffer(int16_t *buffer, int size)
{
  int num_frames;

  if (kws_audio_cache_size) {
    //extend the cache buffer to frames and process it
    num_frames = kws_audio_cache_size/kws_frame_shift+1;
    int extra_size = num_frames*kws_frame_shift+FRAME_TAIL-kws_audio_cache_size;

    //no enough data to process
    if (extra_size > size) {
      num_frames = 0;
      goto out;
    }

    memcpy(kws_audio_cache_buffer+kws_audio_cache_size,buffer,sizeof(int16_t)*extra_size);
    kws_extract_features_with_frames(kws_audio_cache_buffer, num_frames);

    //keep an extra frame to make the features contiguous
    extra_size -= kws_frame_shift;
    buffer += extra_size;
    size -= extra_size;
  }

  num_frames = (size-FRAME_TAIL)/kws_frame_shift;
  kws_extract_features_with_frames(buffer, num_frames);

out:
  //cache the tail
  kws_audio_cache_size = size-num_frames*kws_frame_shift;
  memcpy(kws_audio_cache_buffer,buffer+num_frames*kws_frame_shift,sizeof(int16_t)*kws_audio_cache_size);
}

void kws_extract_features()
{
  if (!kws_audio_buffer)
    return;

  kws_extract_features_with_frames(kws_audio_buffer, recording_win);
}

void kws_classify()
{
  if (kws_state == KWS_STOPPED) {
    memset(kws_output,0,sizeof(q7_t) * kws_num_out_classes);
    kws_output[1] = 126;
    return;
  }

  if (kws_state == KWS_SLEEPING) {
    memset(kws_output,0,sizeof(q7_t) * kws_num_out_classes);
    kws_output[0] = 126;
    return;
  }

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

