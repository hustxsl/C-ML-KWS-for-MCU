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
 * Description: MFCC feature extraction to match with TensorFlow MFCC Op
 */

#include <string.h>

#include "mfcc.h"
#include "float.h"

static int num_mfcc_features;
static int frame_len;
static int frame_len_padded;
static int mfcc_dec_bits;
static float * frame;
static float * buffer;
static float * mel_energies;
static float * window_func;
static int32_t * fbank_filter_first;
static int32_t * fbank_filter_last;
static float ** mel_fbank;
static float * dct_matrix;
static arm_rfft_fast_instance_f32 * rfft;
static float * create_dct_matrix(int32_t input_length, int32_t coefficient_count);
static float ** create_mel_fbank();

static inline float InverseMelScale(float mel_freq) {
  return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
}

static inline float MelScale(float freq) {
  return 1127.0f * logf (1.0f + freq / 700.0f);
}

void mfcc_init(int _num_mfcc_features, int _frame_len, int _mfcc_dec_bits)
{
  num_mfcc_features = _num_mfcc_features;
  frame_len = _frame_len;
  mfcc_dec_bits = _mfcc_dec_bits;

  // Round-up to nearest power of 2.
  frame_len_padded = pow(2,ceil((log(frame_len)/log(2))));

  frame = malloc(sizeof(float) * frame_len_padded);
  buffer = malloc(sizeof(float) * frame_len_padded);
  mel_energies = malloc(sizeof(float) * NUM_FBANK_BINS);

  //create window function
  window_func = malloc(sizeof(float) * frame_len);
  for (int i = 0; i < frame_len; i++)
    window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (frame_len));

  //create mel filterbank
  fbank_filter_first = malloc(sizeof(int32_t) * NUM_FBANK_BINS);
  fbank_filter_last = malloc(sizeof(int32_t) * NUM_FBANK_BINS);;
  mel_fbank = create_mel_fbank();

  //create DCT matrix
  dct_matrix = create_dct_matrix(NUM_FBANK_BINS, num_mfcc_features);

  //initialize FFT
  rfft = malloc(sizeof(arm_rfft_fast_instance_f32));
  arm_rfft_fast_init_f32(rfft, frame_len_padded);
}

void mfcc_deinit() {
  free(frame);
  free(buffer);
  free(mel_energies);
  free(window_func);
  free(fbank_filter_first);
  free(fbank_filter_last);
  free(dct_matrix);
  free(rfft);
  for(int i=0;i<NUM_FBANK_BINS;i++)
    free(mel_fbank[i]);
  free(mel_fbank);
}

static float * create_dct_matrix(int32_t input_length, int32_t coefficient_count) {
  int32_t k, n;
  float * M = malloc(sizeof(float) * input_length*coefficient_count);
  float normalizer;
  arm_sqrt_f32(2.0/(float)input_length,&normalizer);
  for (k = 0; k < coefficient_count; k++) {
    for (n = 0; n < input_length; n++) {
      M[k*input_length+n] = normalizer * cos( ((double)M_PI)/input_length * (n + 0.5) * k );
    }
  }
  return M;
}

static float ** create_mel_fbank() {

  int32_t bin, i;

  int32_t num_fft_bins = frame_len_padded/2;
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ);
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

  float *this_bin = malloc(sizeof(float) * num_fft_bins);

  float ** mel_fbank =  malloc(sizeof(float*) * NUM_FBANK_BINS);

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {

    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (i = 0; i < num_fft_bins; i++) {

      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel-mel) / (right_mel-center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }

    fbank_filter_first[bin] = first_index;
    fbank_filter_last[bin] = last_index;
    mel_fbank[bin] = malloc(sizeof(float) * last_index-first_index+1);

    int32_t j = 0;
    //copy the part we care about
    for (i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  free(this_bin);
  return mel_fbank;
}

int mfcc_compute(const int16_t * audio_data, q7_t* mfcc_out) {

  int32_t i, j, bin;
  int silence = 1;

  //TensorFlow way of normalizing .wav data to (-1,1)
  for (i = 0; i < frame_len; i++) {
    frame[i] = (float)audio_data[i]/(1<<15);
  }
  //Fill up remaining with zeros
  memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));

  for (i = 0; i < frame_len; i++) {
    frame[i] *= window_func[i];
  }

  //Compute FFT
  arm_rfft_fast_f32(rfft, frame, buffer, 0);

  //Convert to power spectrum
  //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32_t half_dim = frame_len_padded/2;
  float first_energy = buffer[0] * buffer[0],
        last_energy =  buffer[1] * buffer[1];  // handle this special case
  for (i = 1; i < half_dim; i++) {
    float real = buffer[i*2], im = buffer[i*2 + 1];
    buffer[i] = real*real + im*im;
  }
  buffer[0] = first_energy;
  buffer[half_dim] = last_energy;

  for (i = 0; i < half_dim; i++) {
    if (buffer[i] > 1)
      silence = 0;
  }

  float sqrt_data;
  //Apply mel filterbanks
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float mel_energy = 0;
    int32_t first_index = fbank_filter_first[bin];
    int32_t last_index = fbank_filter_last[bin];
    for (i = first_index; i <= last_index; i++) {
      arm_sqrt_f32(buffer[i],&sqrt_data);
      mel_energy += (sqrt_data) * mel_fbank[bin][j++];
    }
    mel_energies[bin] = mel_energy;

    //avoid log of zero
    if (mel_energy == 0.0)
      mel_energies[bin] = FLT_MIN;
  }

  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
    mel_energies[bin] = logf(mel_energies[bin]);

  //Take DCT. Uses matrix mul.
  for (i = 0; i < num_mfcc_features; i++) {
    float sum = 0.0;
    for (j = 0; j < NUM_FBANK_BINS; j++) {
      sum += dct_matrix[i*NUM_FBANK_BINS+j] * mel_energies[j];
    }

    //Input is Qx.mfcc_dec_bits (from quantization step)
    sum *= (0x1<<mfcc_dec_bits);
    sum = round(sum);
    if(sum >= 127)
      mfcc_out[i] = 127;
    else if(sum <= -128)
      mfcc_out[i] = -128;
    else
      mfcc_out[i] = sum;
  }

  return silence;
}

