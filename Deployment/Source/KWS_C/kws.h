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

#ifndef __KWS_H__
#define __KWS_H__

#include "nn.h"
#include "mfcc.h"

#ifdef   __cplusplus
extern "C"
{
#endif

void kws_nn_init(int record_win, int sliding_win_len);
void kws_nn_init_with_buffer(int16_t* audio_data_buffer);
void kws_nn_deinit();
void kws_enable(int enable);
void kws_extract_features_with_buffer(int16_t *buffer, int size);
void kws_extract_features();
void kws_classify();
void kws_average_predictions();
int kws_get_top_class(q7_t* prediction);

extern int16_t* kws_audio_buffer;
extern q7_t *kws_output;
extern q7_t *kws_predictions;
extern q7_t *kws_averaged_output;
extern int kws_num_frames;
extern int kws_frame_len;
extern int kws_frame_shift;
extern int kws_num_out_classes;
extern int kws_audio_block_size;
extern int kws_audio_buffer_size;

#ifdef   __cplusplus
}
#endif

#endif
