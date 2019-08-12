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

#ifndef __NN_H__
#define __NN_H__

#include "arm_nnfunctions.h"

#ifdef   __cplusplus
extern "C"
{
#endif

void nn_init();
void nn_deinit();
void nn_run_nn(q7_t* in_data, q7_t* out_data);
int nn_get_num_mfcc_features();
int nn_get_num_frames();
int nn_get_frame_len();
int nn_get_frame_shift();
int nn_get_num_out_classes();
int nn_get_in_dec_bits();

#ifdef   __cplusplus
}
#endif

#endif
