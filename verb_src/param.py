# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#This program was made based on the original run_glue.py program with Copyright
# from the companies above and licensed under the Apache License, Version 2.0 (the "License");
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
#
# THIS PROGRAM CONTAINS CHANGES FROM THE ORIGINAL VERSION, MADE BY
# MARCELO FINGER AND FELIPE SERRAS.
# If not stated otherwise, this version is licensed under the same
# license as the original version .

'''Param Module
Module that defines the class Parameters, used as a data structure to store the set of parameters
for each experiment. Also define the class TrainingResources to be used as a data structure to store
logger, model and tokenizer easily.
'''
# -*- coding: utf-8 -*-
class Parameters:
    adam_epsilon        : float = None
    amp_imported        : bool  = None
    cache_dir           : str   = None
    checkpointing_mode  : str   =None
    config_name         : str   = None
    data_dir            : str   = None
    data_from_main      : bool  = None
    data_main_file      : str   = None
    data_train_file     : str   = None
    data_eval_file     : str   = None
    data_test_file     : str   = None
    device                      = None
    do_eval             : bool  = None
    do_lower_case       : bool  = None
    do_test             : bool  = None
    do_train            : bool  = None
    eval_all_checkpoints : bool = None
    evaluate_during_training : bool = None
    eval_data_ratio     :float = None
    fp16                : bool = None
    fp16_opt_level      : str = None
    gradient_accumulation_steps : int = None
    k_cross_validation  : int = None
    learning_rate       : float = None
    local_rank          : int = None
    logging_steps       : int = None
    max_grad_norm       : float = None
    max_seq_length      : int = None
    max_steps           : int = None
    model_name_or_path  : str = None
    model_type          : str = None
    multilabel_limiar   : float = None
    no_cuda             : bool = None
    num_labels          : int = None
    num_train_epochs    : int = None
    output_dir          : str = None
    output_mode         : str = None
    overwrite_cache     : bool = None
    overwrite_output_dir : bool = None
    per_gpu_eval_batch_size : int = None
    per_gpu_train_batch_size : int = None
    save_steps          : int = None
    seed                : float = None
    server_ip           : str = None
    server_port         : str = None
    task_name           : str = None
    test_data_ratio     : float = None
    test_selection_mode : str = None
    tokenizer_name      : str = None
    warmup_steps        : int = None
    weight_decay        : float = None
    results_dir         : str = None


    #Online parameters, used to store control parameters that change throughout processing
    it_num              : int = None
    t_total             : int = None
    results_last_eval   : dict = None
    current_checkpoint_step :int = None


class TrainingResources:
    logger      = None
    model       = None
    tokenizer   = None
