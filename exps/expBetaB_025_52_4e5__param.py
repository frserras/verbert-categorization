# -*- coding: utf-8 -*-
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

# Example of the main code to establish the parameters values for
# the execution of a verbert experiment.
import sys
# sys.path.append('') # You may need to add the verbert folder path here
from verbert.verb_src.param import Parameters

def exp_param_extractor():
    exp_param = Parameters()

    exp_param.adam_epsilon        : float =1e-08
    exp_param.amp_imported        : bool  =False
    exp_param.cache_dir           : str   ='~/verbert/_cache'
    exp_param.checkpointing_mode  : str   ='max/f1_micro'
    exp_param.config_name         : str   =''
    exp_param.data_dir            : str   = '~/verbert/verb_data/' 
    exp_param.device                      ='cuda'
    exp_param.do_eval             : bool  =True
    exp_param.do_lower_case       : bool  =False
    exp_param.do_train            : bool  =True
    exp_param.eval_all_checkpoints : bool =False
    exp_param.evaluate_during_training : bool =True
    exp_param.fp16                : bool =False
    exp_param.fp16_opt_level      : str ='O1'
    exp_param.gradient_accumulation_steps : int =10
    exp_param.learning_rate       : float =4e-5
    exp_param.local_rank          : int =-1
    exp_param.logging_steps       : int =10
    exp_param.max_grad_norm       : float =1.0
    exp_param.max_seq_length      : int =52
    exp_param.max_steps           : int =-1
    exp_param.model_name_or_path  : str ='neuralmind/bert-large-portuguese-cased'
    exp_param.model_type          : str ='bert'
    exp_param.no_cuda             : bool =False
    exp_param.num_labels          : int = 25
    exp_param.num_train_epochs    : int = 10
    exp_param.output_dir          : str = '~/verbert/models/BetaB_025_52_4e5/'
    exp_param.output_mode         : str ='multiclassification'
    exp_param.overwrite_cache     : bool =False
    exp_param.overwrite_output_dir : bool =True
    exp_param.per_gpu_eval_batch_size : int =32
    exp_param.per_gpu_train_batch_size : int =32
    exp_param.save_steps          : int =10
    exp_param.seed                : float =8756152
    exp_param.server_ip           : str =''
    exp_param.server_port         : str =''
    exp_param.task_name           : str ='koll_KM50_co1000_noOthers'
    exp_param.tokenizer_name      : str ='neuralmind/bert-large-portuguese-cased'
    exp_param.warmup_steps        : int =50
    exp_param.weight_decay        : float =0.01
    exp_param.results_dir         : str ='~/verbert/exps/results_BetaB_025_52_4e5'
    exp_param.do_test             : bool=True
    exp_param.test_selection_mode : str='1checkpoint'
    exp_param.data_from_main      : bool= False
    exp_param.data_main_file      : str= None
    exp_param.data_train_file     : str= 'kollKM50co1000_lvl0_exec14251416_train8975498.csv'
    exp_param.data_eval_file      : str= 'kollKM50co1000_lvl0_exec14251416_eval8975498.csv'
    exp_param.data_test_file      : str= 'kollKM50co1000_lvl0_test14251416.csv'
    exp_param.k_cross_validation  : int=1
    exp_param.eval_data_ratio     : float= None
    exp_param.test_data_ratio     : float= None
    exp_param.multilabel_limiar   : float=0.25

    return exp_param
