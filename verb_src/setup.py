# -*- coding: utf-8 -*
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# This program was made based on the original run_glue.py program with Copyright
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

'''Setup Module
Module that defines several auxiliary functions to setup different structures during processing
'''

import logging
import random

import os
import numpy as np
import torch
import verbert.verb_src.param as param


def setUp(args):
    # Setup CUDA, GPU & distributed training
    setUpPlatform(args)

    res = param.TrainingResources()

    # Setup logging
    setUpLogging(args, res)

    # Set seed
    setUpSeed(args)

    return res


def setUpPlatform(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device


def setUpLogging(args, res):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.local_rank in [-1, 0] else logging.WARN,
    )
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    res.logger = logging.getLogger("sanalysis")
    fh = logging.FileHandler(os.path.join(args.results_dir, 'sanalysis.log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # was logging.ERROR
    # add the handlers to the logger
    res.logger.addHandler(fh)
    res.logger.addHandler(ch)

    res.logger.warning(
        "Starting: Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )


def setUpSeed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
