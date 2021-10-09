# -*- coding: utf-8 -*-
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

'''Save Module
Module that provides a way to save a trained model to disk
'''
import os
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from verbert.verb_src.multilabel import BertForMultiLabelClassification


from verbert.verb_src.param import Parameters, TrainingResources


def save(args: Parameters, res: TrainingResources) -> None:
    """ Saving best-practices: if you use defaults names for the model, 
        you can reload it using from_pretrained() 
    """
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        res.logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            res.model.module if hasattr(res.model, "module") else res.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        res.tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        if args.output_mode == 'multiclassification':
            res.model = BertForMultiLabelClassification.from_pretrained(
                args.output_dir)
        else:
            res.model = AutoModelForSequenceClassification.from_pretrained(
                args.output_dir)

        res.tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        res.model.to(args.device)
