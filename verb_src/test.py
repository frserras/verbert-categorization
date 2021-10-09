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
'''Test Module
Module that provides functions for carrying out the final tests of a trained model.
'''

from verbert.verb_src.param import Parameters, TrainingResources
from typing import Dict, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from verbert.verb_src.multilabel import BertForMultiLabelClassification
from verbert.verb_src.eval import evaluate
import os
import pickle

#


def _test_checkpoint_selector(args: Parameters) -> str:
    """Auxiliary function that selects the correct intermediate model to perform the tests,
    according to a metric of preference, encompassing different forms of selection, more or less appropriate
    according to the specific needs of the hardware used.

    Args:
        args (Parameters): experiment parameters

    Returns:
        str: intermediate model address selected for testing 
    """
    checkpoint = ' '
    if args.test_selection_mode == 'max_acc':
        eval_acc_file = os.path.join(args.results_dir, 'eval_acc.pkl')
        with open(eval_acc_file, 'rb') as evacc_reader:
            checkpoints = pickle.load(evacc_reader)
        max_acc = 0.0
        for i in range(len(checkpoints)):
            if checkpoints[i][1] > max_acc:
                max_acc = checkpoints[i][1]
                checkpoint = checkpoints[i][2]
                if checkpoint == 'final_eval':
                    checkpoint = ''
    elif args.test_selection_mode == 'max_f1_micro':
        eval_acc_file = os.path.join(args.results_dir, 'eval_f1_micro.pkl')
        with open(eval_acc_file, 'rb') as evacc_reader:
            checkpoints = pickle.load(evacc_reader)
        max_acc = 0.0
        for i in range(len(checkpoints)):
            if checkpoints[i][1] > max_acc:
                max_acc = checkpoints[i][1]
                checkpoint = checkpoints[i][2]
                if checkpoint == 'final_eval':
                    checkpoint = ''
    elif args.test_selection_mode == 'last_step':
        checkpoint = ''
    elif args.test_selection_mode == '1checkpoint':
        if 'max/' in args.checkpointing_mode:
            chckpt_criterion = args.checkpointing_mode.split('/')[1]
            eval_file = os.path.join(
                args.results_dir, 'eval_' + chckpt_criterion + '.pkl')
            with open(eval_file, 'rb') as eval_reader:
                checkpoints = pickle.load(eval_reader)
            with open(os.path.join(args.output_dir, 'checkpoint-1', 'corresponding_step.txt'), 'r') as f:
                max_step = int(f.readline())
            for cpt in checkpoints:
                if cpt[2] == 'final_eval':
                    final_eval_performance = cpt[1]
                if cpt[0] == max_step:
                    max_eval_performance = cpt[1]
            if final_eval_performance > max_eval_performance:
                checkpoint = ''
            else:
                checkpoint = 'checkpoint-1'
        else:
            checkpoint = 'checkpoint-1'
    return checkpoint


def final_test(args: Parameters, res: TrainingResources) -> Dict:
    """Function that performs the final set of tests on the selected model, 
    in order to calculate the overall performance obtained:

    Args:
        args (Parameters): experiment parameters
        res (TrainingResources): experiment training resources

    Returns:
        Dict: performance results obtained during testing
    """
    results: Dict = {}
    if args.do_test and args.local_rank in [-1, 0]:
        res.tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = _test_checkpoint_selector(args)
        checkpoint = os.path.join(args.output_dir, checkpoint)
        global_step = checkpoint.split("-")[-1]
        prefix = checkpoint.split(
            "/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        if args.output_mode == 'multiclassification':
            res.model = BertForMultiLabelClassification.from_pretrained(
                checkpoint)
        else:
            res.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint)
        res.model.to(args.device)
        result = evaluate(args, res, prefix=prefix+'_test', test=True)
        result = dict((k + "_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)
    return results
