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

'''Example of the main code for the execution of a verbert experiment.
It loads the experiment's parameters and execute the full cycle of
experimentation with it (loading, training, saving, evaluation and testing)
'''

import sys
import random
import os
# sys.path.append('') # You may need to add the verbert folder path here
from expBetaB_025_52_4e5__param import exp_param_extractor
from verbert.verb_src import setup
from verbert.verb_src import loader
from verbert.verb_src import train
from verbert.verb_src import save
from verbert.verb_src import eval
from verbert.verb_src import test
from verbert.verb_src import util


def main():
    # get paramenters
    args = exp_param_extractor()
    if not args.data_from_main or not isinstance(args.k_cross_validation, int):
        args.k_cross_validation = 1

    random.seed(args.seed)
    original_output_dir = args.output_dir
    original_results_dir = args.results_dir
    for it in range(args.k_cross_validation):
        for _ in range(it+1):
            seed = int(random.random()*1000000)
        it_files = util.data_splitter(args, it, seed)
        args.data_train_file, args.data_eval_file, args.data_test_file = it_files
        args.output_dir = os.path.join(original_output_dir, 'it_' + str(it))
        args.results_dir = os.path.join(original_results_dir, 'it_' + str(it))
        args.it_num = it
        # setup
        res = setup.setUp(args)

        # Load pretrained model and tokenizer
        res.model, res.tokenizer = loader.loadModelAndTokenizer(args)
        # Training
        args.do_train = True
        train.do_train(args, res)
        # Saving best-practices: if you use defaults names for the model,
        #   you can reload it using from_pretrained()
        save.save(args, res)
        # Evaluation
        eval.final_eval(args, res)
        test.final_test(args, res)
    print('\a')


if __name__ == "__main__":
    main()
