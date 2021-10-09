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

''' Util Module
Module created to gather utility tools that would not fit in any of the other modules.
'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from verbert.verb_src.param import Parameters

#


def data_splitter(args: Parameters, it_num: int, seed: int):
    """Function that segments a data set into training, validation, and testing subsets.

    Args:
        args (Parameters): experiment parameters
        it_num (int): number of current iteration
        seed (int): seed for random data segmentation 

    Returns:
        [tuple]: addresses of the generated training, validation and test files 
    """
    if not args.data_from_main:
        return (args.data_train_file, args.data_eval_file, args.data_test_file)
    else:
        it_data_dir = os.path.join(args.data_dir, 'it_'+str(it_num))
        if not os.path.exists(it_data_dir):
            os.mkdir(it_data_dir)
        all_data = pd.read_csv(os.path.join(
            args.data_dir, args.data_main_file))
        train_data = all_data
        eval_data = None
        test_data = None
        if args.do_eval:
            train_data, eval_data = train_test_split(all_data, test_size=args.eval_data_ratio,
                                                     random_state=seed)
        if args.do_test:
            updated_test_ratio = args.test_data_ratio * \
                len(all_data)/len(train_data)
            train_data, test_data = train_test_split(train_data, test_size=updated_test_ratio,
                                                     random_state=seed)
        train_it_file = os.path.join('it_'+str(it_num), 'train.csv')
        train_data.to_csv(os.path.join(
            args.data_dir, train_it_file), index=False)
        eval_it_file = None
        if isinstance(eval_data, pd.core.frame.DataFrame):
            eval_it_file = os.path.join('it_'+str(it_num), 'eval.csv')
            eval_data.to_csv(os.path.join(
                args.data_dir, eval_it_file), index=False)
        test_it_file = None
        if isinstance(test_data, pd.core.frame.DataFrame):
            test_it_file = os.path.join('it_'+str(it_num), 'test.csv')
            test_data.to_csv(os.path.join(
                args.data_dir, test_it_file), index=False)

        return (train_it_file, eval_it_file, test_it_file)
