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

'''Metrics Module
Module that defines how to compute the performance metrics for the multi-label and multiclass case
to be used during the validation and testing stages.
'''
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as basic_scores
from sklearn.metrics import accuracy_score as subset_accuracy
from sklearn.metrics import hamming_loss
from sklearn.metrics import multilabel_confusion_matrix


def compute_multilabel_metrics(task_name, preds, labels):
    """ Function that computes the performance metrics for the multilabel and
    multiclass case.

    Args:
        task_name (str): name of the task being evaluated
        preds (list): list of predicted classes for a set of examples
        labels (list): list of correct classes for the same set of examples

    Returns:
        [dict]: dictionary containig the obtained values for each performance metrics
    """
    assert len(preds) == len(labels)
    prc_mi, rec_mi, f1_mi, _ = basic_scores(labels, preds, average='micro')
    prc_ma, rec_ma, f1_ma, _ = basic_scores(labels, preds, average='macro')
    prc_sm, rec_sm, f1_sm, _ = basic_scores(labels, preds, average='samples')
    hl = hamming_loss(labels, preds)
    mcm = multilabel_confusion_matrix(labels, preds)
    sub_acc = subset_accuracy(labels, preds)
    return {'precision_micro': prc_mi, 'precision_macro': prc_ma,
            'precision_sample': prc_sm,
            'recall_micro': rec_mi, 'recall_macro': rec_ma,
            'recall_sample': rec_sm,
            'f1_micro': f1_mi, 'f1_macro': f1_ma, 'f1_sample': f1_sm,
            'hamming_loss': hl, 'multi_confusion_matrix': mcm,
            'acc': sub_acc,
            'preds': preds,
            'labels': labels}
