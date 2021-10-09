# -*- coding: utf-8 -*-
# This program was made by Felipe Serras as part of his Master's degree,
# under the guidance of Prof. Marcelo Finger. All rights reserved.
# We tried to make explicit all our references and all the works on which ours is based.
# Please contact us if you encounter any problems in this regard.
# If not stated otherwise, this software is licensed under the Apache License, Version 2.0 (the "License");
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

"""Baselines

Auxiliary script that provides two baseline methods to be computed on
pre-existing corpus, returning the performance metrics obtained by
the methods.

Recieves as argument the baseline method, that can be random_over_prob or
const_most_prob_5
"""

import pandas as pd
import numpy as np
import sys
import random
from metrics import compute_multilabel_metrics

mode = sys.argv[1]

train_set = pd.read_csv('kollKM50_lvl0_exec2938294_train4589824.csv')
test_set = pd.read_csv('kollKM50_lvl0_test2938294.csv')
test_size = test_set['Verbetes'].size

labels_prob_distr = np.zeros(26)
labels_positive_occ = np.zeros(26)
labels_total = 0
for raw_classes in train_set['Verbetes']:
    labels_total = labels_total + 1
    classes = raw_classes.split('/')
    for i in range(26):
        if i == 25:
            clust_id = 'clust_26'
        else:
            clust_id = 'clust_' + str(i)

        if clust_id in classes:
            labels_prob_distr[i] = labels_prob_distr[i]+1

labels_prob_distr = labels_prob_distr / labels_total

max_indexes = labels_prob_distr.argsort()[-5:][::-1]


print()

predicitions = []
if mode == 'random_over_prob':
    # baseline method of classifying with a random classification, generated
    # using the probability distribution of classes in the training set.
    for i in range(test_size):
        prediction_i = []
        for class_prob in labels_prob_distr:
            prediction_i.append(1.0 * (random.random() <= class_prob))
        predicitions.append(prediction_i)
    print(len(predicitions))
elif mode == 'const_most_prob_5':
    # baseline method of classifying with a standard classification,
    # containing the n=5 most likely classes.
    prediction = np.zeros(26)
    for i in max_indexes:
        prediction[i] = 1
    for i in range(test_size):
        predicitions.append(prediction)

print('pred')
print(predicitions[0:5])

labels = []
for raw_classes in test_set['Verbetes']:
    label_i = np.zeros(26)
    classes = raw_classes.split('/')
    for i in range(26):
        if i == 25:
            clust_id = 'clust_26'
        else:
            clust_id = 'clust_' + str(i)

        if clust_id in classes:
            id_ = int(clust_id.split('_')[1])
            if id_ == 26:
                id_ = 25
            label_i[id_] = 1.0
    labels.append(label_i)

predicitions = np.array(predicitions)
labels = np.array(labels)


results = compute_multilabel_metrics('_', predicitions, labels)
for i in results.keys():
    if i != 'multi_confusion_matrix' and i != 'preds' and i != 'labels':
        print(str(results[i]))
