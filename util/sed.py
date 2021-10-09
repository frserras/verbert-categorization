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

'''SED (Split Execution Data)
Auxiliar script to split the execution corpus

It receives three named arguments: 
    - corpus: the corpus file name
    - ratio: the size of the portion of the corpus that will be separated for validation
    - seed: seed to be used for random shuffling the data 
'''

import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split

corpus_file_name = ''
validation_ratio = 0.1
rand_seed = 42

for arg in sys.argv:
    if 'corpus' in arg:
        corpus_file_name = arg.split('=')[1]
    elif 'ratio' in arg:
        validation_ratio = float(arg.split('=')[1])
    elif 'seed' in arg:
        rand_seed = int(arg.split('=')[1])
        random.seed(rand_seed)

df_exec_corpus = pd.read_csv(corpus_file_name)
df_train_corpus, df_validation_corpus = train_test_split(df_exec_corpus, test_size=validation_ratio,
                                                         random_state=random.randint(0, 10000000))

df_train_corpus.to_csv(corpus_file_name.split(
    '.')[0]+'_train'+str(rand_seed)+'.csv', index=False)
df_validation_corpus.to_csv(corpus_file_name.split(
    '.')[0]+'_eval'+str(rand_seed)+'.csv', index=False)
