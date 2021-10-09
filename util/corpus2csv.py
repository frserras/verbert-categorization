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

'''Corpus2CSV
Auxiliary Script that converts the corpus into a .csv file, with the appropriate format
for be processed by verBERT.

Receives a string identifying the corpus as a mandatory argument, and  as optional arguments, 
it recieves 'split', to indicate whether the corpus should be segmented and 
'tree structure' to indicate  whether the conversion should take into account the hierarchical 
structure of the corpus.
'''
import sys
import pickle
import random
import pandas as pd
from tqdm import tqdm
from getpass import getpass
from namedlist import namedlist
from naive_criptography import *
from sklearn.model_selection import train_test_split

SEED = 42

corpus_identifier = sys.argv[1]
tree_structure = 'tree_structure' in sys.argv
split = 'split' in sys.argv

for arg in sys.argv:
    if 'seed' in arg:
        SEED = int(arg.split('=')[1])
random.seed(SEED)


if tree_structure:
    print('ERROR: tree-structure friendly treatment for corpus conversion '
          'not implemented.')
else:
    corpus_file_name = 'koll_corpus_' + corpus_identifier + '.pkl'
    fields_file_name = 'koll_fields_' + corpus_identifier + '.pkl'

    with open(fields_file_name, 'rb') as f:
        fields = pickle.load(f)

    Processo = namedlist('Processo', fields)
    with open(corpus_file_name, 'rb') as f:
        corpus = pickle.load(f)


df_corpus = pd.DataFrame(columns=['Ementa', 'Verbetes'])
for i in tqdm(range(len(corpus))):
    processo = corpus[i]
    verbet_string = ''
    for verb in processo.verbetacao:
        if verb != 'clust_26':  # Removes the 'Others' super-class
            verbet_string = verbet_string + verb + '/'
    if verbet_string != '':   # Removes the 'Others' super-class
        verbet_string = verbet_string[:len(verbet_string)-1]
        df_corpus.loc[i] = [processo.ementa, verbet_string]
df_corpus = df_corpus.sample(frac=1, random_state=random.randint(0, 10000000))
df_corpus.to_csv('koll' + corpus_identifier +
                 '_all'+str(SEED)+'.csv', index=False)

# Splitting test data:
if split:
    df_execution_data, df_test_data = train_test_split(df_corpus, test_size=0.2,
                                                       random_state=random.randint(0, 10000000))
    df_execution_data.to_csv(
        'koll' + corpus_identifier + '_exec'+str(SEED)+'.csv', index=False)
    df_test_data.to_csv('koll' + corpus_identifier +
                        '_test'+str(SEED)+'.csv', index=False)
