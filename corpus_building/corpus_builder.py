
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

"""Corpus Builder
This script builds a processable corpus from the original data, containing real 
verbetation examples and arranged in a csv file named koll2.csv.

It asks the user to provide a password and the the generated corpus, ecrypted 
with this password, is saved in the files koll_corpus_01.pkl and koll_fields_01.pkl' 
"""
import csv
from namedlist import namedlist
from bs4 import BeautifulSoup as BSoup
from tqdm import tqdm
import re
from naive_criptography import *  #
import pickle
from getpass import getpass
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

print('VERBERT PROJECT')
print('|=> Corpus Builder --v=0.1')
# get a password to encrypt the final corpus
password = getpass('Enter Password:')
password2 = getpass('Repeat Password:')
if password != password2:
    print("    |==> Passwords don't match! Try Again.")
else:
    corpus_file_name = 'koll_corpus_01.pkl'
    fields_file_name = 'koll_fields_01.pkl'

    print('    |==> Building Corpus...')
    corpus = []
    with open('koll2.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';', quotechar='"')
        row_count = 0
        for row in tqdm(csv_reader, desc='         ', total=26402):
            # set the corpus fields from those available in the original data:
            if row_count == 0:
                len0 = len(row)
                fields = ''
                verbetation_index = row.index('verbetacao')
                abstract_index = row.index('ementa')
                for field in row:
                    fields = fields + ' ' + field
                Processo = namedlist('Processo', fields)
            else:
                if len(row) < 15:
                    row.append('')  # corrects the abscence of a field:
                if row[verbetation_index] != row[abstract_index]:
                    for i in range(len(row)):
                        # unify spacing, remove textual noise,
                        # collapse names of legal entities and
                        # adjust text case:
                        row[i] = BSoup(row[i], "html.parser").text
                        row[i] = row[i].replace('\xa0', ' ')
                        row[i] = row[i].replace('\\n', ' ')
                        row[i] = row[i].replace('\t', ' ')
                        row[i] = row[i].replace('/p>', ' ')
                        row[i] = re.sub(' {2,}', ' ', row[i])
                        row[i] = row[i].lower()
                        row[i] = row[i].replace('cg n.º', 'cg nº')
                        row[i] = row[i].replace('lei n.º', 'lei nº')
                        row[i] = row[i].replace('cg n.', 'cg nº')
                        row[i] = row[i].replace('lei n.', 'lei nº')
                        row[i] = re.sub('[\\\\]+', '', row[i])
                        row[i] = re.sub("""[\']+""", '', row[i])

                        # unify verbets segmentation symbols and split verbets:
                        if i == verbetation_index:
                            row[i] = re.sub(
                                r'([^ \-\.\–]) ([^ \-\.\–])', r'\1_\2', row[i])
                            row[i] = row[i].replace(' ', '')
                            row[i] = re.split(
                                '(?<!proc)(?<!art)(?<!fls)(?<!\d)\.|(?<!art)\.(?=[0-9]+[ºª°]{1})|(?<=\d)\.(?=\D)',
                                row[i])
                            for j in row[i]:
                                if j == '_' or j == ' ' or j == '':
                                    row[i].remove(j)

                    corpus.append(Processo(*row))

            row_count += 1

    ne = Naive_Criptography()
    print('    |==> Filtering Corpus...')
    # filter entries that do not have verbetacao or ementa:
    filtered_corpus = []
    for processo in tqdm(corpus, desc='         '):
        if len(processo.verbetacao) != 0 and len(processo.ementa) != 0:
            filtered_corpus.append(processo)
    corpus = filtered_corpus

    print('    |==> Encrypting Corpus...')
    # encrypt and save the resulting corpus:
    corpus = ne.naive_corpus_encryption(password, corpus)
    with open(corpus_file_name, 'wb') as f:
        pickle.dump(corpus, f)

    with open(fields_file_name, 'wb') as f:
        pickle.dump(fields, f)

    print('    |==> Encrypted Corpus saved @ ' +
          corpus_file_name + ' & ' + fields_file_name)
