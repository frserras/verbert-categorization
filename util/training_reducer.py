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

'''Training Reducer
Auxiliary script that allows you to generate a reduced version
of a pre-existing corpus, through random sampling.

It recieves as arguments:
    - the execution corpus file name [mandatory]
    - the ratio of the corpus that will be used to build the training corpus [optional (standart value 0.7)]
    - the seed to be used for random shuffling the data [optional]
'''

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

train_file_name = sys.argv[1]
ratio = 0.7
if len(sys.argv) > 2:
    ratio = float(sys.argv[2])
seed = 50809545
if len(sys.argv) > 3:
    seed = int(sys.argv[3])

train_df = pd.read_csv(train_file_name)

reduced_df, _ = train_test_split(train_df, test_size=(1.0 - ratio),
                                 random_state=seed)
reduced_df.to_csv(train_file_name.split('.csv')[0] + '__reduced_' + str(100*ratio) + '_'
                  + str(seed)+'__.csv', index=False)

print('Training Set Reduced: ' + str(len(train_df)) +
      ' -> ' + str(len(reduced_df)))
