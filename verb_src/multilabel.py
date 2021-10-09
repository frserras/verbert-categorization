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

'''Multilabel module
Module that defines the class BertForMultiLabelClassification , for multilabel 
and multiclass classification. The code for this module is partially based on the examples
contained  in https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html
and on the tutorial contained in https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
'''

from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


class BertForMultiLabelClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        # output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
