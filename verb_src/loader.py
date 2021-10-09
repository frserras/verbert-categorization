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

'''Loader Module
Module that brings together several functions for loading models, tokenizers and data sets
'''

from __future__ import absolute_import, division, print_function

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from verbert.verb_src.multilabel import BertForMultiLabelClassification
import csv
import logging
import os
import sys
from io import open


from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def loadModelAndTokenizer(args):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        # finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = None
    # We added the multiclassification option, that uses our BertForMultiLabelClassification model.
    if args.output_mode == 'multiclassification':
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,  # multi_config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return model, tokenizer


logger = logging.getLogger(__name__)
csv.field_size_limit(2147483647)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and eval examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        return 'InputExample({})'.format(self.__dict__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __str__(self):
        return 'InputFeature({})'.format(self.__dict__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, args):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_eval_examples(self, args):
        """Gets a collection of `InputExample`s for the eval set."""
        raise NotImplementedError()

    def get_test_examples(self, args):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)  # , delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            pop = lines.pop(0)
            return lines


class RAProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def get_train_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, os.path.basename(args.data_train_file))), "train")

    def get_eval_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, os.path.basename(args.data_eval_file))), "eval")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, os.path.basename(args.data_test_file))), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]  # Error line[3] l 146
            label = line[1]
            input_example = InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label)
            # if i < 5:
            #     print("ie::",input_example)
            examples.append(input_example)
        return examples


class ACProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def get_train_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_train_file)), "train")

    def get_eval_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_eval_file)), "eval")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["Aves", "Computadores"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]  # Error line[3] l 146
            label = line[1]
            input_example = InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label)
            # if i < 5:
            #     print("ie::",input_example)
            examples.append(input_example)
        return examples


class AMCProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def get_train_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_train_file)), "train")

    def get_eval_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_eval_file)), "eval")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["Aves", "Computadores", "Música"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1].split('/')
            input_example = InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label)
            examples.append(input_example)
        return examples


class KollKM50Processor(DataProcessor):
    """Processor for our first verbetation corpus (Corpus 1)"""

    def get_train_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_train_file)), "train")

    def get_eval_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_eval_file)), "eval")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_test_file)), "test")

    def get_labels(self):
        """See base class."""
        labels = ['clust_' + str(i) for i in range(27)]
        labels.remove('clust_25')
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1].split('/')
            input_example = InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label)
            examples.append(input_example)
        return examples


class KollKM50co1000Processor(DataProcessor):
    """Processor for our second verbetation corpus (Corpus 2)"""

    def get_train_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_train_file)), "train")

    def get_eval_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_eval_file)), "eval")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(args.data_dir, args.data_test_file)), "test")

    def get_labels(self):
        """See base class."""
        labels = ['clust_' + str(i) for i in range(25)]
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1].split('/')
            input_example = InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label)
            examples.append(input_example)
        return examples


def convert_example_to_feature(example_row,
                               pad_token=0,
                               sequence_a_segment_id=0,
                               sequence_b_segment_id=1,
                               cls_token_segment_id=1,
                               pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    (example, label_map, max_seq_length, tokenizer, output_mode,
     cls_token_at_end, cls_token, sep_token, cls_token_segment_id,
     pad_on_left, pad_token_segment_id) = example_row
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1]
                      * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    elif output_mode == "multiclassification":
        label_id = [0]*len(label_map)  # in this case each label id is a list.
        for label in example.label:
            label_id[label_map[label]] = 1
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 output_mode,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    examples = [(example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token,
                 sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]

    process_count = cpu_count() - 2

    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=100),
                             total=len(examples)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# Processors map defines the correct processor for each task
processors = {
    "aves-ou-computadores": ACProcessor,
    "a-m-c": AMCProcessor,
    "koll_KM50": KollKM50Processor,
    "koll_KM50_co1000_noOthers": KollKM50co1000Processor
}

# Output modes map defines the correct output mode for each task
output_modes = {
    "aves-ou-computadores": "classification",
    "a-m-c": "multiclassification",
    "koll_KM50": "multiclassification",
    "koll_KM50_co1000_noOthers": "multiclassification"
}

# This map defines the correct number of labels for each task
GLUE_TASKS_NUM_LABELS = {
    "aves-ou-computadores": 2,
    "a-m-c": 3,
    "koll_KM50": 26,
    "koll_KM50_co1000_noOthers": 25

}
