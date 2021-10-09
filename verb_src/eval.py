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

'''Eval module
Module that defines functions for the evaluation of the trained models.
'''
from typing import Dict, Tuple, Any
import os
import glob
import logging
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import glue_compute_metrics as compute_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    WEIGHTS_NAME, AutoConfig
from verbert.verb_src.multilabel import BertForMultiLabelClassification
from verbert.verb_src.metrics import compute_multilabel_metrics

from verbert.verb_src import train
from verbert.verb_src.param import Parameters, TrainingResources


def evaluate(args: Parameters, res: TrainingResources, prefix: str = "", test: bool = False) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        "mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" \
        else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = train.load_and_cache_examples(
            args, eval_task, res, evaluate=True, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(res.model, torch.nn.DataParallel):
            res.model = torch.nn.DataParallel(res.model)

        # Eval!
        preds, out_label_ids = _eval_batches(
            args, res, eval_dataloader, eval_dataset, prefix)
        eval_task_substitute = "mnli" if eval_task == "aves-ou-computadores" else eval_task

        if args.output_mode == 'multiclassification':
            result = compute_multilabel_metrics(
                eval_task_substitute, preds, out_label_ids)
        else:
            result = compute_metrics(
                eval_task_substitute, preds, out_label_ids)
        results.update(result)
        _report_result(result, res.logger, eval_output_dir,
                       args.results_dir, prefix, args)

    return results


def _eval_batches(args: Parameters, res: TrainingResources,
                  eval_dataloader, eval_dataset, prefix: str) -> Tuple[Any, Any]:
    res.logger.info("***** Running evaluation {} *****".format(prefix))
    res.logger.info("  Num examples = %d", len(eval_dataset))
    res.logger.info("  Batch size = %d", args.eval_batch_size)
    total_eval_loss = 0.0
    n_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        res.model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs, outputs = _apply_model_to_batch(batch, args, res)
        tmp_eval_loss, logits = outputs[:2]
        total_eval_loss += tmp_eval_loss.mean().item()
        n_eval_steps += 1
        if args.output_mode == 'multiclassification':
            logits = logits.sigmoid()
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    elif args.output_mode == 'multiclassification':
        preds = (preds >= args.multilabel_limiar)*1.0
    else:
        preds = np.argmax(preds, axis=1)

    eval_loss = total_eval_loss / n_eval_steps
    res.logger.info("  Eval loss = %d", eval_loss)

    return preds, out_label_ids


def _apply_model_to_batch(batch, args: Parameters, res: TrainingResources) -> Tuple[Dict, Tuple]:
    with torch.no_grad():
        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in [
                    "bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = res.model(**inputs)
    return inputs, outputs


def _extract_step(args, prefix: str) -> (int, str):
    # Extracts the corresponding step from the name prefix of a result's file
    step = None
    suffix = None
    if '_test' in prefix:
        prefix = prefix.split('_')[0]
        if prefix == '':
            step = args.t_total
        else:
            step = int(prefix.split('-')[1])
            if step == 1:
                step = args.current_checkpoint_step
        suffix = 'test'
    elif 'checkpoint' in prefix:
        step = int(prefix.split('-')[1])
        suffix = prefix
    elif 'train' in prefix:
        step = int(prefix.split('_')[1])
        suffix = prefix
    elif prefix == '':
        step = args.t_total
        suffix = 'final_eval'
    else:
        step = None
        suffix = prefix
    return (step, suffix)


def _report_result(result, logger, eval_output_dir: str, results_dir: str, prefix: str, args: Parameters = None) -> None:
    # We complement this function so that it saves the curves through pickle files, for easier manipulation
    output_eval_file = os.path.join(
        eval_output_dir, prefix, "eval_results.txt")
    if not os.path.exists(os.path.abspath(os.path.join(output_eval_file, os.pardir))):
        os.mkdir(os.path.abspath(os.path.join(output_eval_file, os.pardir)))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            key_file = os.path.join(results_dir, 'eval_' + key + '.pkl')
            if os.path.exists(key_file) and os.path.isfile(key_file):
                with open(key_file, 'rb') as kreader:
                    key_story = pickle.load(kreader)
                key_story.append([_extract_step(args, prefix)[
                                 0], result[key], _extract_step(args, prefix)[1]])
                with open(key_file, 'wb') as kwriter:
                    pickle.dump(key_story, kwriter)
            else:
                key_story = [
                    [_extract_step(args, prefix)[0], result[key], _extract_step(args, prefix)[1]]]
                with open(key_file, 'wb') as kwriter:
                    pickle.dump(key_story, kwriter)


def final_eval(args: Parameters, res: TrainingResources) -> Dict:
    results: Dict = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        res.tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        res.logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            if args.output_mode == 'multiclassification':
                res.model = BertForMultiLabelClassification.from_pretrained(
                    checkpoint)
            else:
                res.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint)

            res.model.to(args.device)
            result = evaluate(args, res, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v)
                          for k, v in result.items())
            results.update(result)

    return results
