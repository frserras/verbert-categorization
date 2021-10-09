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

'''Training Module
Module responsible for providing tools for training a pre-trained model.
'''
from typing import Tuple, Dict

import os
import json
import pickle
import shutil

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (
    #    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    #    WEIGHTS_NAME,
    AdamW,
    #    AutoConfig,
    AutoModelForSequenceClassification,
    #    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LambdaLR


from verbert.verb_src import loader
from verbert.verb_src import setup
from verbert.verb_src import eval as saeval
from verbert.verb_src.param import Parameters, TrainingResources


def _get_t_total(args, train_dataloader):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    return t_total


def _save_optimizer_and_scheduler(args: Parameters, optimizer: AdamW, scheduler: LambdaLR) -> None:
    """ Check if saved optimizer or scheduler states exist """
    if (os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and
            os.path.isfile(os.path.join(
                args.model_name_or_path, "scheduler.pt"))
        ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))


def _optimize_fp16(args: Parameters, res: TrainingResources, optimizer: AdamW) -> None:
    if args.fp16:
        try:
            from apex import amp
            args.amp_imported = True
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            res.model, optimizer, opt_level=args.fp16_opt_level)


def _prepare_optimizer_and_scheduler(args: Parameters,
                                     res: TrainingResources,
                                     t_total: int) -> Tuple[AdamW, LambdaLR]:
    """ repare optimizer and schedule (linear warmup and decay) """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in res.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in res.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
         },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    _save_optimizer_and_scheduler(args, optimizer, scheduler)
    _optimize_fp16(args, res, optimizer)
    return optimizer, scheduler


def _activate_training(args: Parameters,
                       res: TrainingResources,
                       train_dataset: TensorDataset) -> Tuple[DataLoader, AdamW, LambdaLR]:
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (RandomSampler(train_dataset) if args.local_rank == -1 else
                     DistributedSampler(train_dataset))
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = _get_t_total(args, train_dataloader)
    args.t_total = t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = _prepare_optimizer_and_scheduler(args, res, t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(res.model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger = res.logger
    logger.info("***** Running training *****")
    logger.info("  Device = %s", args.device.type)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    return train_dataloader, optimizer, scheduler


def _check_continuing_from_checkpoint(args: Parameters,
                                      res: TrainingResources,
                                      train_dataloader: TensorDataset) -> Tuple[int, int, int]:
    """ Check if continuing training from a checkpoint """
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

        logger = res.logger
        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)
    return global_step, epochs_trained, steps_trained_in_current_epoch


def _compute_batch_loss(args: Parameters,
                        model: AutoModelForSequenceClassification,
                        batch) -> torch.Tensor:
    model.train()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {"input_ids": batch[0],
              "attention_mask": batch[1], "labels": batch[3]}
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch[2] if args.model_type in [
                "bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    outputs = model(**inputs)
    # model outputs are always tuple in transformers (see doc)
    return outputs[0]


def _scale_loss_and_backpropagate(args, loss, optimizer) -> None:
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    if args.fp16:
        if not args.amp_imported:
            from apex import amp
            args.amp_imported = True
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


def _is_last_step_in_batch(step: int, args: Parameters, epoch_iterator: int) -> bool:
    return (step + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        len(epoch_iterator) <= args.gradient_accumulation_steps
        and (step + 1) == len(epoch_iterator)
    )


def _evaluate_results(args: Parameters, res: TrainingResources, logs: Dict, prefix: str = "") -> None:
    """ Only evaluate when single GPU otherwise metrics may not average well """
    if args.local_rank == -1 and args.evaluate_during_training:
        # res.model.eval()
        results = saeval.evaluate(args, res, prefix=prefix)
        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value
        res.model.zero_grad()
        # res.model.train()
        return results


def _save_model_checkpoint(args: Parameters,
                           res: TrainingResources,
                           global_step: int,
                           optimizer: AdamW,
                           scheduler: LambdaLR,
                           eval_results: dict) -> None:
    # We have implemented different ways to save intermediate models, which may be more appropriate for
    # hardware with less disk space:
    if 'max' in args.checkpointing_mode:
        output_dir = os.path.join(args.output_dir, 'checkpoint-1')
        checkpoint_criterion = args.checkpointing_mode.split('/')[1]
        if args.results_last_eval == None or eval_results[checkpoint_criterion] > args.results_last_eval[checkpoint_criterion]:

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            if hasattr(res.model, "module"):
                print('>>>hasattr')
            # Takes care of distributed/parallel training
            model_to_save = res.model.module if hasattr(
                res.model, "module") else res.model
            model_to_save.save_pretrained(output_dir)
            res.tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            res.logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(
                output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(
                output_dir, "scheduler.pt"))
            res.logger.info(
                "Saving optimizer and scheduler states to %s", output_dir)
            with open(os.path.join(output_dir, 'corresponding_step.txt'), 'w') as f:
                f.write(str(global_step))
            args.current_checkpoint_step = global_step
            args.results_last_eval = eval_results

    else:
        output_dir = os.path.join(
            args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Takes care of distributed/parallel training
        model_to_save = res.model.module if hasattr(
            res.model, "module") else res.model
        model_to_save.save_pretrained(output_dir)
        res.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        res.logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(
            output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(
            output_dir, "scheduler.pt"))
        res.logger.info(
            "Saving optimizer and scheduler states to %s", output_dir)


def _treat_last_step_in_batch(args: Parameters,
                              res: TrainingResources,
                              global_step: int,
                              optimizer: AdamW,
                              scheduler: LambdaLR,
                              logging_loss: float,
                              tr_loss: float) -> float:
    # tb_writer    :SummaryWriter) -> float:
    if args.fp16:
        if not args.amp_imported:
            from apex import amp
            args.amp_imported = True
        torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer), args.max_grad_norm)
    else:
        torch.nn.utils.clip_grad_norm_(
            res.model.parameters(), args.max_grad_norm)

    optimizer.step()
    scheduler.step()  # Update learning rate schedule
    res.model.zero_grad()

    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
        logs = {}
        eval_results = _evaluate_results(
            args, res, logs, prefix='train_'+str(int(global_step)))
        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
        learning_rate_scalar = scheduler.get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar
        logs["loss"] = loss_scalar
        logging_loss = tr_loss

        train_stats_file = os.path.join(args.results_dir, 'train_stats.pkl')
        if os.path.exists(train_stats_file) and os.path.isfile(train_stats_file):
            with open(train_stats_file, "rb") as tsf:
                train_stats = pickle.load(tsf)
            train_stats.append(
                [global_step, logs["learning_rate"], logs["loss"]])
            with open(train_stats_file, "wb") as tsf:
                pickle.dump(train_stats, tsf)
        else:
            train_stats = [[global_step, logs["learning_rate"], logs["loss"]]]
            with open(train_stats_file, "wb") as tsf:
                pickle.dump(train_stats, tsf)

    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
        # Save model checkpoint
        _save_model_checkpoint(args, res, global_step,
                               optimizer, scheduler, eval_results)
        # setup.setUpSeed(args)

    return logging_loss


def train(args, train_dataset, model, tokenizer, res):
    """ Train the model """

    train_dataloader, optimizer, scheduler = _activate_training(
        args, res, train_dataset)
    global_step, epochs_trained, steps_trained_in_current_epoch = \
        _check_continuing_from_checkpoint(args, res, train_dataloader)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    setup.setUpSeed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            loss = _compute_batch_loss(args, model, batch)
            _scale_loss_and_backpropagate(args, loss, optimizer)
            tr_loss += loss.item()

            if _is_last_step_in_batch(step, args, epoch_iterator):
                global_step += 1
                logging_loss = _treat_last_step_in_batch(args, res, global_step, optimizer,
                                                         scheduler, logging_loss, tr_loss)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def _form_cache_file_name(args: Parameters, task: str, evaluate: bool, test: bool) -> str:
    mode = None
    if evaluate and test:
        mode = 'test'
    elif evaluate:
        mode = 'eval'
    else:
        mode = 'train'
    return os.path.join(
        args.data_dir, "it_"+str(args.it_num),
        "cached_{}_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )


def _load_cached_features_file(file_name: str,
                               res: TrainingResources) -> loader.InputFeatures:
    """ Load data features from cached file """
    res.logger.info("Loading features from cached file %s", file_name)
    return torch.load(file_name)


def _load_dataset(args: Parameters,
                  res: TrainingResources,
                  task: str,
                  evaluate: bool,
                  output_mode: str,
                  cache_file: str,
                  test: bool = False) -> loader.InputFeatures:
    """ Load data features from dataset file """
    res.logger.info("Creating features from dataset file at %s", args.data_dir)
    processor = loader.processors[task]()
    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    if evaluate and test:
        examples = processor.get_test_examples(args)
    elif evaluate and not test:
        examples = processor.get_eval_examples(args)
    else:
        examples = processor.get_train_examples(args)

    features = loader.convert_examples_to_features(
        examples,
        label_list,
        args.max_seq_length,
        res.tokenizer,
        output_mode,
        # pad on the left for xlnet
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=res.tokenizer.convert_tokens_to_ids(
            [res.tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    if args.local_rank in [-1, 0]:
        res.logger.info("Saving features into cached file %s", cache_file)
        torch.save(features, cache_file)
    return features


def _build_dataset(features: loader.InputFeatures, output_mode) -> TensorDataset:
    """ Convert to Tensors and build dataset """
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
    elif output_mode == "multiclassification":
        all_labels = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def load_and_cache_examples(args: Parameters,
                            task: str,
                            res: TrainingResources,
                            evaluate: bool = False,
                            test: bool = False) -> TensorDataset:
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    else:
        res.logger.warning("Logger and tokenizer set for loading")

    output_mode = loader.output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = _form_cache_file_name(args, task, evaluate, test)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = _load_cached_features_file(cached_features_file, res)
    else:
        features = _load_dataset(
            args, res, task, evaluate, output_mode, cached_features_file, test)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    dataset = _build_dataset(features, output_mode)
    return dataset


train_dataset = None


def do_train(args: Parameters, res: TrainingResources) -> None:
    global train_dataset
    res.logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, res, evaluate=False)
        res.logger.info("Dataset Loaded")
        global_step, tr_loss = train(
            args, train_dataset, res.model, res.tokenizer, res)
        res.logger.info(" global_step = %s, average loss = %s",
                        global_step, tr_loss)
