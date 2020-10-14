# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""BERT finetuning runner.
===================================

This is a simplified version 
of implementation which uses
session instead of estimator

===================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_v0 as modeling
import optimization_v0 as optimization
import tokenization
import tensorflow as tf
import horovod.tensorflow as hvd
import time
from utils.utils import LogTrainRunHook, OomReportingHook
import utils.dllogger_class
from dllogger import Verbosity
from utils.create_glue_data import *
import numpy as np
import tf_metrics

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("dllog_path", "/results/bert_dllog.json",
                    "filename where dllogger writes to")

flags.DEFINE_string("optimizer_type", "lamb", "Optimizer type : adam or lamb")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_bool("use_trt", False, "Whether to use TF-TRT")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("display_loss_steps", 10,
                     "How often to print loss from estimator")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer(
    "num_accumulation_steps", 1,
    "Number of accumulation steps before gradient update"
    "Global batch size = num_accumulation_steps * train_batch_size")
flags.DEFINE_bool(
    "amp", True,
    "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS."
)
flags.DEFINE_bool("use_xla", True, "Whether to enable XLA JIT compilation.")
flags.DEFINE_bool("horovod", False,
                  "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


def file_based_input(input_file,
                     batch_size,
                     seq_length,
                     drop_remainder,
                     hvd=None):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    # create dataset

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)

    # split across processes
    d = d.shard(hvd.size(), hvd.rank())
    d = d.map(lambda record: _decode_record(record, name_to_features))
    d = d.repeat()
    d = d.shuffle(buffer_size=100)
    d = d.batch(batch_size)

    iterator = d.make_one_shot_iterator()
    example = iterator.get_next()

    return example


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings,
                               compute_type=tf.float32)

    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels],
                                  initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias, name='cls_logits')
        probabilities = tf.nn.softmax(logits,
                                      axis=-1,
                                      name='cls_probabilities')
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(
            one_hot_labels * log_probs, axis=-1, name='cls_per_example_loss')
        loss = tf.reduce_mean(per_example_loss, name='cls_loss')

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(task_name,
                     bert_config,
                     num_labels,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_one_hot_embeddings,
                     hvd=None):
    """Returns `model_fn` closure for Estimator."""
    def model_fn(features, labels):  # pylint: disable=unused-argument
        """The `model_fn` for Estimator."""
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" %
                                      (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        (total_loss, per_example_loss, logits,
         probabilities) = create_model(bert_config, True, input_ids,
                                       input_mask, segment_ids, label_ids,
                                       num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if FLAGS.verbose_logging:
            tf.compat.v1.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s",
                                          var.name, var.shape, init_string)

        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, hvd, False,
                                                 FLAGS.amp,
                                                 FLAGS.num_accumulation_steps,
                                                 FLAGS.optimizer_type)
        return train_op, total_loss

    return model_fn


def main(_):
    # causes memory fragmentation for bert leading to OOM
    if os.environ.get("TF_XLA_FLAGS", None) is not None:
        os.environ["TF_XLA_FLAGS"] += " --tf_xla_enable_lazy_compilation false"
    else:
        os.environ["TF_XLA_FLAGS"] = " --tf_xla_enable_lazy_compilation false"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    dllogging = utils.dllogger_class.dllogger_class(FLAGS.dllog_path)

    if FLAGS.horovod:
        hvd.init()

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()
    processor = MnliProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                           do_lower_case=FLAGS.do_lower_case)

    tf.compat.v1.logging.info("Multi-GPU training with TF Horovod")
    tf.compat.v1.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(),
                              hvd.rank())
    global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps * hvd.size(
    )
    master_process = (hvd.rank() == 0)
    hvd_rank = hvd.rank()

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    start_index = 0
    end_index = len(train_examples)
    tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

    if FLAGS.horovod:
        tmp_filenames = [
            os.path.join(FLAGS.output_dir, "train.tf_record{}".format(i))
            for i in range(hvd.size())
        ]
        num_examples_per_rank = len(train_examples) // hvd.size()
        remainder = len(train_examples) % hvd.size()
        if hvd.rank() < remainder:
            start_index = hvd.rank() * (num_examples_per_rank + 1)
            end_index = start_index + num_examples_per_rank + 1
        else:
            start_index = hvd.rank() * num_examples_per_rank + remainder
            end_index = start_index + (num_examples_per_rank)

    model_fn = model_fn_builder(task_name=task_name,
                                bert_config=bert_config,
                                num_labels=len(label_list),
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate
                                if not FLAGS.horovod else FLAGS.learning_rate *
                                hvd.size(),
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False,
                                hvd=None if not FLAGS.horovod else hvd)

    # file_based_convert_examples_to_features(
    #     train_examples[start_index:end_index], label_list,
    #     FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank])

    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)

    features = file_based_input(input_file=tmp_filenames,
                                batch_size=FLAGS.train_batch_size,
                                seq_length=FLAGS.max_seq_length,
                                drop_remainder=True,
                                hvd=None if not FLAGS.horovod else hvd)

    train_op, loss = model_fn(
        features=features,
        labels=None,
    )

    # set training hooks
    training_hooks = []
    training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    # training_hooks.append(
    #     LogTrainRunHook(global_batch_size,
    #                     hvd_rank,
    #                     FLAGS.save_checkpoints_steps,
    #                     num_steps_ignore_xla=10))
    training_hooks.append(OomReportingHook())
    training_hooks.append(
        tf.train.LoggingTensorHook(
            tensors={
                # 'step': global_step,
                'loss': loss
            },
            every_n_iter=10), )

    # train_start_time = time.time()
    # does not consume all GPU RAM
    # config.gpu_options.allow_growth = True
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.48
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        tf.enable_resource_variables()

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.output_dir if master_process else None,
            hooks=training_hooks,
            config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            mon_sess.run(train_op)


if __name__ == "__main__":
    tf.compat.v1.app.run()
