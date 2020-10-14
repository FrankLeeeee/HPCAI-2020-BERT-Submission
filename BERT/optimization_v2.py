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
"""Functions and classes related to optimization (weight updates).
===================================

This is the version which is similar
to the Efficient DenseNet Implementation

===================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from horovod.tensorflow.compression import Compression


def create_optimizer(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     hvd=None,
                     manual_fp16=False,
                     use_fp16=False,
                     num_accumulation_steps=1,
                     optimizer_type="adam",
                     allreduce_post_accumulation=False,
                     init_loss_scale=2**32):
    """Creates an optimizer training op."""
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # avoid step change in learning rate at end of warmup phase
    if optimizer_type == "adam":
        power = 1.0
        decayed_learning_rate_at_crossover_point = init_lr * (
            (1.0 - float(num_warmup_steps) / float(num_train_steps))**power)
    else:
        power = 0.5
        decayed_learning_rate_at_crossover_point = init_lr
    adjusted_init_lr = init_lr * (init_lr /
                                  decayed_learning_rate_at_crossover_point)
    print(
        'decayed_learning_rate_at_crossover_point = %e, adjusted_init_lr = %e'
        % (decayed_learning_rate_at_crossover_point, adjusted_init_lr))

    learning_rate = tf.constant(value=adjusted_init_lr,
                                shape=[],
                                dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate,
                                                        global_step,
                                                        num_train_steps,
                                                        end_learning_rate=0.0,
                                                        power=power,
                                                        cycle=False)

    # native adam optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=adjusted_init_lr,
                                           momentum=0.9)

    # Choose a loss scale manager which decides how to pick the right loss scale
    # throughout the training process.
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
        128, 100)
    # Wraps the original optimizer in a LossScaleOptimizer.
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(
        optimizer, loss_scale_manager)

    compression = hvd.Compression.fp16

    optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    global_step = tf.identity(global_step, name='step_update')

    return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.identity(learning_rate, name='learning_rate')
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self,
                        grads_and_vars,
                        global_step=None,
                        name=None,
                        manual_fp16=False):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
            if has_shadow:
                # create shadow fp32 weights for fp16 variable
                param_fp32 = tf.get_variable(name=param_name + "/shadow",
                                             dtype=tf.float32,
                                             trainable=False,
                                             initializer=tf.cast(
                                                 param.initialized_value(),
                                                 tf.float32))
            else:
                param_fp32 = param

            m = tf.get_variable(name=param_name + "/adam_m",
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.zeros_initializer())
            v = tf.get_variable(name=param_name + "/adam_v",
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) +
                      tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) +
                      tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param_fp32

            update_with_lr = self.learning_rate * update

            next_param = param_fp32 - update_with_lr

            if has_shadow:
                # cast shadow fp32 weights to fp16 and assign to trainable variable
                param.assign(tf.cast(next_param, param.dtype.base_dtype))
            assignments.extend([
                param_fp32.assign(next_param),
                m.assign(next_m),
                v.assign(next_v)
            ])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class LAMBOptimizer(tf.compat.v1.train.Optimizer):
    """A LAMB optimizer that includes "correct" L2 weight decay."""
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="LAMBOptimizer"):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = tf.identity(learning_rate, name='learning_rate')
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self,
                        grads_and_vars,
                        global_step,
                        name=None,
                        manual_fp16=False):
        """See base class."""
        assignments = []
        steps = tf.cast(global_step, tf.float32)
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
            if has_shadow:
                # create shadow fp32 weights for fp16 variable
                param_fp32 = tf.get_variable(name=param_name + "/shadow",
                                             dtype=tf.float32,
                                             trainable=False,
                                             initializer=tf.cast(
                                                 param.initialized_value(),
                                                 tf.float32))
            else:
                param_fp32 = param

            m = tf.get_variable(name=param_name + "/adam_m",
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.zeros_initializer())
            v = tf.get_variable(name=param_name + "/adam_v",
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.zeros_initializer())

            # LAMB update
            next_m = (tf.multiply(self.beta_1, m) +
                      tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) +
                      tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            beta1_correction = (1 - self.beta_1**steps)
            beta2_correction = (1 - self.beta_2**steps)

            next_m_unbiased = next_m / beta1_correction
            next_v_unbiased = next_v / beta2_correction

            update = next_m_unbiased / (tf.sqrt(next_v_unbiased) +
                                        self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param_fp32

            w_norm = linalg_ops.norm(param, ord=2)
            g_norm = linalg_ops.norm(update, ord=2)
            ratio = array_ops.where(
                math_ops.greater(w_norm, 0),
                array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm),
                                1.0), 1.0)

            update_with_lr = ratio * self.learning_rate * update

            next_param = param_fp32 - update_with_lr

            if has_shadow:
                # cast shadow fp32 weights to fp16 and assign to trainable variable
                param.assign(tf.cast(next_param, param.dtype.base_dtype))
            assignments.extend([
                param_fp32.assign(next_param),
                m.assign(next_m),
                v.assign(next_v)
            ])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
