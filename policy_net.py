"""
Handles construction of policy nets.
Should more or less wrap the tensorflow business required.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def _filter_summary(filters):
    """save some pics of the convolutional filters"""
    # some fancy reshaping/shuffling to turn
    # [height, width, in_channel, out_channel] filter tensor into
    # a list of [out_channel, height, width, 1] to get greyscale
    # summaries of each filter channel individually
    with tf.name_scope('summaries'):
        all_filters = tf.unpack(tf.transpose(filters, [3, 2, 0, 1]))
        all_filters = [tf.expand_dims(filter_, [3]) for filter_ in all_filters]
        name = filters.op.name
        for i, filter_ in enumerate(all_filters):
            tf.image_summary(name + '/filter_{}'.format(i), filter_)


def _activation_summary(act):
    """summarise activations for a given layer"""
    with tf.name_scope('summaries'):
        name = act.op.name
        tf.histogram_summary(name + '/activations', act)
        tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(act))


def _conv_layer(input_var, shape, stride, name, summarise=True, averager=None):
    """Makes a single convolutional layer.
    input_var should be `[batch, height, width, channels]`.
    shape should be the shape of the weight tensor required:
    `[filter_height, filter_width, in_channels, out_channels]`.
    `stride` should just be an int, will be both vertical and horizontal.
    """
    with tf.variable_scope(name) as scope:
        if not averager:
            filters = tf.get_variable('weights', shape)
            biases = tf.get_variable('biases', [shape[-1]],
                                     initializer=tf.constant_initializer(0.1))
        else:
            filters = averager.average(tf.get_variable('weights', shape))
            biases = averager.average(tf.get_variable('biases', [shape[-1]]))

        conv = tf.nn.conv2d(input_var, filters, [1, stride, stride, 1],
                            padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
        if summarise:
            _activation_summary(conv)
            _filter_summary(filters)
        return conv


def _fc_layer(input_var, size, name, nonlin=tf.nn.relu, summarise=True,
              averager=None):
    """Make a fully connected layer
    input_var should be `[batch, inputs]` and `size` should be a scalar
    number of outputs. If input size has more dimensions, we reshape,
    assuming that the first dimension is the batch.
    """
    with tf.variable_scope(name) as scope:
        input_shape = input_var.get_shape().as_list()
        if len(input_shape) != 2:
            input_var = tf.reshape(input_var, [input_shape[0], -1])
            input_dim = input_var.get_shape()[1]
        else:
            input_dim = input_shape[1]
        if not averager:  # get variables
            weights = tf.get_variable('weights', shape=[input_dim, size],
                                      trainable=True)
            biases = tf.get_variable('biases', shape=[size],
                                     initializer=tf.constant_initializer(0.1),
                                     trainable=True)
        else:  # get shadow variables from the moving average
            weights = averager.average(
                tf.get_variable('weights'))
            biases = averager.average(tf.get_variable('biases'))
        activation = nonlin(tf.matmul(input_var, weights) + biases,
                            name=scope.name)
        if summarise:
            _activation_summary(activation)
        return activation


def convolutional_inference(input_var, shape, averager=None, summarise=False):
    """Build the feedforward part of the model minus the final softmax"""

    layer_input = input_var
    nonlin = tf.nn.relu
    for i, layer in enumerate(shape):
        if i == len(shape)-1:
            nonlin = tf.identity
        try:
            if len(layer) == 4:
                # then it is a conv layer
                layer_input = _conv_layer(layer_input, layer, 1,
                                          'conv{}'.format(i+1),
                                          averager=averager,
                                          summarise=summarise)
            elif len(layer) == 1:
                layer_input = _fc_layer(layer_input, layer[0],
                                        'full{}'.format(i+1),
                                        averager=averager,
                                        nonlin=nonlin,
                                        summarise=summarise)
            else:
                raise ValueError("Can't deal with shape {}".format(layer))
        except TypeError:
            # int has no len()
            layer_input = _fc_layer(layer_input, layer,
                                    'full{}'.format(i+1),
                                    averager=averager,
                                    nonlin=nonlin,
                                    summarise=summarise)
    return layer_input  # the final one is in fact the output


def policy_gradient_loss(logits, actions, rewards):
    """Computes a loss given to encourage good moves and discourage bad
    ones. This is the top level, organising trajectories into batches
    and whatnot will be a pain but will have to be done external to here.

    Args:
        logits: the output of the network (all of them). Expected to be
            `[batch_size, num_actions]`.
        actions: the actual actions chosen, so `[batch_size]` of integers.
        rewards: `[batch_size]` tensor of whatever advantage function you want.

    Returns:
        scalar tensor: the loss, which is the negative weighted average
            reward (something we want to minimise).
    """
    with tf.name_scope('loss'):
        # first we have to get the probabilities of the actions we actually
        # took
        log_probs = tf.nn.log_softmax(logits)
        # we have to construct some indices, this is a bit awkward
        # because of https://github.com/tensorflow/tensorflow/issues/206
        prob_shape = log_probs.get_shape().as_list()
        idx_flattened = tf.range(0, prob_shape[0]) * prob_shape[1] + actions
        probs = tf.gather(tf.reshape(log_probs, [-1]), idx_flattened)
        return -tf.squeeze(tf.matmul(tf.expand_dims(probs, 1),
                                     tf.expand_dims(rewards, 1),
                                     transpose_a=True))


def get_placeholders(batch_size, image_size):
    """Gets the placeholders required for training:
        - `inputs`: the actual images, `[batch_size, *image_size]`,
        - `actions`: `[batch_size]` integers which reflect the final choices.
        - `advantages`: `[batch_size]` floats which are whatever advantage
            used. It would be nice to keep these all in tensorflow,
            but for now this is the simplest"""
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + image_size,
                            name='inputs')
    actions = tf.placeholder(tf.int32, shape=[batch_size], name='actions')
    advantages = tf.placeholder(tf.float32, shape=[batch_size],
                                name='advantages')
    return inputs, actions, advantages


def get_training_op(loss, learning_rate=0.0001,
                    collection=None,
                    global_step=None):
    """Gets an op that runs RMSProp on the given collection of trainable
    variables wrt. the given loss.

    Args:
        loss: the (scalar) loss tensor.
        learning_rate: what to use for a learning rate (a tensor or a float).
        collection: list of variables. Defaults to tf.trainable_variables().
        global_step: integer tensor if you want to keep track of training
            steps.

    Returns:
        an op that runs a step of training.
    """
    opt = tf.train.RMSPropOptimizer(learning_rate)
    return opt.minimize(loss, global_step=global_step, var_list=collection)


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    # make sure it does something
    shape = [[3, 3, 3, 16], [6, 6, 16, 4], 3]
    # make some random 9x9 3 channel images
    input_var = tf.Variable(tf.random_normal([10, 9, 9, 3]), trainable=False)

    logits = convolutional_inference(input_var, shape)

    # make some random rewards
    advantages = tf.Variable(tf.random_uniform([10], maxval=2, dtype=tf.int32),
                             trainable=False)
    advantages = 2.0 * tf.cast(advantages, tf.float32) - 1.0
    # assume policy is as greedy as can be
    actions = tf.cast(tf.argmax(logits, 1), tf.int32)

    loss = policy_gradient_loss(logits, actions, advantages)

    print([var.name for var in tf.trainable_variables()])
    print(tf.gradients(loss, tf.trainable_variables()))

    global_step = tf.Variable(0, trainable=False)
    train_op = get_training_op(loss, global_step=global_step)
    sess = tf.Session()
    writer = tf.train.SummaryWriter('/tmp/test/logs', sess.graph)
    all_summaries = tf.merge_all_summaries()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        print(sess.run(loss))
        for _ in range(100):
            loss_val, _, summary = sess.run([loss, train_op, all_summaries])
            print('\r{}'.format(-loss_val), end='')
            writer.add_summary(summary, global_step=global_step.eval())
        print('\r{}'.format(-loss_val))
