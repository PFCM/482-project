"Tensorflow running average (not exponential)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RunningAverage(object):
    """Keeps a running average of the variables given to it. The goal is for this guy
    to behave like tf's built in exponential moving average, just without the
    exponential moving part.

    In particular we do:
       M_1 = x_1
       M_k = M_{k-1} + \frac{x_k - M_{k-1}}{k}

    To update the average at each step. The averages are kept in shadow variables
    etc etc, hopefully use should be as per tf's ema.
    """

    def __init__(self):
        self.shadow_vars = {}  # will store a shadow variable and an update count

    def apply(self, var_list):
        """Applies the running average to a list of variables
        Creates shadow variables and update op. Returns a grouped update op for
        all the averages in the list."""
        update_ops = []
        with tf.variable_scope('running_average'):
            for var in var_list:
                # add a shadow var that gets initialized to the same value
                # and a count to keep track of how many times it's been updated
                name = var.op.name
                count = tf.get_variable(
                    name+'_count', dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0),
                    shape=[])
                shadow = tf.get_variable(
                    name+'_shadow', dtype=var.dtype,
                    initializer=var.initialized_value())
                # now make the update ops
                # increase the count
                count_update = tf.assign_add(count, 1.0)
                with tf.control_dependencies([count_update]):
                    difference = (var - shadow)/count
                    update = tf.assign_add(shadow, difference)
                update_ops.append(update)
                self.shadow_vars[var] = (shadow, count)

        return update_ops

    def average(self, var):
        """Get the average for a variable"""
        return self.shadow_vars[var][0]

    def average_name(self, var):
        return self.shadow_vars[var][0].op.name


if __name__ == '__main__':
    # quick test
    # average of a normal best be close to the mean
    rando = tf.random_normal(shape=[1], mean=1.0)
    var = tf.Variable(rando)
    averager = RunningAverage()
    update_avge = averager.apply([var])
    with tf.control_dependencies(update_avge):
        update = tf.assign(var, rando)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    with sess.as_default():
        for i in range(50000):
            sess.run(update)
            print('\r{}'.format(i), end='')
        print('\rAverage: {}'.format(averager.average(var).eval()))
                    
