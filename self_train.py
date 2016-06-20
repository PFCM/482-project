"""Train the policy net by having it play hex against a running average of
itself"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import time
import datetime
import logging

import numpy as np
import tensorflow as tf

import gym

import policy_net
from averager import RunningAverage

flags = tf.app.flags
flags.DEFINE_string('logdir', 'logs', 'where the summaries for tensorboard go')
flags.DEFINE_integer('batch_size', 100, 'size of training batches. Affects how'
                     ' often we train.')
flags.DEFINE_string('savepath', 'models/convolutional/conv',
                    'where to save checkpoints')
flags.DEFINE_integer('max_updates', 100, 'how many times to update parameters '
                     'at most')
flags.DEFINE_string('opponent', 'random', 'one of [random, self] whether to '
                    'play a running average of oneself or a random opponent')
flags.DEFINE_string('loadpath', '', 'where to look for a model. If empty, '
                    'then initialize fresh')
FLAGS = flags.FLAGS


def flip_state(state):
    """take a 3d game state array and shuffle it around so that the
    goal and stone colours are consistent regardless of which player
    the convnet is.
    """
    # first transpose the board
    state = np.transpose(state, (0, 2, 1))
    p1 = state[0, ...]
    state[0, ...] = state[1, ...]
    state[1, ...] = p1
    return state


def sample_action(probabilities, epsilon, available_set=None):
    """samples an action from given probabilities"""
    # sometimes numerical issues make problems, so do it by hand
    if available_set:
        mask = np.ones_like(probabilities, np.bool)
        mask[available_set] = 0
        probabilities[mask] = 0
        probabilities /= probabilities.sum()
    if np.random.sample() > epsilon:
        return np.searchsorted(np.cumsum(probabilities), np.random.sample())
    
    if available_set:
        return np.random.choice(available_set)
    return 82


class TFSamplePolicy(object):
    """Provides a callable that gets a policy using some logits from tf.
    If a session is not specified in the constructor, will just use the
    default.
    """

    def __init__(self, logit_var, input_pl, session=None, invert_state=False,
                 keep_actions=False, epsilon=0.0):
        """Gets ready to go.

        Args:
            logit_var: the tensor that will evaluate to the logits we will use.
            input_pl: placeholder for an input image.
            session: optional session to use.
            invert_state: if we're playing as white, might want to flip the
                pieces around (and transpose the board)
            keep_actions: if this is the real player we are training, it is
                probably a good plan to keep track of what it has seen and
                done. This way we can train it up whoever it has become.
        """
        self.probs = tf.nn.softmax(logit_var)
        self.input = input_pl
        self.session = session or tf.get_default_session()
        self.invert_state = invert_state
        if keep_actions:
            self.trajectory = []
        else:
            self.trajectory = None
        self.epsilon = epsilon

    def __call__(self, state):
        """produces a move from a state"""
        state = state.copy()
        if self.invert_state:
            state = flip_state(state)
        likelihoods, = self.session.run(
            self.probs, {self.input: state.reshape((1, 9, 9, 3))})

        moves = gym.envs.board_game.HexEnv.get_possible_actions(state)
        move = sample_action(likelihoods, self.epsilon, moves)
        # protect from losing all the time due to illegal moves
        # if not gym.envs.board_game.HexEnv.valid_move(state, move):
        #    if np.random.sample() > 0.25:
        #        logging.debug('illegal move, choosing at random')
        #        moves = gym.envs.board_game.HexEnv.get_possible_actions(state)
        #        if len(moves):
        #            move = np.random.choice(moves)
        #        else:
        #            loging.debug('nothing left')
        #            move = 0  # must be illegal, so will lose
        #    else:
        #        logging.debug('illegal move, losing')
        if self.trajectory is not None:
            self.trajectory.append((state, move))
        return move


def get_hexenv(opponent, num):
    """Gets a gym env to play hex with the opponent installed.

    Args:
        opponent: the opponent policy.

    Returns:
        an environment.
    """
    # some gross hacking
    try:
        get_hexenv.env.opponent = opponent
    except AttributeError:
        name = 'PrettyHex9x9-v{}'.format(num)
        gym.envs.register(
            id=name,
            entry_point='prettyhex:PrettyHexEnv',
            kwargs={
                'player_color': 'black',
                'observation_type': 'numpy3c',
                'opponent': opponent,
                'illegal_move_mode': 'tie',  # try learn not to
                'board_size': 9
                }
            )
        get_hexenv.env = gym.make(name)
    return get_hexenv.env


def sample_trajectory(black, white, step, render=False):
    """plays out a game.

    Args:
        black: a policy that will produce moves agains the env.
        white: a policy that will be embedded in the env as the opponent.
        render: if we should spit it out as we go. Probably a bad plan if
            just trying to train, but periodically checking in isn't a terrible
            idea.

    Returns:
        (black, white, reward): the policy objects (which may have saved state)
            and the reward: 1.0 if black wins or -1.0 if white wins.
    """
    env = get_hexenv(white, step)
    observation = env.reset()
    done = False
    while not done:
        if render:
            env.render()

        player_action = black(observation)
        # logging.info('got move %d', player_action)
        observation, reward, done, info = env.step(player_action)

    return black, white, reward


def get_advantages(reward, length, discount=1.0):
    """appropriately discounts the rewards"""
    # first let's figure out the discount factors
    advs = np.ones(length, dtype=np.float32) * discount
    advs = np.power(advs, np.arange(length, 0, -1, dtype=np.float32))
    return advs * -reward


def weight_decay_regularizer(amount):
    def _reg(t):
        return amount * tf.nn.l2_loss(t)
    return _reg


def random_policy(state):
    possibles = gym.envs.board_game.HexEnv.get_possible_actions(state)
    if len(possibles) > 0:
        a = np.random.randint(len(possibles))
        return possibles[a]
    return 81


def main(_):
    logging.getLogger().setLevel(logging.DEBUG)
    # first we have to get the models
    logging.info('getting model')
    shape = [[3, 3, 3, 64]] + [[3, 3, 64, 64]]*3 + [256, 81]
    inputs_pl_t, action_pl, advantage_pl = policy_net.get_placeholders(
        FLAGS.batch_size, [9, 9, 3])
    # get another input placeholder so we can do feedforward one at a time
    inputs_pl_p = tf.placeholder(tf.float32, shape=[1, 9, 9, 3],
                                 name='one_input')

    # add a summary of the inputs so we can see what's going on
    tf.image_summary('inputs', inputs_pl_t)

    with tf.variable_scope('policy_model',
                           regularizer=weight_decay_regularizer(0.001)) as scope:
        player_logits_train = policy_net.convolutional_inference(
            inputs_pl_t, shape, summarise=True, dropout=0.5)
        logging.debug('got player train')
        # get a running average of the params
        avger = RunningAverage()
        update_averages = avger.apply(tf.trainable_variables())
        logging.debug('got averager')
        scope.reuse_variables()
        player_logits_play = policy_net.convolutional_inference(
            inputs_pl_p, shape, dropout=0.5)
        logging.debug('got player')
        opponent_logits_play = policy_net.convolutional_inference(
            inputs_pl_p, shape, averager=avger, dropout=0.5)
        logging.debug('got average model')
        # summarise the outputs so we can check in on what it thinks is good
        tf.image_summary(
            'output',
            tf.reshape(tf.nn.softmax(player_logits_train),
                       [FLAGS.batch_size, 9, 9, 1]))

    # and now get the loss and training ops
    loss_op = policy_net.policy_gradient_loss(player_logits_train, action_pl,
                                              advantage_pl)
    tf.scalar_summary('loss', loss_op)
    loss_op += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = policy_net.get_training_op(loss_op, global_step=global_step,
                                          learning_rate=0.0001)

    # make sure the average gets updated
    with tf.control_dependencies([train_op]):
        train_op = tf.group(*update_averages)
    logging.info('got model')
    logging.debug('geting misc ops')
    # get set up to save models
    to_save = tf.trainable_variables()
    to_save += tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    to_save += [global_step]
    saver = tf.train.Saver(to_save)
    all_summaries = tf.merge_all_summaries()
    logging.debug('got misc ops')
    # get ready
    sess = tf.Session()
    logging.info('initialising')
    if FLAGS.loadpath:
        # do the restoring
        # the optimizer likes to add some variables, we haven't been saving
        # them because we are only interested in saving the model rather
        # picking up training where we left off.
        # the easiest thing to do is initialize the lot and then restore
        # which is double work but also actually possible.
        sess.run(tf.initialize_all_variables())
        path = tf.train.latest_checkpoint(FLAGS.loadpath)
        saver.restore(sess, path)
        logging.info('restored from %s', path)
    else:
        sess.run(tf.initialize_all_variables())
        logging.info('initialised fresh')
    game_num = 0
    with sess.as_default():
        summary_writer = tf.train.SummaryWriter(
            FLAGS.logdir+'/{}'.format(datetime.datetime.now().strftime(
                '%d-%m-%y--%H%M')),
            sess.graph)
        # ready to do some training
        data = collections.deque()
        # now let's sample a few trajectories
        rl_is_black = True
        total_reward = 0.0  # for logging
        max_step = int(global_step.eval()) + FLAGS.max_updates
        logging.info('starting with a model that has had %d updates', 
                     global_step.eval())
        while global_step.eval() < max_step:
            while len(data) < FLAGS.batch_size:
                logging.info('not enough data to train, playing')
                # then we are going to have to get some
                rl_player = TFSamplePolicy(
                    player_logits_play, inputs_pl_p, keep_actions=True,
                    session=sess, invert_state=not rl_is_black)
                if FLAGS.opponent == 'self':
                    av_player = TFSamplePolicy(
                        opponent_logits_play, inputs_pl_p, keep_actions=False,
                        session=sess, invert_state=rl_is_black, epsilon=0.05)
                else:
                    av_player = random_policy
                player = rl_player if rl_is_black else av_player
                opponent = av_player if rl_is_black else rl_player
                # play it out
                player, opponent, reward = sample_trajectory(
                    player, opponent, game_num, render=False)
                # calculate the advantages
                reward = (reward * -1) if not rl_is_black else reward
                total_reward += reward
                logging.debug('  reward: %f', reward)
                advantages = get_advantages(reward, len(rl_player.trajectory),
                                            0.999999)
                # flatten out the tuples a little bit
                trajectory = [(s, a, r)
                              for r, (s, a)
                              in zip(advantages, rl_player.trajectory)]
                data.extend(trajectory)
                logging.info('game %d complete.', game_num)
                game_num += 1
                logging.info('~~ average reward %f', total_reward / game_num)
            logging.debug('out of play loop, queue has %d', len(data))
            # ok we have at least a batch
            batch_tuples = [data.pop() for _ in range(FLAGS.batch_size)]
            np.random.shuffle(batch_tuples)
            # split up the guys we want
            states = np.array([np.transpose(item[0], (1, 2, 0))
                               for item in batch_tuples],
                              dtype=np.float32)
            actions = np.array(
                [item[1] for item in batch_tuples],
                dtype=np.int32)
            advantages = np.array(
                [item[2] for item in batch_tuples],
                dtype=np.float32)
            # now we can run a training step on it
            loss, _, summaries = sess.run(
                [loss_op, train_op, all_summaries],
                {inputs_pl_t: states,
                 action_pl: actions,
                 advantage_pl: advantages})
            print('\r({}): {}'.format(global_step.eval(), loss))
            # save the model and the summaries
            summary_writer.add_summary(summaries, global_step.eval())
            saver.save(sess, FLAGS.savepath, global_step=global_step.eval())
            logging.debug('saved model and summaries')


if __name__ == '__main__':
    tf.app.run()
