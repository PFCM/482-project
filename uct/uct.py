"""Classic UCT, hopefully fairly modular"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import time
import logging

import numpy as np


class UCTNode(object):
    """The nodes for our tree"""
    state_table = {}  # the transposition table

    def __init__(self, state, parent, action, descr):
        """Make a new node. If parent is not None, appends self to
        parent's list of children.

        Args:
          state: whatever we are using for the state.
          parent: another node or None for the root.
          action: the action taken to get here (or None for the root)
          descr: the GameDescription, this is used to potentially hash the
            state.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        if descr.state_hash:
            state_hash = descr.state_hash(state)
            self._state_hash = state_hash
            if state_hash not in self.state_table:
                self.state_table[state_hash] = [0, 0]
            # otherwise use what's there)
        else:
            self._state_hash = None
            self._Q = 0
            self._count = 0
        if parent:
            parent.children.append(self)

    @property
    def Q(self):
        if self._state_hash:
            return self.state_table[self._state_hash][0]
        return self._Q

    @Q.setter
    def Q(self, value):
        if self._state_hash:
            self.state_table[self._state_hash][0] = value
        else:
            self._Q = value

    @property
    def count(self):
        if self._state_hash:
            return self.state_table[self._state_hash][1]
        return self._count

    @count.setter
    def count(self, value):
        if self._state_hash:
            self.state_table[self._state_hash][1] = value
        else:
            self._count = value


def negamax_backup(node, reward, runs=1, **kwargs):
    """negamax backup -- for two player zero sum games.
    Args:
      node: a UCTNode to back up from.
      reward: the reward for the state at Node
      **kwargs: not used.
    """
    while node is not None:
        node.count += runs
        node.Q += reward
        reward *= -1
        # reward = 1 - reward
        node = node.parent


class maxdepth_tree_policy(object):
    """Returns a tree policy which expands from the given node
    using the supplied function until a maximum depth is reached.

    Args:
      depth: the max depth we will consider.
      choice_func: how to choose children if the node is fully
        expanded.
    """

    def __init__(self, max_depth, choice_func):
        self.max_depth = max_depth
        self.choice_func = choice_func

    def __call__(self, node, expand, descr):
        depth = 0
        while depth < self.max_depth and \
          not descr.terminal_predicate(node.state):
            actions = descr.legal_actions(node.state)
            if len(node.children) < len(actions):
                return expand(node, descr)
            node = self.choice_func(node)
        return node


def uniform_rollout(start_node, game_descr):
    """Does rollout with uniform probabilities. Returns reward
    from the terminal state we find."""
    state = start_node.state
    while not game_descr.terminal_predicate(state):
        action = np.random.choice(game_descr.legal_actions(state))
        state = game_descr.transition_function(state, action)
    return game_descr.reward_function(state)


class choose_ucb(object):
    """gets a function which chooses a child node according to
    ucb with given constant"""

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, node):
        # TODO: make this tidier and more robust
        # eg, what if it has no children?
        best_val = 0
        best_child = node.children[0]
        for child in node.children:
            val = (child.Q / child.count) + self.constant * \
                np.sqrt(2 * np.log(node.count)/child.count)
            if val > best_val:  # or maybe child.count is 0?
                best_val = val
                best_child = child
        return best_child


def uniform_expand(node, game_descr):
    """adds a child to node by sampling uniformly at random from available
    moves"""
    legal_moves = game_descr.legal_actions(node.state)
    # get the set of actions we haven't tried
    done = [child.action for child in node.children]
    moves = [move for move in legal_moves if move not in done]
    action = np.random.choice(moves)
    # got action
    child = UCTNode(game_descr.transition_function(node.state, action),
                    node, action, game_descr)
    return child


class GameDescription(object):
    """Contains the transition functions etc for the game"""

    def __init__(self, transition_function, reward_function,
                 terminal_predicate, legal_actions, state_eq=None,
                 state_hash=None):
        """
        Args:
          transition_function: a function which takes a state and an
            action that returns a new state.
          reward_function: a function which takes a state and returns
            a reward. Should never be evaluated for non-terminal
            states.
          terminal_predicate: a function that takes a state and returns
            True if the state is terminal.
          legal_actions: a function that takes a state and returns a
            list of available actions.
          state_eq: optional function to compare states, if classic == won't
            do it.
          state_hash: optional function to compute some kind of hash of states,
            if you want to use a transposition table (recommended).
        """
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.terminal_predicate = terminal_predicate
        self.legal_actions = legal_actions
        if not state_eq:
            self.state_eq = lambda a, b: a == b
        else:
            self.state_eq = state_eq
        self.state_hash = state_hash


class UCTSearch(object):
    """UCT search :)"""

    def __init__(self, descr,
                 tree_policy=maxdepth_tree_policy(25, choose_ucb(1)),
                 expand=uniform_expand, default_policy=uniform_rollout,
                 best_child=choose_ucb(0), backup=negamax_backup):
        """Initialise the search object.

        Args:
          descr: a GameDescription containing the game specific info
            such as transition functions etc.
          tree_policy: a function which takes a UCTNode, a method to expand
            a node by one child and a function to get the set of legal actions
            from a state and returns a
            child on which to start a rollout.
          expand: a method takes a node that does not have all of its
            children, adds a child and returns it.
          default_policy: a function which takes a node and the game
            description and performs a rollout returning the reward.
          best_child: a function which takes a node and returns the
            best child.
          backup: a function which takes a node and a reward and backs
            up the reward all the way to the root, updating the node
            statistics.
        """
        self.descr = descr
        self.tree_policy = tree_policy
        self.expand = expand
        self.default_policy = default_policy
        self.best_child = best_child
        self.backup = backup
        self.last_node = None

    def search(self, state, length):
        """Does a search from a state for around `length` seconds.
        Timing is pretty rough.

        Args:
          state: the state
          length: how long to try it out.

        Return:
          an action to take
        """
        start = time.time()
        if not self.last_node:
            root = UCTNode(state, None, None, self.descr)
        else:
            root = self._get_child(self.last_node, state, self.descr)
        logging.info('starting rollouts (%d seconds)', length)
        rollouts = 0
        begin_r = time.time()
        while time.time() < (start + length):
            # this is where we could be parallel
            rollout_start = self.tree_policy(root,
                                             self.expand,
                                             self.descr)
            reward = self.default_policy(rollout_start,
                                         self.descr)
            self.backup(rollout_start, reward)

            # av_time = _batch(self, root, batch_size=16)
            rollouts += 1
            # print('\r{} rollouts {}s each'.format(rollouts, av_time), end='')
            if rollouts % 100 == 0:
                print('\r{} rollouts ({:.3f}s)'.format(
                    rollouts, (time.time()-begin_r)/100), end='')
                begin_r = time.time()
        print('\r{} rollouts'.format(rollouts))
        if len(root.children) == 0:
            raise Exception(
                'something is really quite wrong -- root has no children')
        action_node = self.best_child(root)
        action = action_node.action
        logging.info('Transposition table has %d elements',
                      len(UCTNode.state_table))
        logging.info('Got action %s in %f seconds. (average Q: %f, %d visits)',
                     action, time.time() - start,
                     action_node.Q / action_node.count,
                     action_node.count)
        self.last_node = action_node
        return action

    def _get_child(self, node, state, descr):
        """Looks for a child of the given node with an equivalent state"""
        for child in node.children:
            # if state.board == child.state.board:
            if descr.state_eq(state, child.state):
                print('reusing! ({:.1f}/{} = {:.4f})'.format(
                    child.Q, child.count,
                    child.Q/child.count))
                print('  ({}) children'.format(len(child.children)))
                # go ahead and del the parent, hoefully gc will pickup
                # setting parent to None wasn't doing it
                return child
        # print('could not find')
        return UCTNode(state, None, None, descr)
