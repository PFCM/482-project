"""Classic UCT, hopefully fairly modular"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import time
import logging

from deco import concurrent, synchronized

import numpy as np


class UCTNode(object):
    """The nodes for our tree"""

    def __init__(self, state, parent, action):
        """Make a new node. If parent is not None, appends self to
        parent's list of children.

        Args:
          state: whatever we are using for the state.
          parent: another node or None for the root.
          action: the action taken to get here (or None for the root)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.Q = 0
        self.count = 0
        self.children = []
        if parent:
            parent.children.append(self)


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
        node = node.parent


# TODO these guys will have to be classes if we want to use deco
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
        while depth < self.max_depth:
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
        try:  # This shouldn't be happening, but for now it is
            state = game_descr.transition_function(state, action)
        except:
            break
    return game_descr.reward_function(state)


class choose_ucb1(object):
    """gets a function which chooses a child node according to
    ucb 1 with given constant"""

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, node):
        # TODO: make this tidier and more robust
        best_val = 0
        best_child = node.children[0]
        for child in node.children:
            val = (node.Q / child.count) + self.constant * np.sqrt(2 * np.log(node.count)/child.count)
            if val > best_val:  # or maybe child.count is 0?
                best_val = val
                best_child = child
        return best_child


def uniform_expand(node, game_descr):
    """adds a child to node by sampling uniformly at random from available moves"""
    legal_moves = game_descr.legal_actions(node.state)
    # get the set of actions we haven't tried
    # somehow we are getting illegal moves here
    done = [child.action for child in node.children]
    moves = [move for move in legal_moves if move not in done]
    # print('{}/{} moves'.format(len(moves), len(legal_moves)))
    action = np.random.choice(moves)
    # got action
    try:
        child = UCTNode(game_descr.transition_function(node.state, action),
                        node, action)
    except:
        #print('illegal move in expand')
        child = node

    return child


class GameDescription(object):
    """Contains the transition functions etc for the game"""

    def __init__(self, transition_function, reward_function,
                 terminal_predicate, legal_actions):
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
        """
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.terminal_predicate = terminal_predicate
        self.legal_actions = legal_actions

@concurrent
def _single_search(obj, rollout_start, num=16):
    #logging.info('%d rolling out', index)
    start = time.time()
    reward, runs = 0, 0
    for i in range(num):
        reward += obj.default_policy(rollout_start,
                                     obj.descr)
        runs += 1
    return (reward, runs, time.time()-start)


@synchronized
def _batch(obj, root, batch_size=16):
    """Does batch_size concurrent searches.
    Backs them up serially.
    """
    backups = {}
    for index in range(batch_size):
        rollout_start = obj.tree_policy(
            root, obj.expand, obj.descr)
        #logging.info('Started %d', index)
        backups[index] = _single_search(obj, rollout_start)
    total_time = 0
    for index, result in backups.items():
        #logging.info('got results for %d', index)
        reward, runs, time = result
        obj.backup(node, reward, runs)
        total_time += time
    return total_time / batch_size / 16


class UCTSearch(object):
    """UCT search :)"""

    def __init__(self, descr,
                 tree_policy=maxdepth_tree_policy(25, choose_ucb1(.7)),
                 expand=uniform_expand, default_policy=uniform_rollout,
                 best_child=choose_ucb1(0), backup=negamax_backup):
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
          default_policy: a function which takes a node and the game description
            and performs a rollout returning the reward.
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
        root = UCTNode(state, None, None)
        logging.info('starting rollouts (%d seconds)', length)
        rollouts = 0
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
            #print('\r{} rollouts {}s each'.format(rollouts, av_time), end='')
            print('\r{} rollouts'.format(rollouts), end='')
        print()

        action = self.best_child(root).action
        logging.info('Got action %s in %f seconds.', action, time.time() - start)
        return action
