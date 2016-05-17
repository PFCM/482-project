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


def negamax_backup(node, reward, **kwargs):
    """negamax backup -- for two player zero sum games.
    Args:
      node: a UCTNode to back up from.
      reward: the reward for the state at Node
      **kwargs: not used.
    """
    while node is not None:
        node.count += 1
        node.Q += reward
        reward *= -1
        node = node.parent


def maxdepth_tree_policy(max_depth, choice_func):
    """Returns a tree policy which expands from the given node
    using the supplied function until a maximum depth is reached.
    
    Args:
      depth: the max depth we will consider.
      choice_func: how to choose children if the node is fully
        expanded.
    """
    def _treepolicy(node, expand, descr):
        depth = 0
        while depth < max_depth:
            actions = descr.legal_actions(node.state)
            if len(node.children) < len(actions):
                return expand(node, descr)
            node = choice_func(node)
        return node

    return _treepolicy
        

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


def choose_ucb1(constant):
    """gets a function which chooses a child node according to 
    ucb 1 with given constant"""

    def _ucb1(node):
        best_val = 0
        best_child = None
        for child in node.children:
            val = (node.Q / child.count) + constant * np.sqrt(2 * np.log(node.count)/child.count)
            if val > best_val:  # or maybe child.count is 0?
                best_val = val
                best_child = child
        return best_child
    return _ucb1
    
def uniform_expand(node, game_descr):
    """adds a child to node by sampling uniformly at random from available moves"""
    moves = game_descr.legal_actions(node.state)
    # get the set of actions we haven't tried
    done = [child.action for child in node.children]
    moves = [move for move in moves if move not in done]
    action = np.random.choice(moves)
    # got action
    child = UCTNode(game_descr.transition_function(node.state, action),
                    node, action)
    
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


class UCTSearch(object):
    """UCT search :)"""

    def __init__(self, descr,
                 tree_policy=maxdepth_tree_policy(10, choose_ucb1(.1)),
                 expand=uniform_expand, default_policy=uniform_rollout,
                 best_child=choose_ucb1(.1), backup=negamax_backup):
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
        logging.info('starting rollouts (%f seconds)', length)
        rollouts = 0
        while time.time() < (start + length):
            # this is where we could be parallel
            rollout_start = self.tree_policy(root,
                                             self.expand,
                                             self.descr)
            reward = self.default_policy(rollout_start,
                                         self.descr)
            self.backup(rollout_start, reward)
            rollouts += 1
            print('\r{} rollouts'.format(rollouts), end='')
        print()

        action = self.best_child(root).action
        logging.info('Got action in %f seconds.', time.time() - start)
        return action

