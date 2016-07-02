"""do the UCT on hex maybe"""
import time
import sys
import logging
import argparse
import hashlib

import numpy as np
import gym

from gym.envs import board_game
from gym.envs.registration import register

import uct

# game state will be a tuple:
# (board, player)

ENV = None


def _reward(state):
    if not isinstance(state[0], str):
        return board_game.HexEnv.game_finished(state[0]) * ((2 * state[1]) - 1)
    # not a hundred per cent this will be the right way around
    return -1  # ?? Need to be a bit careful


def _terminal(state):
    return isinstance(state[0], str) or \
        board_game.HexEnv.game_finished(state[0]) != 0


def _actions(state):
    if isinstance(state[0], str):
        return []
    return board_game.HexEnv.get_possible_actions(state[0]) + \
        [81]  # make sure resign is a possible action


def _transition(state, action):
    # TODO: investigate not copying the state all the time
    if action == 81:  # then resign
        return ('resigned', state[1])
    board = state[0].copy()
    board_game.HexEnv.make_move(board, action, state[1])
    return (board, 1 - state[1])


def state_equality(a, b):
    """compare two state tuples"""
    return np.all(a[0] == b[0]) and a[1] == b[1]


def state_hash(state):
    """hash a state tuple"""
    try:
        m = hashlib.sha1(state[0].view(np.uint8))
    except AttributeError:
        # probably a str
        m = hashlib.sha1(state[0].encode('utf-8'))
    m.update(bytes(state[1]))
    return m.digest()


def make_description():
    """make a uct.GameDescription describing 9x9 hex"""
    return uct.GameDescription(_transition, _reward, _terminal, _actions,
                               state_equality, state_hash)


def human_text_policy(state):
    """Asks at the terminal for a move. Asks again if the move is illegal"""
    global ENV  # :(
    ENV.render()

    def _parse_move(text):
        try:
            coords = [int(pos.strip())-1 for pos in text.split(',')]
        except:
            return -1
        return board_game.HexEnv.coordinate_to_action(state, coords)

    a = _parse_move(input('Enter a move: '))
    while not board_game.HexEnv.valid_move(state, a):
        print('Invalid move.')
        a = _parse_move(input('Enter a move: '))
    return a


def get_args():
    """Get some arguments for things like time to think or what the opponent
    should be."""
    parser = argparse.ArgumentParser(description='Play Hex.')
    parser.add_argument('--opponent', '-o', choices=['human', 'random'],
                        default='random',
                        help='Who the tree search plays')
    parser.add_argument('--time', '-t', type=float, default=0.1,
                        help='How long to give the tree search to think.')

    return parser.parse_args()


def get_human_opponent_env():
    """Registers and makes a 9x9 hex environment with the above function that
    prompts for input as opponent"""
    # this part is pretty much the same as gym.envs.__init__.py
    register(
        id='PrettyHex9x9-human-v0',
        entry_point='prettyhex:PrettyHexEnv',
        kwargs={
            'player_color': 'black',
            'observation_type': 'numpy3c',
            'opponent': human_text_policy,  # apart from here
            'illegal_move_mode': 'raise',  # and here for debugging
            'board_size': 9
        }
    )
    return gym.make('PrettyHex9x9-human-v0')


def get_random_opponent_env():
    """Registers and makes a 9x9 hex env, exactly like the standard but a bit
    easier to see what's going on."""
    register(
        id='PrettyHex9x9-random-v0',
        entry_point='prettyhex:PrettyHexEnv',
        kwargs={
            'player_color': 'black',
            'observation_type': 'numpy3c',
            'opponent': 'random',  # apart from here
            'illegal_move_mode': 'raise',  # and here for debugging
            'board_size': 9
        }
    )
    return gym.make('PrettyHex9x9-random-v0')


def main():
    logging.getLogger().setLevel(logging.INFO)

    args = get_args()

    if args.opponent == 'random':
        env = get_random_opponent_env()
    elif args.opponent == 'human':
        # the harder part
        env = get_human_opponent_env()
        global ENV
        ENV = env  # this is gross
    observation = env.reset()
    mcts = uct.UCTSearch(make_description(),
                         tree_policy=uct.maxdepth_tree_policy(
                             10000,
                             uct.choose_ucb(1)))

    done = False
    while not done:
        env.render()

        action = mcts.search((env.state, env.to_play),
                             args.time)
        observation, reward, done, info = env.step(action)
        print('reward: {}'.format(reward))

        if done:
            break


if __name__ == '__main__':
    main()
