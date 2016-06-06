"""do the UCT on hex maybe"""
import time
import sys
import logging

import numpy as np
import gym

from gym.envs import board_game

import uct

# game state will be a tuple:
# (board, player)


def _reward(state):
    if not isinstance(state[0], str):
        return board_game.HexEnv.game_finished(state[0]) * ((2 * state[1]) - 1)
    # not a hundred per cent this will be the right way around
    return 2 * state[1] - 1


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


def make_description():
    """make a uct.GameDescription describing 9x9 hex"""
    return uct.GameDescription(_transition, _reward, _terminal, _actions,
                               state_equality)


def human_text_policy(state):
    """Asks at the terminal for a move. Asks again if the move is illegal"""
    def _parse_move(text):
        coords = [int(pos)-1 for pos in text.split(',')]
        return board_game.HexEnv.coordinate_to_action(state, coords)

    a = _parse_move(input('Enter a move: '))
    while not board_game.HexEnv.valid_move(state, a):
        print('Invalid move.')
        a = _parse_move(input('Enter a move: '))
    return a


def main():
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('Hex9x9-v0')
    observation = env.reset()
    mcts = uct.UCTSearch(make_description())

    done = False
    while not done:
        env.render()

        action = mcts.search((env.state, env.to_play),
                             float(sys.argv[1]))
        observation, reward, done, info = env.step(action)
        print('reward: {}'.format(reward))
        if done:
            break


if __name__ == '__main__':
    main()
