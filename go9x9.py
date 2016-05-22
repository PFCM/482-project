"""use uct on 9x9 go"""
import time
import logging
import sys

import gym
from gym.envs.registration import register
import pachi_py

import uct


def coord_to_action(board, c):
    if c == pachi_py.PASS_COORD:
        return 9**2
    if c == pachi_py.RESIGN_COORD:
        return 9**2+1
    i, j = board.coord_to_ij(c)
    return i * board.size + j


# these functions need to be top level as they may be pickled
class _reward(object):

    def __init__(self, colour):
        self.colour = colour

    def __call__(self, state):
        if state.board.is_terminal:
            white_wins = state.board.official_score > 0
            if self.colour == pachi_py.WHITE and white_wins:
                return 1
            if self.colour == pachi_py.BLACK and not white_wins:
                return 1
            return -1
        raise ValueError('reward of non-terminal state is probably'
                         ' not what you intended.')


def other_reward(state):
    if state.board.is_terminal:
        white_wins = state.board.fast_score > 0
        if state.color == pachi_py.WHITE and white_wins:
            return 1
        if state.color == pachi_py.BLACK and not white_wins:
            return 1
        return -1
    raise ValueError('reward of non-terminal state is probably'
                     ' not what you intended.')


def _terminal(state):
    return state.board.is_terminal


def _actions(state):
    return [coord_to_action(state.board, act)
            for act in state.board.get_legal_coords(state.color)]


def _transition(state, action):
    return state.act(action)


def make_description(env):
    """make a uct.GameDescription describing this environment. Kind of hacky
    and a bit cheating?"""
    colour = env.player_color

    return uct.GameDescription(_transition,
                               other_reward,  # (colour),
                               _terminal,
                               _actions)


def main():
    logging.getLogger().setLevel(logging.INFO)
    # register(
    #     id='Go9x9me-v0',
    #     entry_point='gym.envs.board_game:GoEnv',
    #     kwargs={
    #         'player_color': 'black',
    #         'opponent': 'random',
    #         'observation_type': 'image3c',
    #         'illegal_move_mode': 'raise',
    #         'board_size': 9,
    #     },
    # )
    # env = gym.make('Go9x9me-v0')
    env = gym.make('Go9x9-v0')

    wins = 0

    for game in range(100):
        obs = env.reset()

        descr = make_description(env)
        mcts = uct.UCTSearch(descr)

        done = False
        state = env._state
        moves = 0
        while not done:
            env.render()

            action = mcts.search(state, float(sys.argv[1]))
            moves += 1
            obs, reward, done, info = env.step(action)
            state = info['state']

            if moves % 50 == 0:
                print('                              '
                      '\r{}'.format(moves), end='')
        print('                                   '
              '\r{}'.format(moves), end='')

        print('                                      '
              '\r({}) got reward {}'.format(game+1, reward), end='')
        if reward == 1:
            wins += 1
    print('///Win {}/{}'.format(wins, 100))


if __name__ == '__main__':
    main()
