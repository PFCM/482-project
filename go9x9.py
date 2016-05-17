"""use uct on 9x9 go"""
import time
import logging

import gym
import pachi_py

import uct


def coord_to_action(board, c):
    if c == pachi_py.PASS_COORD:
        return 9**2
    if c == pachi_py.RESIGN_COORD:
        return 9**2+1
    i, j = board.coord_to_ij(c)
    return i * board.size + j


def make_description(env):
    """make a uct.GameDescription describing this environment. Kind of hacky
    and a bit cheating?"""
    colour = env.player_color

    def _reward(state):
        if state.board.is_terminal:
            white_wins = state.board.official_score
            if colour == pachi_py.WHITE and white_wins:
                return 1
            if colour == pachi_py.BLACK and not white_wins:
                return 1
        return 0

    def _terminal(state):
        return state.board.is_terminal

    def _actions(state):
        return [coord_to_action(state.board, act)
                for act in state.board.get_legal_coords(colour)]

    def _transition(state, action):
        return state.act(action)

    return uct.GameDescription(_transition,
                               _reward,
                               _terminal,
                               _actions)


def main():
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('Go9x9-v0')
    obs = env.reset()

    descr = make_description(env)
    mcts = uct.UCTSearch(descr)

    done = False
    state = env._state
    while not done:
        env.render()

        action = mcts.search(state, 1)

        obs, reward, done, info = env.step(action)
        state = info['state']

    print('got reward {}'.format(reward))


if __name__ == '__main__':
    main()
