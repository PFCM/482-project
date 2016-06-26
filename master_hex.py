"""The flexible option, serves for validation by playing different opponents
against one another (including humans)."""

import argparse
import logging

import gym

import prettyhex


# the list of players we are prepared to accomodate
PLAYERS = [
    'human',
    'random',
    'uct',
    'policy-net-random',
    'policy-net-tuned',
    'mcts-policy-net'
]


def get_player(player, colour):
    """Gets a callable that returns a move, finding it in the appropriate
    place."""
    if player == 'human':
        from hex import human_text_policy
        return human_text_policy
    else:
        raise ValueError('I do not know this player: {}'.format(player))


def get_args():
    """set up the arguments with argparse. Could use tf, but there's no need
    to import stuff we don't need, want this to be nice and easy to run."""
    parser = argparse.ArgumentParser(description="play a games")
    parser.add_argument('--black', '-b', default='human', type=str,
                        choices=PLAYERS)
    parser.add_argument('--white', '-w', default='human', type=str,
                        choices=PLAYERS)
    parser.add_argument('--illegal_move_mode', default='raise', type=str,
                        choices=['raise', 'tie', 'lose'])
    return parser.parse_args()


def get_environment(opponent, illegal_move_mode):
    """Gets an environment with the appropriate opponent installed"""
    gym.envs.register(
        id='PrettyHexEnv-v0',
        entry_point='prettyhex:PrettyHexEnv',
        kwargs={
            'player_color': 'black',
            'observation_type': 'numpy3c',
            'opponent': opponent,
            'illegal_move_mode': illegal_move_mode,
            'board_size': 9
            }
        )
    return gym.make('PrettyHexEnv-v0')


def main():
    """load up the args and play a game"""
    logging.basicConfig(level=logging.DEBUG)

    args = get_args()
    black_player = get_player(args.black, 'BLACK')
    white_player = get_player(args.white, 'WHITE')
    logging.info('Setting up a game, %s playing %s', args.black, args.white)
    env = get_environment(black_player, args.illegal_move_mode)
    obs = env.reset()

    done = False
    while not done:
        env.render()
        white_action = white_player(obs)

        observation, reward, done, info = env.step(white_action)



if __name__ == '__main__':
    main()
