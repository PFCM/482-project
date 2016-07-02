"""The flexible option, serves for validation by playing different opponents
against one another (including humans)."""

import argparse
import logging
import shutil
import sys
import os

import numpy as np
import gym

import prettyhex
import hex as hexgame
import uct


# the list of players we are prepared to accomodate
PLAYERS = [
    'human',
    'random',
    'uct',
    'uct-nohash',
    'policy-net-random',
    'policy-net-tuned',
    'mcts-policy-net'
]


def classic_uct_policy(max_depth=100000, search_time=1.0, colour=0,
                       do_hash=True):
    """makes a callable that does uct stuff with given params."""
    mcts = uct.UCTSearch(hexgame.make_description(do_hash=do_hash),
                         tree_policy=uct.maxdepth_tree_policy(
                             max_depth, uct.choose_ucb(1)))

    def _get_uct_move(state):
        return mcts.search((state,
                            gym.envs.board_game.HexEnv.__dict__[colour]),
                           search_time)

    return _get_uct_move


def random_policy(state):
    possibles = gym.envs.board_game.HexEnv.get_possible_actions(state)
    if len(possibles) > 0:
        a = np.random.randint(len(possibles))
        return possibles[a]
    return 81


def get_full_policy_net(model_path, colour):
    """Get a policy net callable, with its own tensorflow session and graph
    completely self-contained"""
    # at this stage they will be loaded from the same file, so there is
    # definitely no point making stacks of graphs/sessions
    # just cache one per colour/path combo and return that
    if model_path + colour not in get_full_policy_net.__dict__:

        import tensorflow as tf
        import self_train
        import policy_net

        graph = tf.Graph()
        with graph.as_default():
            input_pl = tf.placeholder(tf.float32, shape=[1, 9, 9, 3],
                                      name='one_input')
            # have to make sure the names of the variables line up
            with tf.variable_scope('policy_model'):
                logits = policy_net.convolutional_inference(
                    input_pl, self_train.canonical_shape())
            saver = tf.train.Saver(tf.trainable_variables())
            if os.path.isdir(model_path):
                model_path = tf.train.latest_checkpoint(model_path)
            logging.info('Attempting to restore policy net from %s',
                         model_path)
            sess = tf.Session()
            saver.restore(sess, model_path)
        # drop out of the graph as default, but we still have the session
        get_full_policy_net.__dict__[model_path + colour] = \
            self_train.TFSamplePolicy(
            logits, input_pl, session=sess, invert_state=colour != 'BLACK')
    return get_full_policy_net.__dict__[model_path + colour]


def get_player(player, colour, args):
    """Gets a callable that returns a move, finding it in the appropriate
    place."""
    if player == 'human':
        return hexgame.human_text_policy
    elif player == 'uct':
        return classic_uct_policy(max_depth=args.tree_depth,
                                  search_time=args.search_time,
                                  colour=colour)
    elif player == 'uct-nohash':
        return classic_uct_policy(max_depth=args.tree_depth,
                                  search_time=args.search_time,
                                  colour=colour,
                                  do_hash=False)
    elif player == 'policy-net-random':
        # the policy net bootstrap
        return get_full_policy_net(args.policy_net_random_path, colour)
    elif player == 'policy-net-tuned':
        return get_full_policy_net(args.policy_net_self_path, colour)
    elif player == 'random':
        return random_policy
    else:
        raise ValueError('I do not know this player: {}'.format(player))


def get_args():
    """set up the arguments with argparse. Could use tf, but there's no need
    to import stuff we don't need, want this to be nice and easy to run."""
    parser = argparse.ArgumentParser(description="play a games")
    parser.add_argument('--player1', '-p1', default='human', type=str,
                        choices=PLAYERS)
    parser.add_argument('--player2', '-p2', default='human', type=str,
                        choices=PLAYERS)

    render_parser = parser.add_mutually_exclusive_group(required=False)
    render_parser.add_argument('--render', dest='render', action='store_true')
    render_parser.add_argument('--no_render', dest='render',
                               action='store_false')
    parser.set_defaults(render=True)

    parser.add_argument('--num_games', '-n', default=1, type=int,
                        help='How many games to play')
    parser.add_argument('--illegal_move_mode', default='lose', type=str,
                        choices=['raise', 'tie', 'lose'])
    parser.add_argument('--search_time', default=1.0, type=float,
                        help='How long to search for, for those that will '
                        'just go until stopped.')
    parser.add_argument('--tree_depth', default=1000000, type=int,
                        help='For the tree searches, how far down to expand '
                        'the tree before going straight to rollouts.')

    parser.add_argument('--policy_net_random_path', type=str)
    parser.add_argument('--policy_net_self_path', type=str)
    return parser.parse_args()


def get_environment(opponent, illegal_move_mode):
    """Gets an environment with the appropriate opponent installed"""
    try:
        get_environment.env.opponent = opponent
    except AttributeError:
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
        get_environment.env = gym.make('PrettyHexEnv-v0')
    return get_environment.env


def print_results(p1_wins, p2_wins, games):
    """make some pretty output"""
    twidth = shutil.get_terminal_size().columns
    print('~'*twidth)
    print('~'*twidth)
    print('{:~^{}}'.format(
        'Results after {} games'.format('\033[1m{}\033[0m'.format(games)),
        twidth))
    print('\033[1m{}\033[0m: {} ({:.2%})'.format(
        'Player 1', p1_wins, p1_wins/games))
    print('\033[1m{}\033[0m: {} ({:.2%})'.format(
        'Player 2', p2_wins, p2_wins/games))
    print('~'*twidth)
    print('~'*twidth)


def main():
    """load up the args and play a game"""
    gym.undo_logger_setup()
    logging.basicConfig(level=logging.INFO,
                        format='{asctime}[{levelname}]({module}): {message}',
                        style='{',
                        stream=sys.stdout)

    args = get_args()
    logging.info('%s playing %s',
                 args.player1, args.player2)
    p1_wins, p2_wins = 0, 0
    for game_num in range(args.num_games):
        logging.info('Game %d getting going', game_num)
        p1_colour = 'BLACK' if game_num % 2 == 0 else 'WHITE'
        p2_colour = 'WHITE' if p1_colour == 'BLACK' else 'BLACK'
        player_1 = get_player(args.player1, p1_colour, args)
        player_2 = get_player(args.player2, p2_colour, args)

        black_player = player_1 if p1_colour == 'BLACK' else player_2
        white_player = player_2 if p1_colour == 'BLACK' else player_2

        logging.info('Player 1 (%s) is %s', args.player1, p1_colour)
        logging.info('Player 2 (%s) is %s', args.player2, p2_colour)

        env = get_environment(black_player, args.illegal_move_mode)

        hexgame.ENV = env

        obs = env.reset()

        done = False
        while not done:
            if args.render:
                env.render()
            white_action = white_player(obs)

            observation, reward, done, info = env.step(white_action)

        if reward == 1.0:
            if p1_colour == 'BLACK':
                p1_wins += 1
                winner = 'player 1'
            else:
                p2_wins += 1
                winner = 'player 2'
        else:  # white has won
            if p1_colour == 'BLACK':
                p2_wins += 1
                winner = 'player 2'
            else:
                p1_wins += 1
                winner = 'player 1'
        logging.info('%s is the winner', winner)

    print_results(p1_wins, p2_wins, args.num_games)


if __name__ == '__main__':
    main()
