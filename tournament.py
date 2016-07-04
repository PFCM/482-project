"""Quick script to run a tournament to find the best algorithm.

Players to test are:
    random
    uct
    policy net
    policy net with self play
    mcts-policy-net
    mcts self-play policy net

There are six, we want to do a round robin and then semis and a final.
So the games are:
    - random v uct
    - random v policy net
    - random v policy net self
    - random v mcts random
    - random v mcts self
    - uct v policy
    - uct v policy self
    - uct v mcts policy
    - uct v mcts self
    - policy v policy self
    - policy v mcts policy
    - policy v mcts policy self
    - policy self v mcts policy
    - policy self v mcts policy self
    - mcts policy v mcts policy self

get top 4 and have semi finals
final

Also we certainly going to try our best to play games in parallel on account of
them taking an awfully long time. As well as each `game` should probably be
5 or 7 or even more actual games, swapping players each time.
"""
import os
import subprocess
import itertools
import time
import sys


# the actual flags that get sent in, so make sure they line up
ALGOLS = [
    'random',
    'uct',
    'policy-net-random',
    'policy-net-tuned',
    'mcts-policy-net-random',
    'mcts-policy-net-tuned'
]

# flags we will always need
common_args = [
    '--exit_result',
    # nb
    '--policy_net_self_path',
    os.path.abspath(
        '/Users/pfcmathews/Google Drive/Uni/MATH482/project/models/convolutional'
        '/self/-46400'),
    '--policy_net_random_path',
    os.path.abspath(
        '/Users/pfcmathews/Google Drive/Uni/MATH482/project/models/convolutional'
        '/bootstrap/bootstrap_fixed-16368'),
    '--tree_depth', '10',  # could experiment
    '--search_time', '10',
    '--illegal_move_mode', 'lose',
    '--num_games', '7',
    '--no_render'
]

results_dir = 'tournament_results'


def round_robin():
    """make everyone play one game against each other once"""
    # get going
    processes = []
    print('Beginning round robin phase')
    os.makedirs(os.path.join(results_dir, 'round_robin'),
                exist_ok=True)
    points = {player: 0 for player in ALGOLS}
    points['error'] = 0
    for player1, player2 in itertools.combinations(ALGOLS, 2):
        print('  {} vs. {}'.format(player1, player2))
        game_path = os.path.join(
            results_dir, 'round_robin', '{}_v_{}'.format(player1, player2))
        processes.append((
            subprocess.Popen([
                sys.executable,
                'master_hex.py',
                '--player1', player1,
                '--player2', player2,
                '--result_path', game_path] + common_args),
            player1,
            player2))
        if 'net' in player1 or 'net' in player2:
            # can't do the tf games in parallel, it freaks all out
            proc = processes[-1]
            winner = proc[0].wait()
            processes.remove(proc)
            if winner > 1:
                winner -= 1
                print('   {} v {} finished, {} won.'.format(
                    proc[1], proc[2], proc[winner]))
                points[proc[winner]] += 1
            else:
                print('Error.')
                points['error'] += 1


    print('Tidying up')
    while processes:
        for proc in processes:
            winner = proc[0].poll()
            if winner is not None:
                processes.remove(proc)
                if winner > 1:
                    winner -= 1
                    print('   {} v {} finished, {} won.'.format(
                        proc[1], proc[2], proc[winner]))
                    points[proc[winner]] += 1
                else:
                    print('Error')
                    points['error'] += 1
                    break
        time.sleep(0.1)
    return points


def run_tournament():
    """does the thing"""
    # first we do the round robin
    results = round_robin()
    print('Round robin results:')
    for player in results:
        print('{:>30}:{}'.format(player, results[player]))


if __name__ == '__main__':
    run_tournament()
