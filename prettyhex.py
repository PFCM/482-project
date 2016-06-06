"""Subclass of Gym's hex with hopefully slightly nicer rendering."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from gym.envs import board_game


class PrettyHexEnv(board_game.HexEnv):

    ENDC = '\033[0m'  # end all attributes
    BLUE = '\033[1;34m'
    RED  = '\033[1;31m'
    BOLD = '\033[1m'
    BLACKONWHITE = '\033[30;47m'
    BOLDBLACKONWHITE = '\033[1;30;47m'
    WHITE_BACK = '\033[47m'

    def __init__(self, *args, **kwargs):
        super(PrettyHexEnv, self).__init__(*args, **kwargs)

    def _render(self, mode='human', close=False):
        # pretty much copied wholesale from
        # https://github.com/openai/gym/blob/master/gym/envs/board_game/hex.py
        if close:
            return

        board = self.state

        outfile = sys.stdout
        outfile.write(' ' * 5 + PrettyHexEnv.BOLD)
        for j in range(board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write(
            '-' * (board.shape[1] * 6 - 1) +
            PrettyHexEnv.ENDC)
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(
                PrettyHexEnv.BOLD +
                ' ' * (2 + i * 3) +  str(i + 1) + '  |' +
                PrettyHexEnv.ENDC)
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write(
                        PrettyHexEnv.BLUE + '  B  ' + PrettyHexEnv.ENDC)
                else:
                    outfile.write(
                        PrettyHexEnv.RED + '  W  ' + PrettyHexEnv.ENDC)
                outfile.write(PrettyHexEnv.BOLD + '|' + PrettyHexEnv.ENDC)
            outfile.write('\n')
            outfile.write(' ' * (i * 3 + 1))
            outfile.write(
                PrettyHexEnv.BOLD +
                '-' * (board.shape[1] * 7 - 1))
            outfile.write(PrettyHexEnv.ENDC)
            outfile.write('\n')

if __name__ == '__main__':
    # test the colours
    print(PrettyHexEnv.BLUE + 'blue' + PrettyHexEnv.ENDC)
    print(PrettyHexEnv.RED + 'red' + PrettyHexEnv.ENDC)
    print(PrettyHexEnv.BOLD + 'bold' + PrettyHexEnv.ENDC)
    print(PrettyHexEnv.BLACKONWHITE + 'black on white' + PrettyHexEnv.ENDC)
