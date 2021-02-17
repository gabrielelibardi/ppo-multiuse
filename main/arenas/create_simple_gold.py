""" Create a train set. """
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../..')

import random
import numpy as np
from animal.arenas.utils import create_gold


if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas train directory')

    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)


    skills = ["box_reasoning"]


    # box reasoning
    for i in range(1, 1000):
        
   
        create_gold( "{}/".format(arguments.target_dir),
            'c2_{}'.format(str(i).zfill(4)))
