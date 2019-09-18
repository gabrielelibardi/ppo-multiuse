""" Create a basic test set. """

import random
from utils import (
    create_c1_arena,
    create_c2_arena,
    create_c3_arena,
    create_c4_arena,
    create_c5_arena,
    create_c6_arena,
    create_c7_arena,
)

if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas test directory')

    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)

    info_rewards = {}

    # c1
    for i in range(1, 11):
            create_c1_arena(arguments.target_dir, 'c1_{}'.format(
                str(i).zfill(2)), max_reward=5)

    # c2
    for i in range(1, 11):
            create_c2_arena(arguments.target_dir, 'c2_{}'.format(
                str(i).zfill(2)), max_reward=5)

    # c3
    for i in range(1, 11):
        create_c3_arena(arguments.target_dir, 'c3_{}'.format(
            str(i).zfill(2)), time=random.choice([250, 500, 1000]))

    # c4
    for i in range(1, 11):
        create_c4_arena(arguments.target_dir, 'c4_{}'.format(
            str(i).zfill(2)), time=random.choice([250, 500, 1000]),
                        num_red_zones=2, max_orange_zones=1)

    # c5
    for i in range(1, 11):
        create_c5_arena(arguments.target_dir, 'c5_{}'.format(str(i).zfill(2)),
                        time=random.choice([250, 500, 1000]))

    # c6
    for i in range(1, 11):
        create_c6_arena(arguments.target_dir, 'c6_{}'.format(str(i).zfill(2)),
                        time=random.choice([250, 500, 1000]))

    # c7
    for i in range(1, 11):
        create_c7_arena(arguments.target_dir, 'c7_{}'.format(str(i).zfill(2)),
                        time=random.choice([250, 500, 1000]))
