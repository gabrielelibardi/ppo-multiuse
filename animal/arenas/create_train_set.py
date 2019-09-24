""" Create a train set. """

import random
from utils import (
    create_c1_arena,
    create_c1_arena_weird,
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
        '-d', '--target-dir', help='path to arenas train directory')

    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)

    info_rewards = {}

    # c1
    for i in range(1, 501):
            create_c1_arena(arguments.target_dir, 'c1_{}'.format(
                str(i).zfill(3)), max_reward=5, is_train=True)

    # c1_weird
    for i in range(501, 701):
        create_c1_arena_weird(arguments.target_dir, 'c1_{}'.format(
            str(i).zfill(3)), is_train=True)

    # c2
    for i in range(1, 501):
            create_c2_arena(arguments.target_dir, 'c2_{}'.format(
                str(i).zfill(3)), max_reward=5, is_train=True)

    # c3
    for i in range(1, 501):
        create_c3_arena(arguments.target_dir, 'c3_{}'.format(
            str(i).zfill(3)), time=random.choice([250, 500, 1000]),
                        is_train=True)

    # c4
    for i in range(1, 501):
        create_c4_arena(arguments.target_dir, 'c4_{}'.format(
            str(i).zfill(3)), time=random.choice([250, 500, 1000]),
                        num_red_zones=8, max_orange_zones=3, is_train=True)

    # c5
    for i in range(1, 501):
        create_c5_arena(arguments.target_dir, 'c5_{}'.format(str(i).zfill(3)),
                        time=random.choice([250, 500, 1000]), is_train=True)

    # c6
    for i in range(1, 11):
        create_c6_arena(arguments.target_dir, 'c6_{}'.format(str(i).zfill(3)),
                        time=random.choice([250, 500, 1000]), is_train=True)

    # c7
    for i in range(1, 501):
        create_c7_arena(arguments.target_dir, 'c7_{}'.format(str(i).zfill(3)),
                        time=random.choice([250, 500, 1000]), is_train=True)
