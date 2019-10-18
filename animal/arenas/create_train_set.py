""" Create a train set. """

import random
import numpy as np
from animal.arenas.utils import (
    create_c1_arena,
    create_c1_arena_weird,
    create_c2_arena,
    create_c3_arena,
    create_c3_arena_basic,
    create_c4_arena,
    create_c5_arena,
    create_c6_arena,
    create_c6_arena_basic,
    create_c7_arena,
    create_maze,
    create_arena_choice,

    # skills
    create_arena_cross,
    create_arena_push1,
    create_arena_push2,
    create_arena_tunnel1,
    create_arena_tunnel2,
    create_arena_ramp1,
    create_arena_ramp2,
    create_arena_ramp3,
    create_arena_narrow_spaces_1,
    create_arena_narrow_spaces_2,
    create_arena_pref1,
    create_blackout_test_1
)

if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas train directory')
    parser.add_argument(
        '-u', '--unify', action='store_true', default=False,
        help='Save all arenas in the same directory')

    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)

    if arguments.unify:
        save_in = arguments.target_dir
    else:
        for skills in ["preferences", "push", "ramps", "narrow", "zones",
                       "tunnels", "navigate", "generalize", "internal_model",
                       "mazes", "choices"]:
            if not os.path.isdir("{}/{}".format(arguments.target_dir, skills)):
                os.mkdir("{}/{}".format(arguments.target_dir, skills))

    # c1
    for i in range(1, 201):
            create_c1_arena(arguments.target_dir if arguments.unify
                            else "{}preferences/".format(arguments.target_dir),
                            'c1_{}'.format(str(i).zfill(4)),
                            max_reward=float(np.random.randint(5, 10)),
                            time=random.choice([250, 500]), is_train=True)

    # c2
    for i in range(1, 201):
            create_c2_arena(arguments.target_dir if arguments.unify
                            else "{}preferences/".format(arguments.target_dir),
                            'c2_{}'.format(str(i).zfill(4)),
                            max_reward=float(np.random.randint(5, 10)),
                            time=random.choice([250, 500]), is_train=True,
                            max_num_good_goals=np.random.randint(1, 2))

    # c3
    for i in range(1, 201):
        create_c3_arena(arguments.target_dir if arguments.unify
                        else "{}navigate/", 'c3_{}'.format(
            str(i).zfill(4)), time=random.choice([500, 1000]), is_train=True)

    # c3
    for i in range(501, 701):
        create_c3_arena_basic(arguments.target_dir if arguments.unify
                              else "{}navigate/".format(arguments.target_dir),
                              'c3_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]),
                              num_walls=np.random.randint(5, 15),
                              is_train=True)

    # c4
    for i in range(1, 501):
        create_c4_arena(arguments.target_dir if arguments.unify
                        else "{}zones/".format(arguments.target_dir),
                        'c4_{}'.format(str(i).zfill(4)),
                        time=random.choice([500, 1000]),
                        num_red_zones=8, max_orange_zones=3, is_train=True)

    # c5
    for i in range(1, 500):
        create_c5_arena(arguments.target_dir if arguments.unify
                        else "{}ramps/".format(arguments.target_dir),
                        'c5_{}'.format(str(i).zfill(4)),
                        time=random.choice([500, 1000]), is_train=True)

    # c6
    for i in range(1, 201):
        create_c6_arena(arguments.target_dir if arguments.unify
                        else "{}generalize/".format(arguments.target_dir),
                        'c6_{}'.format(str(i).zfill(4)),
                        time=random.choice([500, 1000]), is_train=True)

    # c6
    for i in range(501, 701):
        create_c6_arena_basic(arguments.target_dir if arguments.unify
                              else "{}generalize/".format(arguments.target_dir),
                              'c6_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              num_walls=np.random.randint(5, 15))

    # c7
    for i in range(1, 301):
        create_c7_arena(arguments.target_dir if arguments.unify
                        else "{}internal_model/".format(arguments.target_dir),
                        'c7_{}'.format(str(i).zfill(4)),
                        time=random.choice([500, 1000]), is_train=True)

    # mazes
    for i in range(1, 1001):
        create_maze(arguments.target_dir if arguments.unify
                    else "{}mazes/".format(arguments.target_dir),
                    'c8_{}'.format(str(i).zfill(4)),
                    time=random.choice([500, 1000]), num_cells=2,
                    obj=random.choice(['CylinderTunnel', 'door', 'Cardbox1']),
                    is_train=True)

    # choice
    for i in range(1, 400):
        create_arena_choice(arguments.target_dir if arguments.unify
                            else "{}choices/".format(arguments.target_dir),
                            'c9_{}'.format(str(i).zfill(4)),
                            time=random.choice([500, 1000]), is_train=True)

    # cross
    for i in range(1, 501):
        create_arena_cross(arguments.target_dir if arguments.unify
                           else "{}navigate/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]), is_train=True)

    # push1
    for i in range(501, 1001):
        create_arena_push1(arguments.target_dir if arguments.unify
                           else "{}push/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]),
                           is_train=True)

    # push2
    for i in range(1001, 1501):
        create_arena_push2(arguments.target_dir if arguments.unify
                           else "{}push/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]),
                           is_train=True)

    # tunnel1
    for i in range(1501, 2001):
        create_arena_tunnel1(arguments.target_dir if arguments.unify
                             else "{}tunnels/".format(arguments.target_dir),
                             'c10_{}'.format(str(i).zfill(4)),
                             time=random.choice([500, 1000]),
                             is_train=True)
    # tunnel2
    for i in range(2001, 2501):
        create_arena_tunnel2(arguments.target_dir if arguments.unify
                             else "{}tunnels/".format(arguments.target_dir),
                             'c10_{}'.format(str(i).zfill(4)),
                             time=random.choice([500, 1000]),
                             is_train=True)

    # ramp1
    for i in range(2501, 3001):
        create_arena_ramp1(arguments.target_dir if arguments.unify
                           else "{}ramps/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]),
                           is_train=True)
    # ramp2
    for i in range(3001, 3501):
        create_arena_ramp2(arguments.target_dir if arguments.unify
                           else "{}ramps/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]),
                           is_train=True)
    # ramp3
    for i in range(3501, 4001):
        create_arena_ramp3(arguments.target_dir if arguments.unify
                           else "{}ramps/".format(arguments.target_dir),
                           'c10_{}'.format(str(i).zfill(4)),
                           time=random.choice([500, 1000]),
                           is_train=True)

    # narrow1
    for i in range(4001, 4501):
        create_arena_narrow_spaces_1(arguments.target_dir if arguments.unify
                                     else "{}narrow/".format(arguments.target_dir),
                                     'c10_{}'.format(str(i).zfill(4)),
                                     time=random.choice([500, 1000]),
                                     is_train=True)

    # narrow1
    for i in range(4501, 5001):
        create_arena_narrow_spaces_2(arguments.target_dir if arguments.unify
                                     else "{}narrow/".format(arguments.target_dir),
                                     'c10_{}'.format(str(i).zfill(4)),
                                     time=random.choice([500, 1000]),
                                     is_train=True)

    # preference/ with U walls
    for i in range(5001, 5501):
        create_arena_pref1(arguments.target_dir if arguments.unify
                           else "{}/narrow/".format(arguments.target_dir),
                           'c2_{}'.format(str(i).zfill(4)),
                           time=random.choice([1000]),
                           is_train=True)

    # light 10 frames than black, transparent walls
    for i in range(5501, 6001):
        create_blackout_test_1(arguments.target_dir if arguments.unify
                               else "{}/narrow/".format(arguments.target_dir),
                               'c7_{}'.format(str(i).zfill(4)),
                               time=random.choice([1000]),
                               is_train=True)
