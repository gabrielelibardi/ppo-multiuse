from .utils import create_c1_arena, create_c2_arena

if __name__ == '__main__':

    import os
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas test directory')

    arguments = parser.parse_args()
    arguments.target_dir = "/Users/albertbou/tests"
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)

    info_rewards = {}

    for i in range(1, 30):
        info_rewards.update(
            create_c1_arena(arguments.target_dir, 'c1_{}'.format(
                str(i).zfill(2)), max_reward=i))

    for i in range(1, 30):
        info_rewards.update(
            create_c2_arena(arguments.target_dir, 'c2_{}'.format(
                str(i).zfill(2)), max_reward=i))

    json.dump(info_rewards, open(
        arguments.target_dir + '/info_rewards.json', 'w'), indent=4)
