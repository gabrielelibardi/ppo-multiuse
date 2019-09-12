
""" Create sets C1 to C10 of test arenas with known max reward for testing. """

import numpy as np
from .edit_arena import add_object, write_arena
from .sample_features import random_size_rewards

objects_dict = {
    'reward_objects': [
        'GoodGoal',
        'GoodGoalBounce',
        'BadGoal',
        'BadGoalBounce',
        'GoodGoalMulti',
        'GoodGoalMultiBounce'
    ],
    'immovable_objects': [
        'Wall',
        'Ramp',
        'CylinderTunnel',
        'WallTransparent',
        'CylinderTunnelTransparent'],
    'movable_objects': [
        'Cardbox1',
        'Cardbox2',
        'UObject',
        'LObject',
        'LObject2'
    ],
    'zone_objects': [
        'DeathZone',
        'HotZone'
    ],
}


def create_c1_arena(target_path, arena_name, max_reward=20, time=250, max_num_good_goals=1):
    """
    Create .yaml file for C1-type arena.
     - Only goals.
     - Fixed random size for all goals.
     - At least one green ball.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
   """

    allowed_objects = objects_dict['reward_objects']
    size_goal = (np.clip(random_size_rewards()[0], 1.0, max_reward), 0.0, 0.0)
    reward = float(max_reward)
    arena = add_object('', 'GoodGoal', size=size_goal)

    num_goals = 1
    worst_goal = 0.0
    min_reward = 0.0
    best_goal = size_goal[0]

    while reward - best_goal > size_goal[0]:

        category = allowed_objects[np.random.randint(0, len(allowed_objects))]

        if category in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            reward -= size_goal[0]

        if category in ['BadGoal', 'BadGoalBounce']:
            worst_goal = min(worst_goal, size_goal[0])

        if category in ['GoodGoal', 'GoodGoalBounce']:
            best_goal = max(best_goal, size_goal[0])
            if num_goals >= max_num_good_goals:
                continue
            num_goals += 1

        arena = add_object(arena, category, size=size_goal)

    min_reward -= worst_goal
    reward = reward - best_goal
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c1'


def create_c2_arena(target_path, arena_name, max_reward=20, time=250, max_num_good_goals=1):
    """
    Create .yaml file for C2-type arena.
     - Only goals.
     - Different random size for all goals.
     - At least one green and one red ball.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
   """

    allowed_objects = objects_dict['reward_objects']
    reward = float(max_reward)

    size_goal = (np.clip(random_size_rewards()[0], 1.0, max_reward), 0.0, 0.0)
    arena = add_object('', 'GoodGoal', size=size_goal)
    best_goal = size_goal[0]

    size_goal = (random_size_rewards()[0], 0.0, 0.0)
    arena = add_object(arena, 'BadGoal', size=size_goal)
    worst_goal = size_goal[0]

    num_goals = 1
    min_reward = 0.0

    while reward - best_goal > size_goal[0]:

        size_goal = (random_size_rewards()[0], 0.0, 0.0)
        category = allowed_objects[np.random.randint(0, len(allowed_objects))]

        if category in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            reward -= size_goal[0]

        if category in ['BadGoal', 'BadGoalBounce']:
            worst_goal = min(worst_goal, size_goal[0])

        if category in ['GoodGoal', 'GoodGoalBounce']:
            if num_goals >= max_num_good_goals:
                continue
            best_goal = max(best_goal, size_goal[0])
            num_goals += 1

        arena = add_object(arena, category, size=size_goal)

    min_reward -= worst_goal
    reward = reward - best_goal
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c2'


def create_c3_arena(target_path, arena_name, max_reward=20):
    """
    - Minimum one green ball, random sized
    - With probability 0.5 add yellow ball, random sized
    - Add multiple immovable and immovable objects
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    return 'c3'


def create_c4_arena(target_path, arena_name, max_reward=20):
    """
    - 1 green food (stationary) and 1-2 red zones
    - maybe add other objects with low probability
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    return 'c4'
