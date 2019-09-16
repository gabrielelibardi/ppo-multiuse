
""" Create sets C1 to C10 of test arenas with known max reward for testing. """

import random
import numpy as np
from .edit_arenas import add_object, write_arena
from .sample_features import random_size, random_color
from .edit_arenas import add_ramp_scenario

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


def create_c1_arena(target_path, arena_name, max_reward=5, time=250, max_num_good_goals=1):
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
    size_goal = (np.clip(random_size('GoodGoal')[0], 1.0, max_reward), 0.0, 0.0)
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


def create_c2_arena(target_path, arena_name, max_reward=5, time=250, max_num_good_goals=1):
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

    size_goal = (np.clip(random_size('GoodGoal')[0], 1.0, max_reward), 0.0, 0.0)
    arena = add_object('', 'GoodGoal', size=size_goal)
    best_goal = size_goal[0]

    size_goal = random_size('BadGoal')
    arena = add_object(arena, 'BadGoal', size=size_goal)
    worst_goal = size_goal[0]

    num_goals = 1
    min_reward = 0.0

    while reward - best_goal > size_goal[0]:

        category = allowed_objects[np.random.randint(0, len(allowed_objects))]
        size_goal = random_size(category)

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


def create_c3_arena(target_path, arena_name, time=250, num_movable=1, num_immovable=1):
    """
    - One random positive reward ball, random sized
    - With probability 0.5 add red ball, random sized
    - Add multiple movable and immovable objects
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])

    size_goal = random_size(category)
    arena = add_object('', category, size=size_goal)

    if random.random() > 0.5:
        category = random.choice(['BadGoal', 'BadGoalBounce'])
        size_goal = random_size(category)
        arena = add_object(arena, category, size=size_goal)

    for _ in range(num_movable):
        category = random.choice(objects_dict['movable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object)

    for _ in range(num_immovable):
        category = random.choice(objects_dict['immovable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c3'


def create_c4_arena(target_path, arena_name, time=250, num_red_zones=2, max_orange_zones=1, max_movable=3, max_immovable=3):
    """
    - 1 green food (stationary) and some red zones
    - add orange zone with probability 0.5
    - add immobable object with probability 0.1
    - add movable object with probability 0.1
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    size_goal = random_size('GoodGoal')
    arena = add_object('', 'GoodGoal', size=size_goal)

    if random.random() > 0.5:
        size_goal = random_size('GoodGoalMulti')
        arena = add_object(arena, 'GoodGoalMulti', size=size_goal)

    for _ in range(num_red_zones):
        size_object = random_size('DeathZone')
        arena = add_object(arena, 'DeathZone', size=size_object)

    for _ in range(max_orange_zones):
        if random.random() > 0.5:
            size_object = random_size('HotZone')
            arena = add_object(arena, 'HotZone', size=size_object)

    for _ in range(max_movable):
        if random.random() > 0.9:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category)
            arena = add_object(arena, category, size=size_object)

    for _ in range(max_immovable):
        if random.random() > 0.9:
            category = random.choice(objects_dict['immovable_objects'])
            size_object = random_size(category)
            arena = add_object(arena, category, size=size_object)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c4'


def create_c5_arena(target_path, arena_name, time=250, num_movable=1, num_immovable=1):

    arena = ''
    arena = add_ramp_scenario(arena)

    for _ in range(num_movable):
        category = random.choice(objects_dict['movable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object,
                           RGB=random_color())

    for _ in range(num_immovable):
        category = random.choice(['Wall', 'Ramp', 'CylinderTunnel'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object,
                           RGB=random_color())

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c5'


def create_c6_arena(target_path, arena_name, time=250, num_movable=1, num_immovable=1):
    """
    - One random positive reward ball, random sized
    - add a second positive rewards , with probability 0.5
    - add up to 2 red balls , with probability 0.5 each
    - With probability 0.5 add red balls, random sized
    - Add multiple movable and immovable objects (with random color)
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    category = random.choice(['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])

    size_goal = random_size(category)
    arena = add_object('', category, size=size_goal)

    if random.random() > 0.5:
        category = random.choice(['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])
        size_goal = random_size(category)
        arena = add_object(arena, category, size=size_goal)

    for _ in range(2):
        if random.random() > 0.5:
            category = random.choice(['BadGoal', 'BadGoalBounce'])
            size_goal = random_size(category)
            arena = add_object(arena, category, size=size_goal)

    for _ in range(num_movable):
        category = random.choice(objects_dict['movable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object, RGB=random_color())

    for _ in range(num_immovable):
        category = random.choice(['Wall', 'Ramp', 'CylinderTunnel'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object, RGB=random_color())

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c6'


def create_c7_arena(target_path, arena_name, time=250, num_movable=1, num_immovable=1): # Not correct!
    """
    - One random positive reward ball, random sized
    - add a second positive rewards , with probability 0.5
    - add up to 2 red balls , with probability 0.5 each
    - With probability 0.5 add red balls, random sized
    - Add multiple movable and immovable objects
    - random blackouts
    :param target_path:
    :param arena_name:
    :param max_reward:
    :return:
    """

    blackout_options = [[-20], [-40], [-60], [25, 30, 50, 55, 75], [50, 55, 75, 80, 100, 105, 125]]
    category = random.choice(['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])

    size_goal = random_size(category)
    arena = add_object('', category, size=size_goal)

    if random.random() > 0.5:
        category = random.choice(['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])
        size_goal = random_size(category)
        arena = add_object(arena, category, size=size_goal)

    for _ in range(2):
        if random.random() > 0.5:
            category = random.choice(['BadGoal', 'BadGoalBounce'])
            size_goal = random_size(category)
            arena = add_object(arena, category, size=size_goal)

    for _ in range(num_movable):
        category = random.choice(objects_dict['movable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object)

    for _ in range(num_immovable):
        category = random.choice(objects_dict['immovable_objects'])
        size_object = random_size(category)
        arena = add_object(arena, category, size=size_object)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena, blackouts=random.choice(blackout_options))

    return 'c7'
