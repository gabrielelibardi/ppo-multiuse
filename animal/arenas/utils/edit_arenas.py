import random
import numpy as np
from .sample_features import random_size, random_pos, \
    sample_position_with_respect_to, random_color


def add_object(s, object_name, pos=None, size=None, RGB=None, rot=None):
    s += "    - !Item \n      name: {} \n".format(object_name)

    if RGB is None:
        if object_name is 'Ramp':
            RGB = (255, 0, 255)
        elif object_name is 'Tunnel':
            RGB = (153, 153, 153)
        elif object_name is 'Wall':
            RGB = (153, 153, 153)

    if pos is not None:
        s += "      positions: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(
            pos[0], pos[1], pos[2])
    if size is not None:
        s += "      sizes: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(
            size[0], size[1], size[2])
    if RGB is not None:
        s += "      colors: \n      - !RGB {{r: {}, g: {}, b: {}}}\n".format(
            RGB[0], RGB[1], RGB[2])
    if rot is not None:
        s += "      rotations: [{}]\n".format(rot)
    return s


def write_arena(fname, time, arena_str, blackouts=None):
    blackouts_str = "blackouts: {} \n    ".format(
        blackouts) if blackouts else ""
    with open("{}.yaml".format(fname), 'w+') as f:
        f.write(
            "!ArenaConfig \narenas: \n  0: !Arena \n    t: {} \n    {}items:\n".format(
                time, blackouts_str))
        f.write(arena_str)


def add_ramp_scenario(arena, is_train=False):
    # create a wall as a platform
    category = 'Wall'

    size_wall = (
        np.clip(random_size(category)[0], 2, 5),
        np.clip(random_size(category)[1], 2, 3),
        np.clip(random_size(category)[2], 2, 5))

    rotation_wall = random.choice([0])
    position_wall = sample_position_with_respect_to((20, 0, 20), 'far')
    arena = add_object(arena, category, size=size_wall, pos=position_wall,
                       RGB=(0, 0, 255), rot=rotation_wall)

    # locate a reward on the wall
    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    size_object = random_size(category)
    pos_goal = (
        position_wall[0], size_wall[1] + size_object[1] + 0.1,
        position_wall[2])
    arena = add_object(arena, category, size=size_object, RGB=(0, 0, 255),
                       pos=pos_goal)
    pos_agent = sample_position_with_respect_to(pos_goal, random.choice(
        ['close', 'medium', 'far']))

    # create ramps to access the reward
    category = 'Ramp'

    # randomly choose how many ramps
    for _ in range(3):
        for rot, pos_shift, size, lol in zip(
                [0, 90, 180, 270],
                [(0.0, 0.5), (0.5, 0.0), (0.0, -0.5), (-0.5, 0.0)],
                [(0, 2), (2, 0), (0, 2), (2, 0)],
                [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]):

            size_object = (
                max(random_size(category)[0], size_wall[1] * lol[1]),
                size_wall[1],
                max(random_size(category)[2], size_wall[1] * lol[0]))

            if random.random() > 0.0:
                position_object = (
                    position_wall[0] + (
                        size_wall[0] + size_object[size[0]]) * pos_shift[0],
                    0.0,
                    position_wall[2] + (
                        size_wall[2] + size_object[size[1]]) * pos_shift[1])

                arena = add_object(
                    arena, category, size=size_object, rot=rot,
                    pos=position_object)

    if random.random() < 0.1:
        pos_agent = position_object

    arena = add_object(arena, "Agent",
                       pos=(pos_agent[0], size_wall[1] + 0.1, pos_agent[2]))

    return arena


def add_choice(arena, is_train=False):
    # create platform in the middle

    height = np.random.randint(1, 10)
    category = 'Wall'
    size_wall = (4, 1, 4)
    position_wall = (20., 0., 20.)
    rotation_wall = 0.
    arena = add_object(
        arena, category, size=size_wall, pos=position_wall,
        rot=rotation_wall, RGB=(0, 0, 255))

    # create walls to divide the arena
    size_wall = (0.5, height, 18)
    for position_wall, rotation_wall in zip(
            [(9., 0., 20.), (31., 0., 20.), (20., 0., 9), (20., 0., 31)],
            [90., 270., 0., 180.]):
        arena = add_object(
            arena, category, size=size_wall, pos=position_wall,
            rot=rotation_wall)

    # locate agent in the platform
    position_agent = (20., 5., 20.)
    arena = add_object(arena, "Agent", pos=position_agent)

    return arena


def add_walled(arena, num_walls=1, random_rgb=False):
    category = 'Wall'

    for _ in range(num_walls):
        position_wall = sample_position_with_respect_to((20, 0, 20), 'medium')
        rotation_wall = random.choice([0, 90, 180, 360])
        size_wall = (0.5, 5, 10)
        arena = add_object(
            arena, category, size=size_wall, pos=position_wall,
            rot=rotation_wall, RGB=random_color() if random_rgb else None)

    return arena


def need_move(arena):
    # needs to learn to move an object to reach goal

    return arena


def create_wall(A, B,z_size = 5, obj='CylinderTunnel', gap=2):
    # A is the statrting point, B is  the endpoint
    # dor can be empty, cylinder, ramp, box

    Ax, Ay = A
    Bx, By = B

    if Bx == Ax:
        if Ay != 0 and Ay != 40:
            Ay = Ay + 0.5
        if By != 0 and By != 40:
            By = By - 0.5

        x_size = 1
        y_size = round((By - Ay), 2)

        y_pos = round(Ay + (By - Ay) / 2, 2)
        x_pos = Bx
        z_pos = 0

        # print('x_pos',x_pos,'y_pos',y_pos,'x_size',x_size,'y_size',y_size)
        if obj == 'door':
            y_size_1 = (y_size - gap) / 2
            y_size_2 = (y_size - gap) / 2

            y_pos_1 = y_pos - gap / 2 - (y_size - gap) / 4
            y_pos_2 = y_pos + gap / 2 + (y_size - gap) / 4

            return (
            (x_size, z_size, y_size_1), (x_pos, z_pos, y_pos_1), (x_size, z_size, y_size_2), (x_pos, z_pos, y_pos_2))

        if obj == 'CylinderTunnel':
            return ((1, gap, gap), (x_pos, 0.5, y_pos))
        if obj == 'Cardbox2':
            gap_box = gap - 0.2
            return ((gap_box, gap_box, gap_box), (x_pos + 1.5, 0.5, y_pos))
        if obj == 'Cardbox1':
            gap_box = gap - 0.2
            return ((gap_box, gap_box, gap_box), (x_pos + 1.5, 0.5, y_pos))
        if obj == 'Ramp':
            return ((gap, 2, 3), (x_pos, 0.5, y_pos))

    if By == Ay:


        if Ax != 0 and Ax != 40:
            Ax = Ax + 0.5
        if Bx != 0 and Bx != 40:
            Bx = Bx - 0.5

        y_size = 1
        x_size = round((Bx - Ax), 2)

        x_pos = round(Ax + (Bx - Ax) / 2, 2)
        y_pos = By
        z_pos = 0

        #print('x_pos', x_pos, 'y_pos', y_pos, 'x_size', x_size, 'y_size', y_size)
        if obj == 'door':
            x_size_1 = (x_size - gap) / 2
            x_size_2 = (x_size - gap) / 2

            x_pos_1 = x_pos - gap / 2 - (x_size - gap) / 4
            x_pos_2 = x_pos + gap / 2 + (x_size - gap) / 4

            return (
            (x_size_1, z_size, y_size), (x_pos_1, z_pos, y_pos), (x_size_2, z_size, y_size), (x_pos_2, z_pos, y_pos))

        if obj == 'CylinderTunnel':
            return ((gap, gap, 1), (x_pos, 0.5, y_pos))
        if obj == 'Cardbox2':
            gap_box = gap - 0.2
            return ((gap_box, gap_box, gap_box), (x_pos, 1.5, y_pos + 0.5))
        if obj == 'Cardbox1':
            gap_box = gap - 0.2
            return ((gap_box, gap_box, gap_box), (x_pos, 1.5, y_pos + 0.5))
        if obj == 'Ramp':
            return ((gap, 2, 3), (x_pos, 0.5, y_pos))

    return ((x_size, z_size, y_size), (x_pos, z_pos, y_pos))