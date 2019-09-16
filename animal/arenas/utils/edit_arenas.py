  
import random
import numpy as np
from .sample_features import random_size, random_pos


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
        s += "      positions: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(pos[0], pos[1], pos[2])
    if size is not None:
        s += "      sizes: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(size[0], size[1], size[2])
    if RGB is not None:
        s += "      colors: \n      - !RGB {{r: {}, g: {}, b: {}}}\n".format(RGB[0], RGB[1], RGB[2])
    if rot is not None:
        s += "      rotations: [{}]\n".format(rot)
    return s


def write_arena(fname, time, arena_str, blackouts=None):
    blackouts_str = "blackouts: {} \n    ".format(blackouts) if blackouts else ""
    with open("{}.yaml".format(fname), 'w+') as f:
        f.write("!ArenaConfig \narenas: \n  0: !Arena \n    t: {} \n    {}items:\n".format(time, blackouts_str))
        f.write(arena_str)


def add_ramp_scenario(arena):

    # create a wall as a platform
    category = 'Wall'
    size_wall = (np.clip(random_size(category)[0],2,7),
                 np.clip(random_size(category)[1],2,7),
                 np.clip(random_size(category)[2],2,7))
    position_wall = random_pos(with_respect_to_center='close')
    rotation_wall = random.choice([0, 90, 180, 360])
    arena = add_object(arena, category, size=size_wall,
                       pos=position_wall, RGB=(0, 0, 255), rot=rotation_wall)

    # locate a reward on the wall
    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    size_object = random_size(category)
    arena = add_object(
        arena, category, size=size_object,  RGB=(0, 0, 255),
        pos=(position_wall[0], position_wall[1] + size_object[1] + 5, position_wall[2]))

    # create ramps to access the reward
    category = 'Ramp'
    size_object = (size_wall[0], size_wall[1], size_wall[2])

    for rot, pos_shift in zip([0, 90, 180, 270], [(0.0, 1.1), (1.1, 0.0), (0.0, -1.1), (-1.1, 0.0)]):

            if random.random() > 0.5:
                position_object = (
                    position_wall[0] + size_wall[0] * pos_shift[0],  0.0,
                    position_wall[2] + size_wall[2] * pos_shift[1])

                arena = add_object(arena, category, size=size_object, rot=rot, pos=position_object)

    return arena

