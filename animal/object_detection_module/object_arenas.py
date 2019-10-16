import random
from animal.arenas.utils.edit_arenas import add_object, write_arena

objects = [
    'GoodGoal',
    'BadGoal',
    'GoodGoalMulti',
    'Wall',
    'Ramp',
    'CylinderTunnel',
    'WallTransparent',
    'CylinderTunnelTransparent',
    'Cardbox1',
    'Cardbox2',
    'UObject',
    'LObject',
    'LObject2',
    'DeathZone',
    'HotZone'
]

labels = {
    'GoodGoal': 0,
    'BadGoal': 1,
    'GoodGoalMulti': 2,
    'Wall': 3,
    'Ramp': 4,
    'CylinderTunnel': 5,
    'WallTransparent': 6,
    'CylinderTunnelTransparent': 7,
    'Cardbox1': 8,
    'Cardbox2': 9,
    'UObject': 10,
    'LObject': 11,
    'LObject2': 12,
    'DeathZone': 13,
    'HotZone': 14,
}


def create_object_arena(target_path, arena_name, num_objects=7, time=250):
    """ Empty arena with ´num_objects´ objects of the same type"""

    arena = ''
    object = random.choice(objects)

    for _ in range(num_objects):
        arena = add_object(arena, object)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena, blackouts=None)

    return object, labels[object]