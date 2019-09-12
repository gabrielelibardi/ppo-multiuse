

def add_object(s, object_name, pos=None, size=None, RGB=None):
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
    return s


def write_arena(fname, time, arena_str):
    with open("{}.yaml".format(fname), 'w+') as f:
        f.write("!ArenaConfig \narenas: \n  0: !Arena \n    t: {} \n    items:\n".format(time))
        f.write(arena_str)
