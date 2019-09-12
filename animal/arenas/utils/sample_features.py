
import math
import random


ranges = {
    "close": [0, 7.5],
    "medium": [7.5, 15.0],
    "far": [15.0, 60.0]
}


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def random_pos():
    return (random.randint(1, 400) / 10., 0. / 10., random.randint(1, 400) / 10.)


def random_size_reward():
    #according to docs it's 0.5-5
    s = random.randint(5, 50)/10
    return (s,s,s)


def sample_position_with_respect_to(reference_position, range="far"):
    """
    Sample random position within the specified range from reference_position.
        - close: dist in range [0, 7.5)
        - medium: dist in range [7.5, 15)
        - far: dist in range [15, +)
    """

    position = random_pos()

    lower_bound = ranges[range][0]
    upper_bound = ranges[range][1]

    while not lower_bound <= distance(reference_position, position) < upper_bound:
        position = random_pos()

    return position
