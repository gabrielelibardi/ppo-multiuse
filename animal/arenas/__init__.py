from .utils import (
    create_c1_arena,
    create_c2_arena,
    create_c3_arena,
    create_c4_arena,
)

cl_tasks = {
    "generators": {
        1: create_c1_arena,
        2: create_c1_arena,
        3: create_c1_arena,
        4: create_c2_arena,
        5: create_c2_arena,
        6: create_c2_arena,
        7: create_c2_arena,
    },
    "parameters": {
        1: {"max_reward": 5, "time": 250, "max_num_good_goals": 1},
        2: {"max_reward": 10, "time": 250, "max_num_good_goals": 1},
        3: {"max_reward": 15, "time": 250, "max_num_good_goals": 1},
        4: {"max_reward": 5, "time": 250, "max_num_good_goals": 2},
        5: {"max_reward": 10, "time": 250, "max_num_good_goals": 2},
        6: {"max_reward": 15, "time": 250, "max_num_good_goals": 2},
        7: {"max_reward": 20, "time": 250, "max_num_good_goals": 3},
        8: {"max_reward": 20, "time": 250, "max_num_good_goals": 3},
        9: {"max_reward": 25, "time": 250, "max_num_good_goals": 4},
        10: {"max_reward": 30, "time": 250, "max_num_good_goals": 5},
    }
}
