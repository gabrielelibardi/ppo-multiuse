import os
import glob
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from baselines.bench import load_results


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Visualizer():
    """
    A class to visualize information gathered by logger.
    """

    def __init__(self, logs_dir, log_interval=10):
        """
        Initialize a Visualizer.
        """

        if not os.path.isdir("{}/logs_test".format(logs_dir)):
            os.mkdir("{}/logs_test".format(logs_dir))
        for file in glob.glob(logs_dir + "/test*"):
            shutil.copy(file, "{}/logs_test/{}".format(
                logs_dir, file.split("/")[-1]))


        target_dir = "{}/tensorboard_files".format(logs_dir)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)

        df = load_results("{}/logs_test/".format(logs_dir))
        drop = []
        arenas = []
        update = 0
        num_test_files = len(df['arena'].unique())
        arena_update = np.zeros(df.shape[0])
        for index, row in df.iterrows():
            arena_update[index] = update
            if row['arena'] not in arenas:
                arenas.append(row['arena'])
            else:
                drop.append(index)
            if len(arenas) == num_test_files:
                arenas = []
                update += 1

        df['update'] = arena_update
        df = df.drop(drop)
        df['l'] -= 1
        df['ereward'] -= 1/df['max_time']
        df['ereward'][df['ereward'] < 0.0] = 0.0
        df['perf'] = df['ereward'] / df['max_reward']
        df['perf'].where(df['perf'] > 0, 0, inplace=True)
        df['goal'] = df['perf'] > 0.9

        test_avg = df.groupby('update').mean()

        test_writers = SummaryWriter("{}/test".format(target_dir), flush_secs=5)
        for index, row in test_avg.iterrows():
            test_writers.add_scalar(
                "test/performance", row['perf'], global_step=log_interval * index)
            test_writers.add_scalar(
                "test/goal", row['goal'], global_step=log_interval * index)

        arena_types = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
        arena_type_writers = [SummaryWriter("{}/{}".format(target_dir, type), flush_secs=5)
                              for type in arena_types]
        arena_type_dfs = [df[df['arena_type'] == type] for type in arena_types]
        for num_type, arena_type_df in enumerate(arena_type_dfs):
            arena_type_avg = arena_type_df.groupby('update').mean()
            for index, row in arena_type_avg.iterrows():
                arena_type_writers[num_type].add_scalar(
                    "test/performance_by_type", row['perf'],
                    global_step=log_interval * index)
                arena_type_writers[num_type].add_scalar(
                    "test/goal_by_type", row['goal'], global_step=log_interval * index)

        arena_type_writers = [SummaryWriter("{}/{}".format(target_dir, type), flush_secs=5)
                              for type in df['arena'].unique()]
        arena_type_dfs = [df[df['arena'] == type] for type in df['arena'].unique()]
        for num_type, arena_type_df in enumerate(arena_type_dfs):
            arena_type_avg = arena_type_df.groupby('update').mean()
            for index, row in arena_type_avg.iterrows():
                arena_type_writers[num_type].add_scalar(
                    "test/performance_by_arena", row['perf'],
                    global_step=log_interval * index)
                arena_type_writers[num_type].add_scalar(
                    "test/goal_by_arena", row['goal'], global_step=log_interval * index)


if __name__ == "__main__":

    import os
    import sys
    import time
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--logs-dir", help="experiment logs path")
    parser.add_argument("-i", "--logs-interval", default=10, help="log interval during experiments")
    parser.add_argument("-n", "--num-steps", default=1000, help="train num steps before update")
    parser.add_argument(
        "-p", "--port", help="port to connect tensorboard to", type=int)

    arguments = parser.parse_args()
    if arguments.logs_dir:
        if not os.path.isdir(arguments.logs_dir):
            print(colorize(
                "error: experiment directory does not exist.", "red"))
            sys.exit()
    else:
        print(colorize("error: experiment path missing.", "red"))

    if arguments.port:
        port_num = arguments.port
    else:
        port_num = 8888

    visualizer = Visualizer(
        logs_dir=arguments.logs_dir,
        log_interval=int(arguments.logs_interval))
    time.sleep(5)


    args = [
        "tensorboard",
        "--logdir={}".format(arguments.logs_dir + "/tensorboard_files"),
        "--port={}".format(port_num),
    ]

    subprocess.check_call(args)
    sys.exit()
