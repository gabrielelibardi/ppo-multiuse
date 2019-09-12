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

    def __init__(self, logs_dir, num_steps=1000, log_interval=10):
        """
        Initialize a Visualizer.
        """

        if not os.path.isdir("{}/logs_test".format(logs_dir)):
            os.mkdir("{}/logs_test".format(logs_dir))
        for file in glob.glob(logs_dir + "/test*"):
            shutil.copy(file, "{}/logs_test/{}".format(
                logs_dir, file.split("/")[-1]))

        if not os.path.isdir("{}/logs_train".format(logs_dir)):
            os.mkdir("{}/logs_train".format(logs_dir))
        for file in glob.glob(logs_dir + "/train*"):
            shutil.copy(file, "{}/logs_train/{}".format(
                logs_dir, file.split("/")[-1]))

        target_dir = "{}/tensorboard_files".format(logs_dir)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)

        df = load_results("{}/logs_train/".format(logs_dir))
        df['l'] -= 1
        df['r'] -= 1/df['max_time']
        df['update'] = df["l"].cumsum() // (num_steps * df[df['index'] == 0].shape[0])
        df['discounted_max_reward'] = df['max_reward'] - (df['l'] / df['max_time'])
        df['perf'] = df['r'] / df['discounted_max_reward']
        df['perf'].where(df['perf'] > 0, 0, inplace=True)
        df['goal'] = df['perf'] > 0.9
        train_avg = df.groupby('update').mean()

        train_writer = SummaryWriter("{}/train".format(target_dir), flush_secs=5)
        for index, row in train_avg.iterrows():
            train_writer.add_scalar(
                "train/performance", row['perf'], global_step=index)
            train_writer.add_scalar(
                "train/goal", row['goal'], global_step=index)
            train_writer.add_scalar(
                "train/cl_stage", row['cl_stage'], global_step=index)

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
        df['perf'] = df['r'] / df['max_reward']
        df['perf'].where(df['perf'] > 0, 0, inplace=True)
        df['goal'] = df['perf'] > 0.9

        test_avg = df.groupby('update').mean()

        test_writers = SummaryWriter("{}/test".format(target_dir), flush_secs=5)
        for index, row in test_avg.iterrows():
            test_writers.add_scalar(
                "test/performance", row['perf'], global_step=log_interval * index)
            test_writers.add_scalar(
                "test/goal", row['goal'], global_step=log_interval * index)


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
        num_steps=arguments.num_steps,
        log_interval=int(arguments.logs_interval))
    time.sleep(5)


    args = [
        "tensorboard",
        "--logdir={}".format(arguments.logs_dir + "/tensorboard_files"),
        "--port={}".format(port_num),
    ]

    subprocess.check_call(args)
    sys.exit()
