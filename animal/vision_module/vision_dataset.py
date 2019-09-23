import math
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import multivariate_normal


class DatasetVision(Dataset):
    """
    Creates a dataset to train a VAE.
    """

    def __init__(
            self,
            data_filename,
    ):

        data = np.load(data_filename)

        self.observations = data['observations']
        self.positions = data['positions']
        self.rotations = data['rotations']
        self.frames_per_episode = data['frames_per_episode']

        self.num_samples = self.observations.shape[0]

    def __len__(self):
        return self.num_samples

    def get_frames_per_episode(self):
        return self.frames_per_episode

    def __getitem__(self, idx):
        obs = self.observations[idx, :, :, :]
        pos = self.positions[idx, 0:3:2]
        rot = self.rotations[idx, :]

        return (torch.FloatTensor(obs), torch.FloatTensor(pos),
                torch.FloatTensor(rot))


class DatasetVisionPaper(Dataset):
    """
    Creates a dataset to train a VAE.
    """

    def __init__(
            self,
            data_filename,
            N=32,
            M=32,
    ):

        data = np.load(data_filename)
        self.observations = data['observations']
        self.positions = data['positions']
        self.rotations = data['rotations']
        self.frames_per_episode = data['frames_per_episode']
        self.num_samples = self.observations.shape[0]

        self.N = N
        self.pos_means = np.random.uniform(0., 40., (self.N, 2))
        self.pos_var = np.eye(2)
        self.M = M
        self.rot_means = np.random.uniform(-np.pi, np.pi, self.M)
        self.concentration = 1


    def __len__(self):
        return self.num_samples

    def get_frames_per_episode(self):
        return self.frames_per_episode

    def __getitem__(self, idx):
        obs = self.observations[idx, :, :, :]
        pos = self.positions[idx, 0:3:2]
        rot = self.rotations[idx, :]

        return (torch.FloatTensor(obs),
                torch.FloatTensor(self.place_cell_dist(pos)),
                torch.FloatTensor(self.direction_cell_dist(rot)))

    def place_cell_dist(self, position):
        out = np.zeros(self.N)
        total = 0
        for i in range(self.N):
            out[i] = multivariate_normal.pdf(
                position, self.pos_means[i], self.pos_var)
            total += out[i]
        out /= total
        return out

    def direction_cell_dist(self, rotation):
        out = np.zeros(self.M)
        total = 0
        for i in range(self.M):
            out[i] = math.exp(
                self.concentration * (
                    math.cos(np.radians(rotation) - self.rot_means[i])))
            total += out[i]
        out /= total
        return out


class DatasetVisionRecurrent(Dataset):
    """
    Creates a dataset to train a VAE.
    """

    def __init__(
            self,
            data_filename,
    ):

        data = np.load(data_filename)

        self.observations = data['observations']
        self.positions = data['positions']
        self.rotations = data['rotations']
        self.frames_per_episode = data['frames_per_episode']
        self.num_samples = (self.observations.shape[0] // self.frames_per_episode) - 1


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = self.observations[
              self.frames_per_episode * idx:self.frames_per_episode * idx +
              self.frames_per_episode, :, :, :]
        pos = self.positions[
              self.frames_per_episode * idx:self.frames_per_episode * idx +
              self.frames_per_episode, 0:3:2]
        rot = self.rotations[
              self.frames_per_episode * idx:self.frames_per_episode * idx +
              self.frames_per_episode, :]
        return (torch.FloatTensor(obs),
                torch.FloatTensor(pos), torch.FloatTensor(rot))
