import numpy as np
import torch
from torch.utils.data import Dataset

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
        pos = self.positions[idx, ::1][0:3:2]
        rot = self.rotations[idx, :]
        mask = torch.zeros(1, 1) if (
            idx % self.frames_per_episode == 0 and idx != 0) else torch.ones(
            1, 1)

        return (torch.FloatTensor(obs), torch.FloatTensor(pos),
                torch.FloatTensor(rot), mask)
