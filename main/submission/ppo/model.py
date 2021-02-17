import numpy as np
import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init
from torch.nn import ConstantPad2d

pad = ConstantPad2d(22, 0.0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean() 

        return value, action, action_log_probs, rnn_hxs, dist_entropy
        
    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # inputs.requires_grad_(False)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist



class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class FixupCNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, image_size=84):
        super(FixupCNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.main = FixupCNN(image_size,num_inputs,hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs



class FixupCNN(nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.
    """

    def __init__(self, image_size, depth_in, hidden_size):
        super().__init__()
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidual(depth_out, 8),
                FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            FixupResidual(depth_in, 8),
            FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, hidden_size)  ##use init_???

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class FixupResidual(nn.Module):
    def __init__(self, depth, num_residual):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()
        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x


class ImpalaCNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, image_size=84):
        super(ImpalaCNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.main = ImpalaCNN(image_size,num_inputs,hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.

    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self, image_size, depth_in, hidden_size):
        super().__init__()
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, hidden_size))

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(depth, depth, 3, padding=1))
        self.conv2 = init_(nn.Conv2d(depth, depth, 3, padding=1))

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class VAE(NNBase):
    """ Variational Autoencoder """
    def __init__(self, img_channels=3, latent_size=100, num_layers=5, img_size=128,recurrent=False):
        hidden_size = latent_size
        super(VAE, self).__init__(recurrent, hidden_size, hidden_size)

        self.num_layers = num_layers
        self.img_size = img_size

        self.encoder = EncoderVAE(img_channels, latent_size,
                               num_layers=self.num_layers,
                               img_size=self.img_size)
        self.decoder = DecoderVAE(img_channels, latent_size,
                               num_layers=self.num_layers,
                               img_size=self.img_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, x, rnn_hxs, masks):
        x= pad(x)
        mu, logsigma = self.encoder.forward(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(z, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

    def reconstruction_loss(self, x):
        x = pad(x)
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

    def save(self, filename, net_parameters):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, use_device=torch.device('cpu')):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict(
                torch.load(
                    path + "/state.torch",
                    map_location=use_device,
                )
            )
        return net, net_parameters


class EncoderVAE(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, num_layers=4, img_size=84, kernel_size=4):
        super(EncoderVAE, self).__init__()

        self.num_layers = num_layers
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.img_channels = img_channels
        out_size = self.img_size // (2 ** self.num_layers)
        out_channels = 32 * (2 ** (self.num_layers - 1))

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)

        # model encoder
        self.blocks_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_1_{}".format(k), nn.Conv2d(
                            in_channels=32 * (2 ** (k - 2)),
                            out_channels=32 * (2 ** (k - 1)),
                            kernel_size=self.kernel_size,
                            stride=2,
                            padding=(self.kernel_size // 2) - 1,
                            bias=False)),
                        ("batchnorm_1_{}".format(k),
                         nn.BatchNorm2d(32 * (2 ** (k - 1)))),
                        ("relu_1_{}".format(k), nn.LeakyReLU()),
                        ("conv_2_{}".format(k), nn.Conv2d(
                            in_channels=32 * (2 ** (k - 1)),
                            out_channels=32 * (2 ** (k - 1)),
                            kernel_size=self.kernel_size - 1,
                            stride=1,
                            padding=(self.kernel_size - 1) // 2,
                            bias=False)),
                        ("batchnorm_2_{}".format(k),
                         nn.BatchNorm2d(32 * (2 ** (k - 1)))),
                        ("relu_2_{}".format(k), nn.LeakyReLU()),
                    ])
                ) for k in range(2, self.num_layers +1 )
            ]
        )

        self.fc_mu = nn.Linear(out_size ** 2 * out_channels, latent_size)
        self.fc_logsigma = nn.Linear(out_size ** 2 * out_channels, latent_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        for block in self.blocks_encoder:
            x = block(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class DecoderVAE(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, num_layers=2, img_size=84, kernel_size=4):
        super(DecoderVAE, self).__init__()

        self.latent_size = latent_size
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.out_size = self.img_size // (2 ** self.num_layers)
        self.out_channels = 32 * (2 ** (self.num_layers - 1))

        self.fc1 = nn.Linear(latent_size, self.out_size ** 2 * self.out_channels)

        # model decoder
        self.blocks_decoder = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("deconv_1_{}".format(k), nn.ConvTranspose2d(
                            in_channels=32 * (2 ** (k - 1)),
                            out_channels=32 * (2 ** (k - 2)),
                            kernel_size=self.kernel_size,
                            stride=2,
                            padding=(self.kernel_size // 2) - 1,
                            bias=False)),
                        ("batchnorm_1_{}".format(k),
                         nn.BatchNorm2d(32 * (2 ** (k - 2)))),
                        ("relu_1_{}".format(k), nn.ReLU()),
                        ("conv_2_{}".format(k), nn.Conv2d(
                            in_channels=32 * (2 ** (k - 2)),
                            out_channels=32 * (2 ** (k - 2)),
                            kernel_size=self.kernel_size - 1,
                            stride=1,
                            padding=(self.kernel_size - 1) // 2,
                            bias=False)),
                        ("batchnorm_2_{}".format(k),
                         nn.BatchNorm2d(32 * (2 ** (k - 2)))),
                        ("relu_2_{}".format(k), nn.LeakyReLU()),
                    ])
                ) for k in range(2, self.num_layers + 1)[::-1]
            ]
        )

        self.deconv_final = nn.ConvTranspose2d(32, self.img_channels, self.kernel_size, stride=2, padding=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), self.out_channels, self.out_size, self.out_size)

        for block in self.blocks_decoder:
            x = block(x)

        x = self.deconv_final(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction

