import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        # added
        if x.size(0) == 1:
            return x.squeeze()
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, trained_encoder=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            elif len(obs_shape) == 2: # obs_shape = 2 in the sprites env
                base = CNNBase
            else:
                raise NotImplementedError

        # self.base = base(obs_shape[0], **base_kwargs)
        # encoder
        self.base = base(obs_shape[0], trained_encoder, hidden_size=64, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            # self.dist = DiagGaussian(self.base.output_size, num_outputs)
            self.dist = DiagGaussian(32, num_outputs)
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

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


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


def random_crop(imgs, size=64):
    n, c, h, w = imgs.shape
    w1 = torch.randint(0, w - size + 1, (n,))
    h1 = torch.randint(0, h - size + 1, (n,))
    cropped = torch.empty((n, c, size, size),
                            dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i][:] = img[:, h11:h11 + size, w11:w11 + size]
    return cropped

def random_cutout(imgs, min_cut=4, max_cut=24):
    n, c, h, w = imgs.shape
    w_cut = torch.randint(min_cut, max_cut + 1, (n,))  # random size cut
    h_cut = torch.randint(min_cut, max_cut + 1, (n,))  # rectangular shape
    fills = torch.randint(0, 255, (n, c, 1, 1))  # assume uint8.
    for img, wc, hc, fill in zip(imgs, w_cut, h_cut, fills):
        w1 = torch.randint(w - wc + 1, ())  # uniform over interior
        h1 = torch.randint(h - hc + 1, ())
        img[:, h1:h1 + hc, w1:w1 + wc] = fill
    return imgs

def random_flip(imgs, p=0.5):
    n, _, _, _ = imgs.shape
    flip_mask = torch.rand(n, device=imgs.device) < p
    imgs[flip_mask] = imgs[flip_mask].flip([3])  # left-right
    return imgs

class Encoder(nn.Module):
    def __init__(self, image_resolution=64):
        super(Encoder, self).__init__()

        # assume image_resolution=64
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)] # input: 64*64*1
        )

        self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1))  # 32*32*4
        self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1))  # 16*16*8
        self.convs.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)) # 8*8*16
        self.convs.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)) # 4*4*32
        self.convs.append(nn.Conv2d(64, 128, kernel_size=2, stride=2)) # 2*2*64

        # the final 1x1 feature vector gets mapped to a 64-dimensional observation space
        self.fc = nn.Linear(in_features=128, out_features=64)  # input: 1*1*128 output: 64

    # x is the observation at one time step
    def forward(self, x, detach=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(6):
            x = torch.relu(self.convs[i](x))
        out = self.fc(x.squeeze())

        # freeze the encoder
        if detach:
            out.detach()
        return out

class CNNBase(NNBase):
    def __init__(self, num_inputs, trained_encoder, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # The default implementation
        # self.main = nn.Sequential(
        #     # init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        # fixing default impl
        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(512, hidden_size)), nn.ReLU())

        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(1, 32, kernel_size=3, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=3, stride=1)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=3, stride=1)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(32 * 27 * 27, hidden_size)), nn.ReLU())

        # encoder
        encoder = Encoder()
        if trained_encoder:           
            encoder = trained_encoder
            
        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)), nn.ReLU(),
        #     init_(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)), nn.ReLU(),
        #     init_(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)), nn.ReLU(),
        #     init_(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)), nn.ReLU(), 
        #     init_(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 128, kernel_size=2, stride=2)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(128, 64)),
        #     init_(nn.Linear(64, 32)), nn.ReLU())
        
        self.main = nn.Sequential(encoder, init_(nn.Linear(64, 32)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        # self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.critic_linear = nn.Sequential(init_(nn.Linear(32, 1)), nn.ReLU())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # inputs = inputs.unsqueeze(1) # a quick hack: modify the size
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(inputs.shape) == 2:
            inputs = inputs[None, None, :]
        # inputs = random_crop(inputs)
        # x = self.main(inputs / 255.0) # inputs should have already /255 (?)
        x = self.main(inputs)
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

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
