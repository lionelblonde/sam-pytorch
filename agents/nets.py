import math

import torch
import torch.nn as nn
import torch.nn.modules.rnn as rnn
import torch.nn.functional as F
import torch.nn.utils as U


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    """Applies orthogonal initialization for the parameters of a given module"""

    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, rnn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.

class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        # Define feature extractor
        self.fc_1 = nn.Linear(ob_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        ortho_init(self.fc_1, nonlinearity='relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='relu', constant_bias=0.0)
        # Define action head
        self.ac_head = nn.Linear(64, ac_dim)
        ortho_init(self.ac_head, weight_scale=0.01, constant_bias=0.0)
        # Add layer norm
        self.ln_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        # Determine which parameters are perturbable
        self.perturbable_params = [p for p in self.state_dict() if 'ln' not in p]
        self.nonperturbable_params = [p for p in self.state_dict() if 'ln' in p]
        assert (set(self.perturbable_params + self.nonperturbable_params) ==
                set(self.state_dict().keys()))
        # Following the paper 'Parameter Space Noise for Exploration', we do not
        # perturb the conv2d layers, only the fully-connected part of the network.
        # Additionally, the extra variables introduced by layer normalization should remain
        # unperturbed as they do not play any role in exploration.

    def forward(self, ob):
        plop = ob
        # Stack fully-connected layers
        plop = F.relu(self.ln_1(self.fc_1(plop)))
        plop = F.relu(self.ln_2(self.fc_2(plop)))
        ac = float(self.ac_max) * torch.tanh(self.ac_head(plop))
        return ac


class Critic(nn.Module):

    def __init__(self, env, hps):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # Create fully-connected layers
        self.fc_1 = nn.Linear(ob_dim + ac_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        ortho_init(self.fc_1, nonlinearity='relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='relu', constant_bias=0.0)
        # Define Q head
        self.q_head = nn.Linear(64, 1)
        ortho_init(self.q_head, weight_scale=0.01, constant_bias=0.0)
        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x

    def forward(self, ob, ac):
        # Concatenate observations and actions
        plop = torch.cat([ob, ac], dim=-1)
        # Stack fully-connected layers
        plop = F.relu(self.ln_1(self.fc_1(plop)))
        plop = F.relu(self.ln_2(self.fc_2(plop)))
        q = self.q_head(plop)
        return q


class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps

        in_dim = self.ob_dim if self.hps.state_only else self.ob_dim + self.ac_dim

        # Define hidden layers
        self.fc_1 = U.spectral_norm(nn.Linear(in_dim, 64))
        self.fc_2 = U.spectral_norm(nn.Linear(64, 64))
        ortho_init(self.fc_1, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='leaky_relu', constant_bias=0.0)

        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x

        # Define score head
        self.score_head = nn.Linear(64, 1)
        ortho_init(self.score_head, nonlinearity='linear', constant_bias=0.0)

    def get_reward(self, ob, ac):
        """Craft surrogate reward"""
        ob = torch.FloatTensor(ob).cpu()
        ac = torch.FloatTensor(ac).cpu()

        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(self.forward(ob, ac).detach()) + 1e-8)

        if self.hps.minimax_only:
            return minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = torch.log(torch.sigmoid(self.forward(ob, ac).detach()))
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            return non_satur_reward + minimax_reward

    def forward(self, ob, ac):
        if self.hps.state_only:
            plop = ob
        else:
            plop = torch.cat([ob, ac], dim=-1)
        # Add hidden layers
        plop = F.leaky_relu(self.ln_1(self.fc_1(plop)))
        plop = F.leaky_relu(self.ln_2(self.fc_2(plop)))
        # Add output layer
        score = self.score_head(plop)
        return score
