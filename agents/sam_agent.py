from collections import namedtuple

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import logger
from helpers.console_util import log_module_info
from helpers.dataset import Dataset
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import Actor, Critic, Discriminator
from agents.param_noise import AdaptiveParamNoise
from agents.ac_noise import NormalAcNoise, OUAcNoise


class SAM(object):

    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        self.ob_dim = self.ob_shape[-1]  # num dims
        self.ac_dim = self.ac_shape[-1]  # num dims

        self.device = device
        self.hps = hps
        assert self.hps.lookahead > 1 or not self.hps.n_step_returns

        # Define action clipping range
        assert all(self.ac_space.low == -self.ac_space.high)
        self.max_ac = self.ac_space.high[0].astype('float32')
        assert all(ac_comp == self.max_ac for ac_comp in self.ac_space.high)

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Create online and target nets, and initilize the target nets
        self.actor = Actor(self.env, self.hps).to(self.device)
        sync_with_root(self.actor)
        self.targ_actor = Actor(self.env, self.hps).to(self.device)
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.env, self.hps).to(self.device)
        sync_with_root(self.critic)
        self.targ_critic = Critic(self.env, self.hps).to(self.device)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        if self.hps.enable_clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin_critic = Critic(self.env, self.hps).to(self.device)
            sync_with_root(self.twin_critic)
            self.targ_twin_critic = Critic(self.env, self.hps).to(self.device)
            self.targ_twin_critic.load_state_dict(self.targ_twin_critic.state_dict())

        if self.param_noise is not None:
            # Create parameter-noise-perturbed ('pnp') actor
            self.pnp_actor = Actor(self.env, self.hps).to(self.device)
            self.pnp_actor.load_state_dict(self.actor.state_dict())
            # Create adaptive-parameter-noise-perturbed ('apnp') actor
            self.apnp_actor = Actor(self.env, self.hps).to(self.device)
            self.apnp_actor.load_state_dict(self.actor.state_dict())

        # Set up replay buffer
        self.setup_replay_buffer()

        # Set up the optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.hps.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.hps.critic_lr,
                                                 weight_decay=self.hps.wd_scale)
        if self.hps.enable_clipped_double:
            self.twin_critic_optimizer = torch.optim.Adam(self.twin_critic.parameters(),
                                                          lr=self.hps.critic_lr,
                                                          weight_decay=self.hps.wd_scale)

        # Set up demonstrations dataset
        self.e_dataloader = DataLoader(expert_dataset, self.hps.batch_size, shuffle=True)
        # Create discriminator
        self.discriminator = Discriminator(self.env, self.hps).to(self.device)
        sync_with_root(self.discriminator)
        # Create optimizer
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hps.d_lr)

        log_module_info(logger, 'actor', self.actor)
        log_module_info(logger, 'critic', self.critic)

        if not self.hps.pixels:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        ac_noise = None
        param_noise = None
        logger.info("parsing noise type")
        # Parse the comma-seprated (with possible whitespaces) list of noise params
        for cur_noise_type in noise_type.split(','):
            cur_noise_type = cur_noise_type.strip()  # remove all whitespaces (start and end)
            # If the specified noise type is literally 'none'
            if cur_noise_type == 'none':
                pass
            # If 'adaptive-param' is in the specified string for noise type
            elif 'adaptive-param' in cur_noise_type:
                # Set parameter noise
                _, std = cur_noise_type.split('_')
                std = float(std)
                param_noise = AdaptiveParamNoise(initial_std=std, delta=std)
                logger.info("  {} configured".format(param_noise))
            elif 'normal' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Spherical (isotropic) gaussian action noise
                ac_noise = NormalAcNoise(mu=np.zeros(self.ac_dim),
                                         sigma=float(std) * np.ones(self.ac_dim))
                logger.info("  {} configured".format(ac_noise))
            elif 'ou' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                ac_noise = OUAcNoise(mu=np.zeros(self.ac_dim),
                                     sigma=(float(std) * np.ones(self.ac_dim)))
                logger.info("  {} configured".format(ac_noise))
            else:
                raise RuntimeError("unknown specified noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info("setting up replay buffer")
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                self.replay_buffer = UnrealReplayBuffer(self.hps.mem_size,
                                                        self.ob_shape,
                                                        self.ac_shape)
            else:  # Vanilla prioritized experience replay
                self.replay_buffer = PrioritizedReplayBuffer(self.hps.mem_size,
                                                             self.ob_shape,
                                                             self.ac_shape,
                                                             alpha=self.hps.alpha,
                                                             beta=self.hps.beta,
                                                             ranked=self.hps.ranked)
        else:  # Vanilla experience replay
            self.replay_buffer = ReplayBuffer(self.hps.mem_size,
                                              self.ob_shape,
                                              self.ac_shape)
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("  {} configured".format(self.replay_buffer))

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated Q value.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.FloatTensor(ob[None]).to(self.device)

        if not self.hps.pixels:
            ob = ((ob - torch.FloatTensor(self.rms_obs.mean)) /
                  (torch.sqrt(torch.FloatTensor(self.rms_obs.var)) + 1e-8))
            ob = torch.clamp(ob, -5.0, 5.0)

        if apply_noise and self.param_noise is not None:
            # Predict following a parameter-noise-perturbed actor
            ac = self.pnp_actor(ob)
        else:
            # Predict following the non-perturbed actor
            ac = self.actor(ob)

        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()

        if apply_noise and self.ac_noise is not None:
            # Apply additive action noise once the action has been predicted,
            # in combination with parameter noise, or not.
            noise = self.ac_noise.generate()
            assert noise.shape == ac.shape
            ac += noise

        ac = ac.clip(-self.max_ac, self.max_ac)

        return ac

    def store_transition(self, ob0, ac, rew, ob1, done1):
        """Store a experiental transition in the replay buffer"""
        # Scale the reward
        rew *= self.hps.reward_scale

        # Store the transition in the replay buffer
        self.replay_buffer.append(ob0, ac, rew, ob1, done1)

    def train(self, update_critic, update_actor):
        """Train the agent"""
        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(self.hps.batch_size,
                                                        n=self.hps.lookahead,
                                                        gamma=self.hps.gamma)
        else:
            batch = self.replay_buffer.sample(self.hps.batch_size)

        if not self.hps.pixels:
            batch['obs0'] = ((batch['obs0'] - self.rms_obs.mean) /
                             (np.sqrt(self.rms_obs.var) + 1e-8))
            batch['obs0'] = np.clip(batch['obs0'], -5.0, 5.0)

        # Create tensors from the inputs
        state = torch.FloatTensor(batch['obs0']).to(self.device)
        action = torch.FloatTensor(batch['acs']).to(self.device)
        next_state = torch.FloatTensor(batch['obs1']).to(self.device)
        reward = torch.FloatTensor(batch['rews']).to(self.device)
        done = torch.FloatTensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.FloatTensor(batch['iws']).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.FloatTensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.enable_targ_actor_smoothing:
            n_ = action.clone().detach().data.normal_(0, self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
            next_action = (self.targ_actor(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actor(next_state)

        # Create data loaders

        dataset = Dataset(batch)
        dataloader = DataLoader(dataset, self.hps.batch_size, shuffle=True)
        # Iterable over 1 element, but cleaner that way

        # Collect recent pairs uniformly from the experience replay buffer
        window = 128  # HAXX
        assert window >= self.hps.batch_size, "must have window >= batch_size"
        recent_batch = self.replay_buffer.sample_recent(self.hps.batch_size, window)
        recent_dataset = Dataset(recent_batch)
        recent_dataloader = DataLoader(recent_dataset, self.hps.batch_size, shuffle=True)
        # Iterable over 1 element, but cleaner that way

        # Compute losses

        # Compute Q estimate
        q = self.critic(state, action)
        if self.hps.enable_clipped_double:
            twin_q = self.twin_critic(state, action)

        # Compute target Q estimate
        q_prime = self.targ_critic(next_state, next_action)
        if self.hps.enable_clipped_double:
            # Define Q' as the minimum Q value between TD3's twin Q's
            twin_q_prime = self.targ_twin_critic(next_state, next_action)
            q_prime = torch.min(q_prime, twin_q_prime)
        targ_q = reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime.detach()

        # Critic loss
        huber_td_errors = F.smooth_l1_loss(q, targ_q, reduction='none')
        if self.hps.enable_clipped_double:
            twin_huber_td_errors = F.smooth_l1_loss(twin_q, targ_q, reduction='none')

        if self.hps.prioritized_replay:
            # Adjust with importance weights
            huber_td_errors *= iws
            if self.hps.enable_clipped_double:
                twin_huber_td_errors *= iws

        critic_loss = huber_td_errors.mean()
        if self.hps.enable_clipped_double:
            twin_critic_loss = twin_huber_td_errors.mean()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Actor grads
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gradnorm = U.clip_grad_norm_(self.actor.parameters(),
                                           self.hps.clip_norm)

        # Critic(s) grads
        self.critic_optimizer.zero_grad()
        if self.hps.enable_clipped_double:
            self.twin_critic_optimizer.zero_grad()

        critic_loss.backward()
        critic_gradnorm = U.clip_grad_norm_(self.critic.parameters(),
                                            self.hps.clip_norm)
        if self.hps.enable_clipped_double:
            twin_critic_loss.backward()
            twin_critic_gradnorm = U.clip_grad_norm_(self.twin_critic.parameters(),
                                                     self.hps.clip_norm)

        # Update critic(s)
        average_gradients(self.critic, self.device)
        self.critic_optimizer.step()
        if self.hps.enable_clipped_double:
            average_gradients(self.twin_critic, self.device)
            self.twin_critic_optimizer.step()

        if update_actor:
            # Update actor
            average_gradients(self.actor, self.device)
            self.actor_optimizer.step()

            # Update target nets
            self.update_target_net()

        if self.hps.prioritized_replay:
            # Update priorities
            td_errors = q - targ_q
            if self.hps.enable_clipped_double:
                td_errors = torch.min(q - targ_q, twin_q - targ_q)
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # epsilon from paper

            self.replay_buffer.update_priorities(batch['idxs'], new_priorities)

        for _ in range(self.hps.d_update_ratio):

            for chunk, e_chunk in zip(dataloader, self.e_dataloader):
                self.update_discriminator(chunk, e_chunk)

            for chunk, e_chunk in zip(recent_dataloader, self.e_dataloader):
                self.update_discriminator(chunk, e_chunk)

        # Aggregate the elements to return
        losses = {'actor': actor_loss.clone().cpu().data.numpy(),
                  'critic': critic_loss.clone().cpu().data.numpy()}
        gradnorms = {'actor': actor_gradnorm,
                     'critic': critic_gradnorm}
        if self.hps.enable_clipped_double:
            losses.update({'twin_critic': twin_critic_loss.clone().cpu().data.numpy()})
            gradnorms.update({'twin_critic': twin_critic_gradnorm})

        return losses, gradnorms

    def update_discriminator(self, chunk, e_chunk):
        # Create tensors from the inputs
        state = torch.FloatTensor(chunk['obs0']).to(self.device)
        action = torch.FloatTensor(chunk['acs']).to(self.device)
        # Get expert data and create tensors from the inputs
        e_obs, e_acs = e_chunk['obs0'], e_chunk['acs']
        e_state = torch.FloatTensor(e_obs).to(self.device)
        e_action = torch.FloatTensor(e_acs).to(self.device)
        # Compute scores
        p_scores = self.discriminator(state, action)
        e_scores = self.discriminator(e_state, e_action)
        # Create entropy loss
        scores = torch.cat([p_scores, e_scores], dim=0)
        entropy = F.binary_cross_entropy_with_logits(input=scores,
                                                     target=F.sigmoid(scores))
        entropy_loss = -self.hps.ent_reg_scale * entropy
        # Create labels
        fake_labels = torch.zeros_like(p_scores).to(self.device)
        real_labels = torch.ones_like(e_scores).to(self.device)
        # Label smoothing, suggested in 'Improved Techniques for Training GANs',
        # Salimans 2016, https://arxiv.org/abs/1606.03498
        # The paper advises on the use of one-sided label smoothing, i.e.
        # only smooth out the positive (real) targets side.
        # Extra comment explanation: https://github.com/openai/improved-gan/blob/
        # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
        # Additional material: https://github.com/soumith/ganhacks/issues/10
        real_labels.uniform_(0.7, 1.2)
        # Create binary classification (cross-entropy) losses
        p_loss = F.binary_cross_entropy_with_logits(input=p_scores, target=fake_labels)
        e_loss = F.binary_cross_entropy_with_logits(input=e_scores, target=real_labels)
        # Aggregated loss
        d_loss = p_loss + e_loss + entropy_loss

        # Update parameters
        self.d_optimizer.zero_grad()
        d_loss.backward()
        U.clip_grad_norm_(self.discriminator.parameters(), self.hps.clip_norm)
        average_gradients(self.discriminator, self.device)
        self.d_optimizer.step()

    def update_target_net(self):
        """Update the target networks by slowly tracking their non-target counterparts"""
        for param, targ_param in zip(self.actor.parameters(), self.targ_actor.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        for param, targ_param in zip(self.critic.parameters(), self.targ_critic.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        if self.hps.enable_clipped_double:
            for param, targ_param in zip(self.twin_critic.parameters(),
                                         self.targ_twin_critic.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)

    def adapt_param_noise(self):
        """Adapt the parameter noise standard deviation"""
        # Perturb separate copy of the policy to adjust the scale for the next 'real' perturbation

        batch = self.replay_buffer.sample(self.hps.batch_size)

        state = torch.FloatTensor(batch['obs0']).to(self.device)

        # Update the perturbable params
        for p in self.actor.perturbable_params:
            param = (self.actor.state_dict()[p]).clone()
            param_ = param.clone()
            noise = param_.data.normal_(0, self.param_noise.cur_std)
            self.apnp_actor.state_dict()[p].data.copy_((param + noise).data)
        # Update the non-perturbable params
        for p in self.actor.nonperturbable_params:
            param = self.actor.state_dict()[p].clone()
            self.apnp_actor.state_dict()[p].data.copy_(param.data)

        # Compute distance between actor and adaptive-parameter-noise-perturbed actor predictions
        self.pn_dist = torch.sqrt(F.mse_loss(self.actor(state), self.apnp_actor(state)))

        self.pn_dist = self.pn_dist.cpu().data.numpy()

        # Adapt the parameter noise
        self.param_noise.adapt_std(self.pn_dist)

    def reset_noise(self):
        """Reset noise processes at episode termination"""

        # Reset action noise
        if self.ac_noise is not None:
            self.ac_noise.reset()

        # Reset parameter-noise-perturbed actor vars by redefining the pnp actor
        # w.r.t. the actor (by applying additive gaussian noise with current std)
        if self.param_noise is not None:
            # Update the perturbable params
            for p in self.actor.perturbable_params:
                param = (self.actor.state_dict()[p]).clone()
                param_ = param.clone()
                noise = param_.data.normal_(0, self.param_noise.cur_std)
                self.pnp_actor.state_dict()[p].data.copy_((param + noise).data)
            # Update the non-perturbable params
            for p in self.actor.nonperturbable_params:
                param = self.actor.state_dict()[p].clone()
                self.pnp_actor.state_dict()[p].data.copy_(param.data)

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer'])
        actor_bundle = SaveBundle(model=self.actor.state_dict(),
                                  optimizer=self.actor_optimizer.state_dict())
        critic_bundle = SaveBundle(model=self.critic.state_dict(),
                                   optimizer=self.critic_optimizer.state_dict())
        torch.save(actor_bundle._asdict(), "{}_actor_iter{}.pth".format(path, iters))
        torch.save(critic_bundle._asdict(), "{}_critic_iter{}.pth".format(path, iters))

    def load(self, path, iters):
        experiment = path.split('/')[-1]
        actor_bundle = torch.load("{}/{}_actor_iter{}.pth".format(path, experiment, iters))
        self.actor.load_state_dict(actor_bundle['model'])
        self.actor_optimizer.load_state_dict(actor_bundle['optimizer'])
        critic_bundle = torch.load("{}/{}_critic_iter{}.pth".format(path, experiment, iters))
        self.critic.load_state_dict(critic_bundle['model'])
        self.critic_optimizer.load_state_dict(critic_bundle['optimizer'])
