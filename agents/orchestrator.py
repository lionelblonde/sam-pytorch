import time
import copy
import os
from collections import namedtuple, deque, OrderedDict
import yaml

import numpy as np
import visdom

from helpers import logger
from helpers.distributed_util import sync_check, mpi_mean_reduce
from agents.memory import RingBuffer
from helpers.console_util import (timed_cm_wrapper, pretty_iter,
                                  pretty_elapsed, columnize)


def rollout_generator(env, agent, rollout_len, prefill=0):

    pixels = len(env.observation_space.shape) >= 3

    # Reset noise processes
    agent.reset_noise()

    t = 0
    done = True
    env_rew = 0.0
    ob = env.reset()
    if pixels:
        ob = np.array(ob)

    obs = RingBuffer(rollout_len, shape=agent.ob_shape)
    acs = RingBuffer(rollout_len, shape=agent.ac_shape)
    syn_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    env_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    dones = RingBuffer(rollout_len, shape=(1,), dtype='int32')

    while True:
        ac = agent.predict(ob, apply_noise=True)
        # if t < prefill:
        #     logger.info("populating the replay buffer with uniform policy")
        #     # Override predicted action with actions which are sampled
        #     # from a uniform random distribution over valid actions
        #     ac = env.action_space.sample()
        # XXX GYM BUG. LEAVE HERE. `env.action_space.sample()` NON-DETERMINISTIC FUNCTION.

        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:

            obs_ = obs.data.reshape(-1, *agent.ob_shape)

            if not pixels:
                agent.rms_obs.update(obs_)

            out = {"obs": obs_,
                   "acs": acs.data.reshape(-1, *agent.ac_shape),
                   "syn_rews": syn_rews.data.reshape(-1, 1),
                   "env_rews": env_rews.data.reshape(-1, 1),
                   "dones": dones.data.reshape(-1, 1)}
            yield out

        obs.append(ob)
        acs.append(ac)
        dones.append(done)

        # Interact with env(s)
        new_ob, env_rew, done, _ = env.step(ac)

        env_rews.append(env_rew)

        syn_rew = np.asscalar(agent.discriminator.get_reward(ob, ac).cpu().numpy().flatten())
        syn_rews.append(syn_rew)

        # Store transition(s) in the replay buffer
        agent.store_transition(ob, ac, syn_rew, new_ob, done)

        ob = copy.copy(new_ob)
        if pixels:
            ob = np.array(ob)

        if done:
            agent.reset_noise()
            ob = env.reset()

        t += 1


def ep_generator(env, agent, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    ob = env.reset()
    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    env_rews = []

    while True:
        ac = agent.predict(ob, apply_noise=False)
        obs.append(ob)
        acs.append(ac)
        new_ob, env_rew, done, _ = env.step(ac)
        if render:
            env.render()
        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = copy.copy(new_ob)
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            yield {"obs": obs,
                   "acs": acs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            env_rews = []
            agent.reset_noise()
            ob = env.reset()


def evaluate(env,
             agent_wrapper,
             num_trajs,
             iter_num,
             render,
             model_path):

    # Rebuild the computational graph
    # Create an agent
    agent = agent_wrapper()
    # Create episode generator
    ep_gen = ep_generator(env, agent, render)
    # Initialize and load the previously learned weights into the freshly re-built graph

    # Load the model
    agent.load(model_path, iter_num)
    logger.info("model loaded from path:\n  {}".format(model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(args,
          rank,
          world_size,
          env,
          eval_env,
          agent_wrapper,
          experiment_name,
          ckpt_dir,
          enable_visdom,
          visdom_dir,
          visdom_server,
          visdom_port,
          visdom_username,
          visdom_password,
          save_frequency,
          pn_adapt_frequency,
          rollout_len,
          batch_size,
          training_steps_per_iter,
          eval_steps_per_iter,
          eval_frequency,
          actor_update_delay,
          d_update_ratio,
          render,
          expert_dataset,
          add_demos_to_mem,
          prefill,
          max_iters):

    assert training_steps_per_iter % actor_update_delay == 0, "must be a multiple"

    # Create an agent
    agent = agent_wrapper()

    if add_demos_to_mem:
        # Add demonstrations to memory
        agent.replay_buffer.add_demo_transitions_to_mem(expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, rollout_len, prefill)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, render)

    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    keys = ['ac', 'actor_gradnorms', 'actor_losses', 'critic_gradnorms', 'critic_losses']
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        keys.extend(['eval_ac', 'eval_len', 'eval_env_ret'])
    if agent.hps.enable_clipped_double:
        keys.extend(['twin_critic_gradnorms', 'twin_critic_losses'])
    if agent.param_noise is not None:
        keys.extend(['pn_dist', 'pn_cur_std'])
    Deques = namedtuple('Deques', keys)
    deques = Deques(**{k: deque(maxlen=maxlen) for k in keys})

    # Set up model save directory
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup Visdom
    if rank == 0 and enable_visdom:

        # Setup the Visdom directory
        os.makedirs(visdom_dir, exist_ok=True)

        # Create visdom
        viz = visdom.Visdom(env="experiment_{}_{}".format(int(time.time()), experiment_name),
                            log_to_filename=os.path.join(visdom_dir, "vizlog.txt"),
                            server=visdom_server,
                            port=visdom_port,
                            username=visdom_username,
                            password=visdom_password)
        assert viz.check_connection(timeout_seconds=4), "viz co not great"

        viz.text("World size: {}".format(world_size))
        iter_win = viz.text("will be overridden soon")
        mem_win = viz.text("will be overridden soon")
        viz.text(yaml.dump(args.__dict__, default_flow_style=False))

        keys = ['eval_len', 'eval_env_ret']
        if agent.param_noise is not None:
            keys.extend(['pn_dist', 'pn_cur_std'])

        keys.extend(['actor_loss', 'critic_loss', 'actor_gradnorm', 'critic_gradnorm'])
        if agent.hps.enable_clipped_double:
            keys.extend(['twin_critic_loss'])

        if args.algo == 'td4iqr':
            keys.extend(['cdf'])

        # Create (empty) visdom windows
        VizWins = namedtuple('VizWins', keys)
        vizwins = VizWins(**{k: viz.line(X=[0], Y=[np.nan]) for k in keys})
        # HAXX: NaNs ignored by visdom

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        if iters_so_far % 20 == 0:
            # Check if the mpi workers are still synced
            sync_check(agent.actor)
            sync_check(agent.critic)
            sync_check(agent.discriminator)

        if rank == 0 and iters_so_far % save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model:\n  @: {}".format(ckpt_dir))

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("interacting"):
            rollout = roll_gen.__next__()

        # Extend deques with collected experiential data
        deques.ac.extend(rollout['acs'])

        with timed("training"):
            for training_step in range(training_steps_per_iter):

                if agent.param_noise is not None:
                    if training_step % pn_adapt_frequency == 0:
                        # Adapt parameter noise
                        agent.adapt_param_noise()
                        # Store the action-space dist between perturbed and non-perturbed actors
                        deques.pn_dist.append(agent.pn_dist)
                        # Store the new std resulting from the adaption
                        deques.pn_cur_std.append(agent.param_noise.cur_std)

                # Train the actor-critic architecture
                update_critic = True
                update_critic = not bool(training_step % d_update_ratio)
                update_actor = update_critic and not bool(training_step % actor_update_delay)

                losses, gradnorms = agent.train(update_critic=update_critic,
                                                update_actor=update_actor)
                # Store the losses and gradients in their respective deques
                deques.actor_gradnorms.append(gradnorms['actor'])
                deques.actor_losses.append(losses['actor'])
                deques.critic_gradnorms.append(gradnorms['critic'])
                deques.critic_losses.append(losses['critic'])
                if agent.hps.enable_clipped_double:
                    deques.twin_critic_gradnorms.append(gradnorms['twin_critic'])
                    deques.twin_critic_losses.append(losses['twin_critic'])

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                with timed("evaluating"):

                    for eval_step in range(eval_steps_per_iter):

                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()

                        # Aggregate data collected during the evaluation to the buffers
                        deques.eval_ac.extend(eval_ep['acs'])
                        deques.eval_len.append(eval_ep['ep_len'])
                        deques.eval_env_ret.append(eval_ep['ep_env_ret'])

        # Log statistics

        logger.info("logging misc training stats")

        stats = OrderedDict()

        # Add min, max and mean of the components of the average action
        ac_np_mean = np.mean(deques.ac, axis=0)  # vector
        stats.update({'min_ac_comp': np.amin(ac_np_mean)})
        stats.update({'max_ac_comp': np.amax(ac_np_mean)})
        stats.update({'mean_ac_comp': np.mean(ac_np_mean)})
        stats.update({'mean_ac_comp_mpi': mpi_mean_reduce(ac_np_mean)})

        # Add gradient norms
        stats.update({'actor_gradnorm': np.mean(deques.actor_gradnorms)})
        stats.update({'critic_gradnorm': np.mean(deques.critic_gradnorms)})

        if agent.hps.enable_clipped_double:
            stats.update({'twin_critic_gradnorm': np.mean(deques.twin_critic_gradnorms)})

        if agent.param_noise is not None:
            stats.update({'pn_dist': np.mean(deques.pn_dist)})
            stats.update({'pn_cur_std': np.mean(deques.pn_cur_std)})

        # Add replay buffer num entries
        stats.update({'mem_num_entries': agent.replay_buffer.num_entries})

        # Log dictionary content
        logger.info(columnize(['name', 'value'], stats.items(), [24, 16]))

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                # Use the logger object to log the eval stats (will appear in `progress{}.csv`)
                logger.info("logging misc eval stats")
                # Add min, max and mean of the components of the average action
                ac_np_mean = np.mean(deques.eval_ac, axis=0)  # vector
                logger.record_tabular('min_ac_comp', np.amin(ac_np_mean))
                logger.record_tabular('max_ac_comp', np.amax(ac_np_mean))
                logger.record_tabular('mean_ac_comp', np.mean(ac_np_mean))
                # Add episodic stats
                logger.record_tabular('ep_len', np.mean(deques.eval_len))
                logger.record_tabular('ep_env_ret', np.mean(deques.eval_env_ret))
                logger.dump_tabular()

        # Mark the end of the iter in the logs
        logger.info('')

        iters_so_far += 1

        if rank == 0 and enable_visdom:

            viz.text("Current iter: {}".format(iters_so_far), win=iter_win, append=False)
            filled_ratio = agent.replay_buffer.num_entries / (1. * args.mem_size)  # HAXX
            viz.text("Replay buffer: {} (filled ratio: {})".format(agent.replay_buffer.num_entries,
                                                                   filled_ratio),
                     win=mem_win,
                     append=False)

            if iters_so_far % eval_frequency == 0:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_len)],
                         win=vizwins.eval_len,
                         update='append',
                         opts=dict(title='Eval Episode Length'))
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_env_ret)],
                         win=vizwins.eval_env_ret,
                         update='append',
                         opts=dict(title='Eval Episodic Return'))

            if agent.param_noise is not None:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.pn_dist)],
                         win=vizwins.pn_dist,
                         update='append',
                         opts=dict(title='Distance in action space (param noise)'))
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.pn_cur_std)],
                         win=vizwins.pn_cur_std,
                         update='append',
                         opts=dict(title='Parameter-Noise Current Std Dev'))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.actor_losses)],
                     win=vizwins.actor_loss,
                     update='append',
                     opts=dict(title="Actor Loss"))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.critic_losses)],
                     win=vizwins.critic_loss,
                     update='append',
                     opts=dict(title="Critic Loss"))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.actor_gradnorms)],
                     win=vizwins.actor_gradnorm,
                     update='append',
                     opts=dict(title="Actor Gradnorm"))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.critic_gradnorms)],
                     win=vizwins.critic_gradnorm,
                     update='append',
                     opts=dict(title="Critic Gradnorm"))

            if agent.hps.enable_clipped_double:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.twin_critic_losses)],
                         win=vizwins.twin_critic_loss,
                         update='append',
                         opts=dict(title="Twin Critic Loss"))
