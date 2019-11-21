import argparse
from copy import deepcopy
import os
import os.path as osp
import numpy as np
from subprocess import call, check_output
from copy import copy
import yaml

from helpers import logger
from helpers.misc_util import zipsame, boolean_flag
from helpers.experiment import uuid as create_uuid


parser = argparse.ArgumentParser(description="Job Spawner")
parser.add_argument('--config', type=str, default=None)
boolean_flag(parser, 'call', default=False, help="launch immediately?")
boolean_flag(parser, 'sweep', default=False, help="hp search?")
parser.add_argument('--visdom_server', type=str, default=None)
parser.add_argument('--visdom_port', type=str, default=None)
parser.add_argument('--visdom_username', type=str, default=None)
parser.add_argument('--visdom_password', type=str, default=None)
args = parser.parse_args()

# Retrieve config from filesystem
CONFIG = yaml.safe_load(open(args.config))
# Retrieve list of admissible environments and demos
ADM = yaml.safe_load(open("admissible_envs.yaml"))['environments']
# Extract parameters from config
NUM_SEEDS = CONFIG['parameters']['num_seeds']
CLUSTER = CONFIG['resources']['cluster']
CONDA = CONFIG['resources']['conda_env']
# Define experiment type
TYPE = 'sweep' if args.sweep else 'fixed'
# Write out the boolean arguments (using the 'boolean_flag' function)
BOOL_ARGS = ['cuda', 'pixels', 'enable_visdom',
             'render', 'record', 'with_layernorm',
             'prioritized_replay', 'ranked', 'unreal',
             'n_step_returns', 'add_demos_to_mem',
             'enable_clipped_double', 'enable_targ_actor_smoothing',
             'state_only', 'minimax_only']

# Create the list of environments from the indicated benchmark
BENCH = CONFIG['parameters']['benchmark']
DIFFICULTY = CONFIG['parameters']['difficulty']
assert DIFFICULTY in ['easy', 'normal', 'hard']
if BENCH == 'mujoco':
    map_ = {'easy': ['InvertedPendulum-v2'],  # debug, sanity check
            'normal': ['InvertedPendulum-v2',
                       'Hopper-v3',
                       'Walker2d-v3'],
            'hard': ['InvertedPendulum-v2',
                     'Hopper-v3',
                     'Walker2d-v3',
                     'HalfCheetah-v3',
                     'Ant-v3'],  # paper-grade
            }
    ENVS = map_[DIFFICULTY]
else:
    raise NotImplementedError("benchmark not covered by the spawner.")

# Create the list of demonstrations needed
DEMOS = {k: osp.join(CONFIG['parameters']['demo_dir'], ADM[BENCH][k]) for k in ENVS}


def copy_and_add_seed(hpmap, seed):
    hpmap_ = deepcopy(hpmap)
    # Add the seed and edit the job uuid to only differ by the seed
    hpmap_.update({'seed': seed})
    # Enrich the uuid with extra information
    hpmap_.update({'uuid': "{}.{}.{}.seed{}.{}".format(hpmap['task'],
                                                       hpmap['algo'],
                                                       hpmap['uuid'],
                                                       str(seed).zfill(2),
                                                       hpmap['env_id'])})
    return hpmap_


def copy_and_add_env(hpmap, env):
    hpmap_ = deepcopy(hpmap)
    # Add the env and if demos are needed, add those too
    hpmap_.update({'env_id': env})
    hpmap_.update({'expert_path': DEMOS[env]})
    return hpmap_


def rand_tuple_from_list(list_):
    """Return a random tuple from a list of tuples"""
    # Note: `np.random.choice` does not work on lists of tuples, hence this
    assert all(isinstance(v, tuple) for v in list_), "not a list of tuples"
    return list_[np.random.randint(low=0, high=len(list_))]


def get_hps(sweep):
    """Return a list of maps of hyperparameters"""
    # Create a uuid to identify the current job
    uuid = create_uuid()
    # Assemble the hyperparameter map
    if sweep:
        # Random search
        hpmap = {
            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'pixels': CONFIG['parameters']['pixels'],
            'checkpoint_dir': CONFIG['logging']['checkpoint_dir'],
            'log_dir': CONFIG['logging']['log_dir'],
            'enable_visdom': CONFIG['logging']['enable_visdom'],
            'visdom_server': args.visdom_server,
            'visdom_port': args.visdom_port,
            'visdom_username': args.visdom_username,
            'visdom_password': args.visdom_password,
            'render': False,
            'record': False,
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': CONFIG['parameters'].get('save_frequency', 400),
            'num_iters': int(float(CONFIG['parameters'].get('num_iters', 1e6))),
            'training_steps_per_iter': np.random.choice([10, 25, 50]),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 50),
            'eval_frequency': CONFIG['parameters'].get('eval_frequency', 10),

            # Model
            'with_layernorm': CONFIG['parameters'].get('with_layernorm', False),
            'wd_scale': float(np.random.choice([1e-4, 3e-4, 1e-3]),),

            # Optimization
            'actor_lr': float(np.random.choice([1e-3, 3e-4])),
            'critic_lr': float(np.random.choice([1e-3, 3e-4])),
            'clip_norm': np.random.choice([5., 10., 20., 40.]),

            # Algorithm
            'rollout_len': np.random.choice([1, 2, 5]),
            'batch_size': np.random.choice([32, 64, 128]),
            'gamma': np.random.choice([0.99, 0.995]),
            'polyak': np.random.choice([0.001, 0.005, 0.01]),
            'reward_scale': CONFIG['parameters'].get('reward_scale', 1.),
            'noise_type': np.random.choice(['"adaptive-param_0.2, normal_0.2"',
                                            '"adaptive-param_0.2, ou_0.2"',
                                            '"normal_0.2"',
                                            '"ou_0.2"']),
            'pn_adapt_frequency': CONFIG['parameters'].get('pn_adapt_frequency', 50),
            'mem_size': np.random.choice([10000, 50000, 100000]),
            'add_demos_to_mem': CONFIG['parameters'].get('add_demos_to_mem', False),

            # Prioritized replay
            'prioritized_replay': CONFIG['parameters'].get('prioritized_replay', False),
            'alpha': CONFIG['parameters'].get('alpha', 0.3),
            'beta': CONFIG['parameters'].get('beta', 1.),
            'ranked': CONFIG['parameters'].get('ranked', False),
            'unreal': CONFIG['parameters'].get('unreal', False),
            'n_step_returns': CONFIG['parameters'].get('n_step_returns', False),
            'lookahead': np.random.choice([5, 10, 20, 40, 60]),

            # TD3-specific
            'enable_clipped_double': CONFIG['parameters'].get('enable_clipped_double', False),
            'enable_targ_actor_smoothing': CONFIG['parameters'].get('enable_targ_actor_smoothing', False),
            'td3_std': CONFIG['parameters'].get('td3_std', 0.2),
            'td3_c': CONFIG['parameters'].get('td3_c', 0.5),
            'actor_update_delay': np.random.choice([2, 3, 4]),
            'prefill': CONFIG['parameters'].get('prefill', 200),

            # SAM-specific
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'ent_reg_scale': CONFIG['parameters'].get('ent_reg_scale', 0.),
            'd_update_ratio': CONFIG['parameters'].get('d_update_ratio', 2),
            'num_demos': CONFIG['parameters'].get('num_demos', 0),
        }
    else:
        # No search, fixed map
        hpmap = {
            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'pixels': CONFIG['parameters']['pixels'],
            'checkpoint_dir': CONFIG['logging']['checkpoint_dir'],
            'log_dir': CONFIG['logging']['log_dir'],
            'enable_visdom': CONFIG['logging']['enable_visdom'],
            'visdom_server': args.visdom_server,
            'visdom_port': args.visdom_port,
            'visdom_username': args.visdom_username,
            'visdom_password': args.visdom_password,
            'render': False,
            'record': False,
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': CONFIG['parameters'].get('save_frequency', 400),
            'num_iters': int(float(CONFIG['parameters'].get('num_iters', 1e6))),
            'training_steps_per_iter': CONFIG['parameters'].get('train_steps_per_iter', 20),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 50),
            'eval_frequency': CONFIG['parameters'].get('eval_frequency', 10),

            # Model
            'with_layernorm': CONFIG['parameters'].get('with_layernorm', False),
            'wd_scale': float(CONFIG['parameters'].get('wd_scale', 3e-4)),

            # Optimization
            'actor_lr': float(CONFIG['parameters'].get('actor_lr', 3e-4)),
            'critic_lr': float(CONFIG['parameters'].get('critic_lr', 3e-4)),
            'clip_norm': CONFIG['parameters'].get('clip_norm', 5.),

            # Algorithm
            'rollout_len': CONFIG['parameters'].get('rollout_len', 5),
            'batch_size': CONFIG['parameters'].get('batch_size', 128),
            'gamma': CONFIG['parameters'].get('gamma', 0.99),
            'polyak': CONFIG['parameters'].get('polyak', 0.005),
            'reward_scale': CONFIG['parameters'].get('reward_scale', 1.),
            'noise_type': CONFIG['parameters'].get('noise_type', '"normal_0.2"'),
            'pn_adapt_frequency': CONFIG['parameters'].get('pn_adapt_frequency', 50),
            'mem_size': CONFIG['parameters'].get('mem_size', 50000),
            'add_demos_to_mem': CONFIG['parameters'].get('add_demos_to_mem', False),

            # Prioritized replay
            'prioritized_replay': CONFIG['parameters'].get('prioritized_replay', False),
            'alpha': CONFIG['parameters'].get('alpha', 0.3),
            'beta': CONFIG['parameters'].get('beta', 1.),
            'ranked': CONFIG['parameters'].get('ranked', False),
            'unreal': CONFIG['parameters'].get('unreal', False),
            'n_step_returns': CONFIG['parameters'].get('n_step_returns', False),
            'lookahead': CONFIG['parameters'].get('lookahead', 60),

            # TD3-specific
            'enable_clipped_double': CONFIG['parameters'].get('enable_clipped_double', False),
            'enable_targ_actor_smoothing': CONFIG['parameters'].get('enable_targ_actor_smoothing', False),
            'td3_std': CONFIG['parameters'].get('td3_std', 0.2),
            'td3_c': CONFIG['parameters'].get('td3_c', 0.5),
            'actor_update_delay': CONFIG['parameters'].get('actor_update_delay', 2),
            'prefill': CONFIG['parameters'].get('prefill', 200),

            # SAM-specific
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'ent_reg_scale': CONFIG['parameters'].get('ent_reg_scale', 0.),
            'd_update_ratio': CONFIG['parameters'].get('d_update_ratio', 2),
            'num_demos': CONFIG['parameters'].get('num_demos', 0),
        }

    # Duplicate for each environment
    hpmaps = [copy_and_add_env(hpmap, env)
              for env in ENVS]

    # Duplicate for each seed
    hpmapz = [copy_and_add_seed(hpmap_, seed)
              for hpmap_ in hpmaps
              for seed in range(NUM_SEEDS)]

    # Verify that the correct number of configs have been created
    assert len(hpmapz) == NUM_SEEDS * len(ENVS)

    return hpmapz


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    indent = 4 * ' '  # indents are defined as 4 spaces
    arguments = ""

    for k, v in hpmap.items():
        if k in BOOL_ARGS:
            if v is False:
                argument = "no-{}".format(k)
            else:
                argument = "{}".format(k)
        else:
            argument = "{}={}".format(k, v)

        arguments += "{}--{} \\\n".format(indent, argument)

    return arguments


def create_job_str(name, command):
    """Build the batch script that launches a job"""

    if CLUSTER == 'baobab':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --mem=32000\n')
        if CONFIG['parameters']['cuda']:
            contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
            bash_script_str += ('#SBATCH --gres=gpu:1\n'
                                '#SBATCH --constraint="{}"\n'.format(contraint))
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/6.3.0-2.27\n')
        if CONFIG['parameters']['cuda']:
            bash_script_str += ('module load CUDA\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {}')

        bash_script_str = bash_script_str.format(name,
                                                 CONFIG['resources']['partition'],
                                                 CONFIG['resources']['num_workers'],
                                                 CONFIG['resources']['timeout'],
                                                 command)
    elif CLUSTER == 'cscs':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --constraint=gpu\n\n')
        # Load modules
        bash_script_str += ('module load daint-gpu\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {}')

        bash_script_str = bash_script_str.format(name,
                                                 CONFIG['resources']['partition'],
                                                 CONFIG['resources']['num_workers'],
                                                 CONFIG['resources']['timeout'],
                                                 command)
    elif CLUSTER == 'local':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpiexec -n {} --use-hwthread-cpus {}')
        bash_script_str = bash_script_str.format(name,
                                                 CONFIG['resources']['num_workers'],
                                                 command)
    else:
        raise NotImplementedError("cluster selected is not covered.")

    return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args):
    """Spawn jobs"""

    # Create directory for spawned jobs
    os.makedirs("spawn", exist_ok=True)
    if CLUSTER == 'local':
        os.makedirs("tmux", exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [get_hps(sweep=True)
                   for _ in range(CONFIG['parameters']['num_trials'])]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = get_hps(sweep=False)

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = ["{}{}_{}".format(TYPE, str(i).zfill(3), hpmap['uuid'])
             for i, hpmap in enumerate(hpmaps)]
    # Finally get all the required job strings
    jobs = [create_job_str(name, command) for name, command in zipsame(names, commands)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} ready to submit. Config below.".format(i))
        logger.info(job + "\n")
        dir_ = name.split('.')[2]
        os.makedirs("spawn/{}".format(dir_), exist_ok=True)
        job_name = "spawn/{}/{}.sh".format(dir_, name)
        with open(job_name, 'w') as f:
            f.write(job)
        if args.call and not CLUSTER == 'local':
            # Spawn the job!
            check_output(["sbatch", "spawn/{}/{}".format(dir_, job_name)])
            logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} submitted.".format(i))
    # Summarize the number of jobs spawned
    logger.info(">>>>>>>>>>>>>>>>>>>> {} jobs were spawned.".format(len(jobs)))

    if CLUSTER == 'local':
        dir_ = hpmaps[0]['uuid'].split('.')[2]  # arbitrarilly picked index 0
        session_name = "{}_{}seeds_{}".format(TYPE, str(NUM_SEEDS).zfill(2), dir_)
        yaml_content = {'session_name': session_name, 'windows': []}
        for i, name in enumerate(names):
            executable = "{}.sh".format(name)
            single_pane = {'shell_command': ["source activate {}".format(CONDA),
                                             "chmod u+x spawn/{}/{}".format(dir_, executable),
                                             "spawn/{}/{}".format(dir_, executable)]}
            yaml_content['windows'].append({'window_name': "job{}".format(str(i).zfill(2)),
                                            'panes': [single_pane]})
        # Dump the assembled tmux config into a yaml file
        job_config = "tmux/{}.yaml".format(session_name)
        with open(job_config, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        if args.call:
            # Spawn all the jobs in the tmux session!
            check_output(["tmuxp", "load", "{}".format(job_config)])



if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
