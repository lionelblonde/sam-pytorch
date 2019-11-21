import datetime
import random
import os
import os.path as osp

import numpy as np
import yaml

from helpers import logger


def uuid(num_syllables=2, num_parts=3):
    """Randomly create a semi-pronounceable uuid"""
    part1 = ['s', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr']
    part2 = ['a', 'oo', 'ee', 'e', 'u', 'er']
    seps = ['_']  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


class ConfigDumper:

    def __init__(self, args, path=None):
        """Log the job config into a file"""
        self.args = args
        os.makedirs(path, exist_ok=True)
        self.path = path

    def dump(self):
        hpmap = self.args.__dict__
        with open(osp.join(self.path, 'hyperparameters.yml'), 'w') as outfile:
            yaml.safe_dump(hpmap, outfile, default_flow_style=False)


class ExperimentInitializer:

    def __init__(self, args, rank=None, world_size=None):
        """Initialize the experiment"""
        self.uuid = uuid() if args.uuid is None else args.uuid
        self.args = args
        self.rank = rank
        self.world_size = world_size
        # Set printing options
        np.set_printoptions(precision=3)

    def configure_logging(self):
        """Configure the experiment"""
        if self.rank is None:  # eval
            logger.info("configuring logger for evaluation")
            logger.configure(dir_=None, format_strs=['stdout'])

        elif self.rank == 0:  # train, master
            log_path = osp.join(self.args.log_dir, self.get_name())
            formats_strs = ['stdout', 'log', 'csv']
            fmtstr = "configuring logger"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.configure(dir_=log_path, format_strs=formats_strs)
            fmtstr = "logger configured"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.info("  directory: {}".format(log_path))
            logger.info("  output formats: {}".format(formats_strs))
            # In the same log folder, log args in a YAML file
            config_dumper = ConfigDumper(args=self.args, path=log_path)
            config_dumper.dump()
            fmtstr = "experiment configured"
            fmtstr += " [{} MPI workers]".format(self.world_size)
            logger.info(fmtstr)

        else:  # train, worker
            logger.info("configuring logger [worker #{}]".format(self.rank))
            logger.configure(dir_=None, format_strs=None)
            logger.set_level(logger.DISABLED)

    def prepend_date(undated_name):
        """Prepend the date to a file or directory name"""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S__{}".format(undated_name))

    def get_name(self):
        """Assemble long experiment name"""
        name = self.uuid + '.'
        name += "{}.{}.".format(self.args.task, self.args.algo)
        if 'evaluate' in self.args.task:
            name += "{}.".format(self.args.task)
            assert self.args.num_trajs != np.inf, "num trajs must be finite"
            name += "num_trajs_{}.".format(self.args.num_trajs)
        name += self.args.env_id
        name += ".seed_{}".format(str(self.args.seed).zfill(2))
        return name
