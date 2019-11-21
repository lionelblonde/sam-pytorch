import numpy as np
import torch

from helpers import logger


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_map):
        self.data_map = data_map

    def __getitem__(self, i):
        return {k: v[i, ...].astype(np.float32) for k, v in self.data_map.items()}

    def __len__(self):
        return list(self.data_map.values())[0].shape[0]


class DemoDataset(Dataset):

    def __init__(self, expert_path, num_demos):
        self.num_demos = num_demos

        with np.load(expert_path, allow_pickle=True) as data:
            self.data_map, self.stat_map = {}, {}
            for k, v in data.items():
                if k in ['ep_env_rets', 'ep_lens']:
                    self.stat_map[k] = v
                elif k in ['obs0', 'acs', 'env_rews', 'dones1', 'obs1']:
                    self.data_map[k] = np.array(np.concatenate((v[:num_demos])))

        fmtstr = "[DEMOS] >>>> extracted {} transitions, from {} trajectories"
        logger.info(fmtstr.format(len(self), self.num_demos))
        rets_, lens_ = self.stat_map['ep_env_rets'], self.stat_map['ep_lens']
        logger.info("  episodic return: {}({})".format(np.mean(rets_), np.std(rets_)))
        logger.info("  episodic length: {}({})".format(np.mean(lens_), np.std(lens_)))
