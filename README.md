# SAM: Sample-efficient Adversarial Mimic

PyTorch implementation of our work:
"Sample-Efficient Imitation Learning via Generative Adversarial Nets".
A TensorFlow implementation is also available at [sam-tf](https://github.com/lionelblonde/sam-tf). 

Published in AISTATS 2019 |
[arXiv link](https://arxiv.org/abs/1809.02064) |
[Video demos](https://youtu.be/-nCsqUJnRKU) |
[Expert demonstrations](https://drive.google.com/drive/folders/1ihVMUk9Ewm7cHv4tpFgnDkXkxNXjDYeS?usp=sharing)

![grid](images/paper_npages11_ncols4_dpi100.jpg)

# How to

Launching scripts are available in `/tasks`.
To run a task, use:
```bash
python spawner.py \
    --config tasks/sam_fixed_local_mujoco.yaml \
    --no-sweep \
    --call \
    --visdom_server=<visdom_server> \
    --visdom_port=<visdom_port> \
    --visdom_username=<visdom_username> \
    --visdom_password=<visdom_password>
```
Visdom can be disabled (and the associated options can therefore be omitted) by setting the argument `enable_visdom` to `false` in the YAML configuration file.
The command triggers the creation of a tmux session in which jobs run in distinct windows. For example, if the configuration file specifies 3 environments and a number of random seeds equal to 4, the tmux session will have 12 windows, each containing its unique pair of environment and random seed.

## Acknowledgments

* Interaction simulated with the [openai/gym](https://github.com/openai/gym) API.
* Some utilities were inspired from [openai/baselines](https://github.com/openai/baselines).
