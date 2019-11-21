import os
import platform
from mpi4py import MPI
import numpy as np
import torch


COMM = MPI.COMM_WORLD


def mpi_mean_like(x, comm=COMM):
    """Computes element-wise mean across workers.
    The output array has the same shape as the input array.
    This operation will fail if the array can be of different shapes across the workers.
    e.g. used for gradient averaging across workers or when averagin scalars
    """
    assert comm is not None
    num_workers = comm.Get_size()
    x = np.asarray(x)
    # Initialize element-wise sums across the workers
    sums = np.empty_like(x)
    # Sum x's elements across all mpi workers and put the result in `sums`
    comm.Allreduce(x, sums, op=MPI.SUM)
    means = sums / num_workers
    # Make sure input and output have the same shape
    assert means.shape == x.shape
    return means


def mpi_mean_reduce_v(x, comm=COMM, axis=0, keepdims=False):
    """Compute mean locally along `axis` and globally across mpi workers.
    This is the verbose version (hence the 'v') as the number of reductions (local and
    global) is also returned in the output tuple.
    """
    assert comm is not None
    x = np.asarray(x)
    assert x.ndim >= 1
    # Collapse to x.ndim-1 dimensions by summin along `axis`
    sums = x.sum(axis=axis, keepdims=keepdims)
    # Extract the number of elements
    n = sums.size
    # Create a vector of size n+1, put flattened `sums` in the first `n` slots
    # and put how many elements were reduced along `axis` in the n+1-th slot
    # (i.e. the number of elements involved in each reduction)
    local_sums = np.zeros(n + 1, dtype=x.dtype)
    flattened_sums = sums.ravel()
    reduction_depth = x.shape[axis]
    local_sums[:n] = flattened_sums
    local_sums[n] = reduction_depth
    # Sum local_sums's elements across all mpi workers and put the result in `global_sum`
    global_sums = np.zeros_like(local_sums)
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    # Unflatten the result (back to post-reduction along `axis` shape) and
    # divide by the sum across workers of numbers of reductions.
    # In fine, in the returned tensor, each element corresponds to a sum along axis (local)
    # and across workers (global) divided by the sum (across workers) of number of local
    # reductions.
    return global_sums[:n].reshape(sums.shape) / global_sums[n], global_sums[n]


def mpi_mean_reduce(x, comm=COMM, axis=0, keepdims=False):
    """Almost like 'mpi_mean_reduce_v', but only returns the mpi mean"""
    mpi_mean, _ = mpi_mean_reduce_v(x=x, comm=comm, axis=axis, keepdims=keepdims)
    return mpi_mean


def mpi_moments(x, comm=COMM, axis=0, keepdims=False):
    """Compute mpi moments"""
    assert comm is not None
    x = np.asarray(x)
    assert x.ndim >= 1
    # Compute mean
    mean, count = mpi_mean_reduce_v(x, axis=axis, comm=comm, keepdims=True)
    # Compute standard deviation
    squared_diffs = np.square(x - mean)
    mean_squared_diff, count1 = mpi_mean_reduce_v(squared_diffs, axis=axis,
                                                  comm=comm, keepdims=True)
    assert count1 == count1  # verify that nothing ominous happened when squaring
    std = np.sqrt(mean_squared_diff)
    if not keepdims:
        new_shape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(new_shape)
        std = std.reshape(new_shape)
    return mean, std, count


def average_gradients(model, device):
    for param in model.parameters():
        # Place the gradients on cpu
        grads = param.grad.cpu().data.numpy()
        # Average the gradients across workers
        avg_grads = mpi_mean_like(grads)
        # Create a torch tensor out of it
        avg_grads_tensor = torch.FloatTensor(avg_grads).to(device)
        # Replace the param's gradient by the mpi-average
        param.grad.copy_(avg_grads_tensor)


def sync_with_root(model, comm=COMM):
    """Send the root node parameters to every sbire"""
    comm.Barrier()
    rank = comm.Get_rank()
    for param in model.parameters():
        if rank == 0:
            comm.Bcast(param.cpu().data.numpy(), root=0)
        else:
            param_ = np.empty_like(param.cpu().data)
            comm.Bcast(param_, root=0)
            param_ = torch.FloatTensor(param_)
            param.data.copy_(param_.data)


def sync_check(model, comm=COMM):
    """Check whether mpi worker still have the same weights"""
    comm.Barrier()
    rank = comm.Get_rank()
    for param in model.parameters():
        if rank == 0:
            comm.Bcast(param.cpu().data.numpy(), root=0)
        else:
            param_ = np.empty_like(param.cpu().data)
            comm.Bcast(param_, root=0)
            param_ = torch.FloatTensor(param_)
            assert torch.all(torch.eq(param.cpu(), param_.cpu())), "not in sync anymore"


def guess_available_gpus(n_gpus=None):
    """Retrieve availble gpus"""
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_devices = cuda_visible_devices.split(',')
        return [int(n) for n in cuda_visible_devices]
    if 'RCALL_NUM_GPU' in os.environ:
        n_gpus = int(os.environ['RCALL_NUM_GPU'])
        return list(range(n_gpus))
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("couldn't guess the available gpus on this machine")


def setup_mpi_gpus(comm=COMM):
    """Set CUDA_VISIBLE_DEVICES using MPI"""
    available_gpus = guess_available_gpus()
    node_id = platform.node()
    nodes_ordered_by_rank = comm.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:comm.Get_rank()]
                                        if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])
    print("rank {} will use gpu {}".format(local_rank, os.environ['CUDA_VISIBLE_DEVICES']))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Normalizer.

class RunMoms(object):

    def __init__(self, shape, comm=COMM, use_mpi=True):
        """Maintain running statistics across wrokers leveraging Chan's method"""
        # Initialize mean and var with float64 precision (objectively more accurate)
        self.mean = np.zeros(shape, dtype='float64')
        self.var = np.ones(shape, dtype='float64') + 1e-5  # HAXX to avoid any division by zero
        self.count = 1e-4  # HAXX to avoid any division by zero
        self.comm = comm
        self.use_mpi = use_mpi

    def update(self, x):
        """Update running statistics using the new batch's statistics"""
        # Compute the statistics of the batch
        if self.use_mpi:
            batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
        else:
            batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        # Update moments
        self.update_moms(batch_mean, batch_var, batch_count)

    def update_moms(self, batch_mean, batch_var, batch_count):
        """ Implementation of Chan's method to compute and maintain mean and variance estimates
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        # Compute new mean
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        # Compute new var
        new_var = M2 / tot_count
        # Compute new count
        new_count = tot_count
        # Update moments
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
