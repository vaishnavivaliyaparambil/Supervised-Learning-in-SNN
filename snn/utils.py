import torch

def spiketrain_to_spiketimes(spike_train):
    _, times = torch.where(spike_train == 1)
    return spike_train.shape[1] - times

def eps_matrix(spike_times, rows, cols, eps_fn):
    matrix = torch.zeros((rows, cols), dtype=torch.float32)
    matrix[:, :] = eps_fn(spike_times[:, 0])
    return torch.nan_to_num(matrix, nan=0.0)

def eta_matrix(spike_times, rows, cols, eta_fn):
    matrix = torch.zeros((rows, cols), dtype=torch.float32)
    matrix[:, :] = eta_fn(spike_times[:, 0])
    return matrix
