import torch

def initialize_cache(num_neurons, N, device):
    return {
        'spiketrains': torch.zeros((num_neurons, N), dtype=torch.int32, device=device),
        'last_t': -1,
        'last_spike': torch.ones(num_neurons, dtype=torch.int32, device=device) * -1e6,
        'last_potential': torch.zeros(num_neurons, dtype=torch.float32, device=device)
    }

def simulate_spikes(t, spike_train, weights, cache, threshold, kernel_eta, kernel_eps, simulation_window, device):
    window_start = max(0, t + 1 - simulation_window)
    window = torch.flip(spike_train[:, window_start:t + 1], dims=(1,))

    timestep = window.shape[1]
    spiketimes_list = torch.arange(timestep + 1, 1, -1, device=device).view(timestep, 1)
    epsilon_matrix = kernel_eps(spiketimes_list)

    past_spiketimes = torch.nonzero(cache['spiketrains'])[:, -1][-simulation_window:]
    past_spike_mat = torch.zeros((spike_train.shape[0], simulation_window), device=device)
    if len(past_spiketimes) > 0:
        past_spike_mat[:, :len(past_spiketimes)] = past_spiketimes

    weights_window = weights[:, window_start:t + 1]
    incoming = torch.mul(weights_window, window)
    potential_change = torch.sum(incoming * epsilon_matrix, axis=1)

    total_potential = torch.sum(kernel_eta(torch.ones_like(past_spike_mat) * t - past_spike_mat), dim=1) + potential_change
    spiking_neurons = (total_potential > threshold) & (cache['last_potential'] < threshold)

    if spiking_neurons.any():
        cache['spiketrains'][spiking_neurons, t] = 1
        cache['last_spike'][spiking_neurons] = t

    cache['last_potential'] = total_potential
    cache['last_t'] = t

    return total_potential, spiking_neurons.nonzero().flatten()
