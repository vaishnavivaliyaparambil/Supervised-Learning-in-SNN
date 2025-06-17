def train_model():
    import os
    import torch
    from snn import kernels, utils, srm, loss
    from snn.params import *

    if SAVE_RESULTS:
        os.makedirs(SAVE_PATH, exist_ok=True)

    input_weights = torch.rand(NUM_NEURONS_IN, N, requires_grad=True, device=MY_DEVICE)
    output_weights = torch.rand(NUM_NEURONS_OUT, N, requires_grad=True, device=MY_DEVICE)

    errors = []

    for trial in range(N_TRIALS):
        input_cache = srm.initialize_cache(NUM_NEURONS_IN, N, MY_DEVICE)
        output_cache = srm.initialize_cache(NUM_NEURONS_OUT, N, MY_DEVICE)

        input_spikes = torch.zeros((NUM_NEURONS_IN, N), device=MY_DEVICE)
        input_spikes[:, torch.rand(N) < 0.5] = 1

        inter_spikes = torch.zeros_like(input_spikes)
        output_spikes = torch.zeros((NUM_NEURONS_OUT, N), device=MY_DEVICE)

        for t in range(N):
            _, in_spikers = srm.simulate_spikes(t, input_spikes, input_weights, input_cache,
                                                THRESHOLD, 
                                                lambda s: kernels.eta(s, A, TAU_M),
                                                lambda s: kernels.eps(s, ALPHA, BETA, TAU_C),
                                                SIMULATION_WINDOW, MY_DEVICE)
            inter_spikes[in_spikers, t] = 1

            _, out_spikers = srm.simulate_spikes(t, inter_spikes, output_weights, output_cache,
                                                 THRESHOLD,
                                                 lambda s: kernels.eta(s, A, TAU_M),
                                                 lambda s: kernels.eps(s, ALPHA, BETA, TAU_C),
                                                 SIMULATION_WINDOW, MY_DEVICE)
            output_spikes[out_spikers, t] = 1

        err = loss.error_function(input_spikes, output_spikes, TAU)
        errors.append(err)

        if VERBOSE:
            print(f"Trial {trial}: Error = {err}")

        if SAVE_RESULTS:
            torch.save(output_weights.detach().cpu(), os.path.join(SAVE_PATH, f"weights_trial{trial}.pt"))
            torch.save(output_spikes.cpu(), os.path.join(SAVE_PATH, f"spikes_trial{trial}.pt"))
            torch.save(torch.tensor(errors), os.path.join(SAVE_PATH, "errors.pt"))
