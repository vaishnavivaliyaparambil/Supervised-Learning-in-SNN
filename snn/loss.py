import torch
from itertools import combinations

def error_function(desired_spikes, output_spikes, tau):
    errors = []
    for d, o in zip(desired_spikes, output_spikes):
        d_times = torch.sort(d[d > 0])[0]
        o_times = torch.sort(o[o > 0])[0]
        
        sigma1 = sum((v[0]*v[1])/(v[0]+v[1])*torch.exp(-(v[0]+v[1])/tau) for v in combinations(d_times, 2))
        sigma2 = sum((u[0]*u[1])/(u[0]+u[1])*torch.exp(-(u[0]+u[1])/tau) for u in combinations(o_times, 2))
        sigma3 = sum((u*v)/(u+v)*torch.exp(-(u+v)/tau) for u in o_times for v in d_times)

        errors.append(sigma1 + sigma2 - 2*sigma3)
    return torch.tensor(errors)

def compute_gradient(desired_spikes, output_spikes, tau):
    grads = []
    for d, o in zip(desired_spikes, output_spikes):
        grad = torch.zeros_like(o, dtype=torch.float32)
        for i, u in enumerate(o):
            sigma1 = sum(z * (((z - u) - z / tau * (z + u)) / (z + u)**3) * torch.exp(-(z + u)/tau) for z in o if z != u)
            sigma2 = sum(v * (((v - u) - u / tau * (v + u)) / (v + u)**3) * torch.exp(-(v + u)/tau) for v in d)
            grad[i] = 2 * (sigma1 - sigma2)
        grads.append(grad)
    return grads
