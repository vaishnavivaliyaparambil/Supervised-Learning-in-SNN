import torch

def eta(s, A, TAU_M):
    s = s.clone().detach().float().requires_grad_(True)
    return torch.where(s > 0, -A * torch.exp(-s / TAU_M), torch.zeros_like(s))

def d_eta(s, A, TAU_M):
    s = s.clone().detach().float().requires_grad_(True)
    eta_val = torch.where(s > 0, -A * torch.exp(-s / TAU_M), torch.zeros_like(s))
    eta_val.backward(torch.ones_like(s))
    return s.grad

def eps(s, ALPHA, BETA, TAU_C):
    s = s.clone().detach().float().requires_grad_(True)
    part1 = 1 / (ALPHA * torch.sqrt(s))
    part2 = torch.exp(-BETA * (ALPHA ** 2) / s)
    part3 = torch.exp(-s / TAU_C)
    return torch.where(s > 0, part1 * part2 * part3, torch.zeros_like(s))

def d_eps(s, ALPHA, BETA, TAU_C):
    s = s.clone().detach().float().requires_grad_(True)
    eps_val = eps(s, ALPHA, BETA, TAU_C)
    eps_val.backward(torch.ones_like(s))
    return s.grad
