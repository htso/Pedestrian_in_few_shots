import torch
from math import log, pi

def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean**2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)

# This code is checked against the follow SO calculation : 
#      https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# NOTATION :
#   q_mean = mu_1
#   p_mean = mu_2
#   q_logvar = log(sigma_1 ^ 2)
#   p_logvar = log(sigma_2 ^ 2)
#
# SO formula,
#
#   KL(q, p) = log(sigma_2/sigma_1) - 1/2 + ((mu_1 - mu_2)^2 + sigma_1^2) / 2*sigma_2^2
#
def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)
