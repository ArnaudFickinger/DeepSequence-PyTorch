###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch
import torch.nn.functional as F
import numpy as np
from options import Options
import math
from utils import *

import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options().parse()

def loss_theano(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff):
    return (logpx_z+warm_up_scale*kld_latent_theano(mu, logsigma)).mean()+warm_up_scale*sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff

def kld_latent_theano(mu, log_sigma):
    KLD_latent = 0.5 * (1.0 + 2.0 * log_sigma - mu ** 2.0 - (2.0 * log_sigma).exp()).sum(1)
    return KLD_latent

def KLD_diag_gaussians_theano(mu, log_sigma, prior_mu, prior_log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        # return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu).sqrt()) * math.exp(-2. * prior_log_sigma) - 0.5
        return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu)**2) * math.exp(-2. * prior_log_sigma) - 0.5

def sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l):
    # print("sparse")
    # print(KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum())
    # print(KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())
    return - (KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b1, logsigma_b1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W2, logsigma_W2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b2, logsigma_b2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W3, logsigma_W3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b3, logsigma_b3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_C, logsigma_C, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_l, logsigma_l, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())


def total_loss_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff):
    return -sparse_ELBO_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)

def sparse_ELBO_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff):
    return ELBO_original(logpx_z, mu, logsigma, warm_up_scale) - warm_up_scale*sparse_weight_reg_paper(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff

def sparse_weight_reg_paper(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l):
    return kld_diag_gaussian_normal_original_for_reg(mu_W1, logsigma_W1)+kld_diag_gaussian_normal_original_for_reg(mu_b1, logsigma_b1)+kld_diag_gaussian_normal_original_for_reg(mu_W2, logsigma_W2)+kld_diag_gaussian_normal_original_for_reg(mu_b2, logsigma_b2)+kld_diag_gaussian_normal_original_for_reg(mu_W3, logsigma_W3)+kld_diag_gaussian_normal_original_for_reg(mu_b3, logsigma_b3)+kld_diag_gaussian_normal_original_for_reg(mu_C, logsigma_C)+kld_diag_gaussian_normal_original_for_reg(mu_l, logsigma_l)+kld_diag_gaussians_original_for_reg(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse)

def ELBO_original(logpx_z, mu, logsigma, warm_up_scale):
    return logpx_z.mean() - warm_up_scale*kld_diag_gaussian_normal_original(mu, logsigma)

def ELBO_no_mean(logpx_z, mu, logsigma, warm_up_scale):
    return logpx_z + warm_up_scale*kld_diag_gaussian_normal_original_no_mean(mu, logsigma)

def isScalar(mu):
    for dim in mu.shape:
        if dim>1:
            return False
        return True

def kld_diag_gaussian_normal_original(mu, logsigma):
#     print(mu.shape)
#     print(type(mu))
    if len(mu.shape)<2:
        mu = mu.unsqueeze(1)
        logsigma = logsigma.unsqueeze(1)
#     if isScalar(mu):
#         print("scalar")
#         return 0.5 * (mu.pow(2) + torch.exp(2 * logsigma) - 2 * logsigma - 1)
    return (0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(1)).mean()

def kld_diag_gaussian_normal_original_no_mean(mu, logsigma):
#     print(mu.shape)
#     print(type(mu))
    if len(mu.shape)<2:
        mu = mu.unsqueeze(1)
        logsigma = logsigma.unsqueeze(1)
#     if isScalar(mu):
#         print("scalar")
#         return 0.5 * (mu.pow(2) + torch.exp(2 * logsigma) - 2 * logsigma - 1)
    return 0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(1)

def kld_diag_gaussian_normal_original_for_reg(mu, logsigma):
#     print(mu.shape)
#     print(type(mu))
    if len(mu.shape)<2:
        mu = mu.unsqueeze(1)
        logsigma = logsigma.unsqueeze(1)
#     if isScalar(mu):
#         print("scalar")
#         return 0.5 * (mu.pow(2) + torch.exp(2 * logsigma) - 2 * logsigma - 1)
    return (0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(1)).sum()

def kld_diag_gaussians_original(mu, logsigma, mu_prior, logsigma_prior):
    mu_prior = mu_prior*torch.ones_like(mu)
    logsigma_prior = logsigma_prior*torch.ones_like(logsigma)
    return  (0.5 * ((2*(logsigma-logsigma_prior)).exp() + (mu_prior-mu).pow(2) * (-2*logsigma_prior).exp() -1 + 2*(logsigma_prior-logsigma)).sum(1)).mean()

def kld_diag_gaussians_original_for_reg(mu, logsigma, mu_prior, logsigma_prior):
    mu_prior = mu_prior*torch.ones_like(mu)
    logsigma_prior = logsigma_prior*torch.ones_like(logsigma)
    return  (0.5 * ((2*(logsigma-logsigma_prior)).exp() + (mu_prior-mu).pow(2) * (-2*logsigma_prior).exp() -1 + 2*(logsigma_prior-logsigma)).sum(1)).sum()

