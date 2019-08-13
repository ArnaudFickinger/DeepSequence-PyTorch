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


def loss_theano_june(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale = 1.0, mu_W1 = None, logsigma_W1 = None, mu_b1 = None, logsigma_b1 = None, mu_W2 = None, logsigma_W2 = None, mu_b2 = None, logsigma_b2 = None, mu_W3 = None, logsigma_W3 = None, mu_b3 = None, logsigma_b3 = None, mu_S = None, logsigma_S = None, mu_C = None, logsigma_C = None, mu_l = None, logsigma_l = None):

    return (logpx_z-warm_up_scale*kld_latent_theano(mu, logsigma)).mean()+warm_up_scale*sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff


def loss_theano(mu, logsigma, z, px_z, logpx_z, Neff = 1.0, warm_up_scale = 1.0, mu_W1 = None, logsigma_W1 = None, mu_b1 = None, logsigma_b1 = None, mu_W2 = None, logsigma_W2 = None, mu_b2 = None, logsigma_b2 = None, mu_W3 = None, logsigma_W3 = None, mu_b3 = None, logsigma_b3 = None, mu_S = None, logsigma_S = None, mu_C = None, logsigma_C = None, mu_l = None, logsigma_l = None):
    # if opt.IWS_debug:
    #     # print(".........")
    #     # print(logpx_z.shape)
    #     # print(mu.shape)
    #     # print(z.shape)
    #     # print("z")
    #     # print(z)
    #     # print(z.mean())
    #     # print("mu")
    #     # print(mu)
    #     # print(mu.mean())
    #     # print("logsigma")
    #     # print(logsigma)
    #     # print(logsigma.mean())
    #
    #     # loss = logpx_z - warm_up_scale*_kld(z, (mu, logsigma))
    #     # loss = warm_up_scale*_kld(z, (mu, logsigma))
    #     log_var = logsigma**2
    #     # loss = log_standard_gaussian(z)
    #
    #     # print(warm_up_scale)
    #
    #     # loss = _kld(z, (mu, logsigma))
    #     # loss = log_gaussian(z, mu, log_var) - log_standard_gaussian(z)
    #     print("z")
    #
    #     print(z)
    #     print(torch.sum(z, -1))
    #     print(z[0])
    #     loss = - warm_up_scale * log_standard_gaussian(z)
    #
    #     return loss.mean()
    #
    #
    #     elbo = logpx_z - warm_up_scale*_kld(z, (mu, logvar))
    #     # print(elbo.shape)
    #
    #     elbo = log_sum_exp(elbo, dim=-1, sum_op=torch.mean)
    #     # print("elbo")
    #     # print(elbo)
    #     # print(elbo.mean())
    #     # print(elbo.shape)
    #     # print(".........")
    #     return elbo.mean()+warm_up_scale*sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff
    if opt.IWS:
        elbo = logpx_z - warm_up_scale * _kld(z, (mu, logsigma))
        # print(elbo.shape)

        elbo = log_sum_exp(elbo, dim=-1, sum_op=torch.mean)
        return elbo.mean() + warm_up_scale * sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2,
                                                       mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
                                                       logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l) / Neff

    elif mu_W1 is None:
        return (logpx_z-warm_up_scale*kld_latent_theano(mu, logsigma)).mean()
    else:
        return (logpx_z-warm_up_scale*kld_latent_theano(mu, logsigma)).mean()+warm_up_scale*sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff

def kld_latent_theano(mu, log_sigma):
    KLD_latent = -0.5 * (1.0 + 2.0 * log_sigma - mu ** 2.0 - (2.0 * log_sigma).exp()).sum(1)
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

def _kld(z, q_param, p_param=None):
    (mu, log_var) = q_param

    # log_var = log_sigma ** 2

    # if opt.flow is not None:
    #     f_z, log_det_z = self.flow(z)
    #     qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
    #     z = f_z
    # else:
    qz = log_gaussian(z, mu, log_var)

    # print("qz")
    # print(qz)
    # print(qz.mean())

    if p_param is None:
        pz = log_standard_gaussian(z)
    else:
        (mu, log_var) = p_param
        pz = log_gaussian(z, mu, log_var)

    kl = qz - pz

    # print("pz")
    # print(pz)
    # print(pz.mean())

    return kl

def log_standard_gaussian(x):

    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):

    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    # return log_pdf
    return torch.sum(log_pdf, dim=-1)

def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    if opt.IWS_debug:
        pass
        # print("max")
        # print(max)
        # print(max.mean())
        # print(torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max

def total_loss_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff):
    return -sparse_ELBO_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)

def sparse_ELBO_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff):
    return ELBO_original(logpx_z, mu, logsigma, warm_up_scale) - warm_up_scale*sparse_weight_reg_paper(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff

def sparse_weight_reg_paper(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l):
    return kld_diag_gaussian_normal_original_for_reg(mu_W1, logsigma_W1)+kld_diag_gaussian_normal_original_for_reg(mu_b1, logsigma_b1)+kld_diag_gaussian_normal_original_for_reg(mu_W2, logsigma_W2)+kld_diag_gaussian_normal_original_for_reg(mu_b2, logsigma_b2)+kld_diag_gaussian_normal_original_for_reg(mu_W3, logsigma_W3)+kld_diag_gaussian_normal_original_for_reg(mu_b3, logsigma_b3)+kld_diag_gaussian_normal_original_for_reg(mu_C, logsigma_C)+kld_diag_gaussian_normal_original_for_reg(mu_l, logsigma_l)+kld_diag_gaussians_original_for_reg(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse)

def ELBO_original(logpx_z, mu, logsigma, warm_up_scale):
    return logpx_z.mean() - warm_up_scale*kld_diag_gaussian_normal_original(mu, logsigma)

def ELBO_no_mean(logpx_z, mu, logsigma, z, warm_up_scale):
    if opt.IWS:
        # print("................")
        # # print(".........")
        # print(logpx_z)
        # print(mu)
        # print(z)
        elbo = logpx_z - warm_up_scale * _kld(z, (mu, logsigma))
        # print(elbo)
        elbo = log_sum_exp(elbo, dim=-1, sum_op=torch.mean)
        # print(elbo)
        # print(".........")
        # print(elbo.shape)
        elbo = elbo.squeeze()
        # print(elbo.shape)
        return elbo
    else:
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

