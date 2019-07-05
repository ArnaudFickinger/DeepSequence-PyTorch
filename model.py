###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch
import math
# torch.manual_seed(42)
import pickle

from options import Options

opt = Options().parse()


class VAEOriginal_paper(nn.Module):
    def __init__(self, isSparse, isSemiSupervised, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(VAEOriginal_paper, self).__init__()

        self.encoder = EncoderOriginal(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim)

        self.decoder = SparseDecoderOriginal_paper(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary)


    def forward(self, x):
        # print("---Encoder---")
        mu, logsigma = self.encoder(x)
        z = sample_diag_gaussian_original(mu, logsigma)


        # z = torch.Tensor([[1.226298, - 0.50005774, - 0.05264941 , 0.66986938 ,- 0.71224884 ,- 1.15921493,
        #   0.05177535 , 1.1583736 ,  0.92446647 ,- 0.14730169  ,0.78172059 , 2.35628846,
        #   1.32076451  ,1.1621947, - 0.12593087 ,- 0.49765921,  0.55820479 ,- 0.84410041,
        #   - 1.94164654 , 1.19722414 , 0.51352968 , 0.23148123 , 0.56776408 ,- 0.36881278,
        #   - 0.590448  ,  1.50986844 , 0.68191917 ,- 1.4255378, - 0.08560662 , 1.08017904]])
        # print("z")
        # print(z)
        # print("---Decoder---")
        px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(z, x)
        return mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l

# # Parameter initialization
#         self.sigma_init = 0.01
#         self.logsig_init = -5
#         # Glorot initialization
#         create_weight = lambda dim_input, dim_output: self.prng.normal(0.0, \
#             np.sqrt(2.0 / (dim_input + dim_output)), \
#             (dim_input, dim_output)).astype(theano.config.floatX)
#         create_weight_zeros = lambda dim_input, dim_output: \
#             np.zeros((dim_input, dim_output)).astype(theano.config.floatX)
#         create_bias = lambda dim_output:  0.1 \
#             * np.ones(dim_output).astype(theano.config.floatX)
#
#         # Variational uncertainty
#         create_weight_logsig = lambda dim_input, dim_output: self.logsig_init \
#             * np.ones((dim_input, dim_output)).astype(theano.config.floatX)
#         create_bias_logsig = lambda dim_output: self.logsig_init \
#             * np.ones(dim_output).astype(theano.config.floatX)


class SparseDecoderOriginal_paper(nn.Module):  # sparsity ideas of deep generative model for mutation paper
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary):
        super(SparseDecoderOriginal_paper, self).__init__()

        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.mu_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        nn.init.xavier_normal_(self.mu_W1)
        self.logsigma_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        nn.init.constant_(self.logsigma_W1, -5)
        self.mu_b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        nn.init.constant_(self.mu_b1, 0.1)
        self.logsigma_b1 = nn.Parameter(torch.Tensor(h1_dim))
        nn.init.constant_(self.logsigma_b1, -5)
        self.mu_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.xavier_normal_(self.mu_W2)
        self.logsigma_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.constant_(self.logsigma_W2, -5)
        self.mu_b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        nn.init.constant_(self.mu_b2, 0.1)
        self.logsigma_b2 = nn.Parameter(torch.Tensor(h2_dim))
        nn.init.constant_(self.logsigma_b2, -5)
        self.mu_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.xavier_normal_(self.mu_W3)
        self.logsigma_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.constant_(self.logsigma_W3, -5)
        self.mu_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.mu_b3, 0.1)
        self.logsigma_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.logsigma_b3, -5)
        self.mu_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.zeros_(self.mu_S)
        self.logsigma_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.constant_(self.logsigma_S, -5)
        self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.xavier_normal_(self.mu_C)
        self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.constant_(self.logsigma_C, -5)
        self.mu_l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.mu_l)
        self.logsigma_l = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.logsigma_l, -5)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):
        # print("w1")
        # print(self.mu_W1)
        # print(self.logsigma_W1)
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)  # "W_decode_"+str(layer_num)
        # W1 = torch.Tensor(theano_dic['w1'])
        # print(W1)
        lpW1 = log_gaussian_logsigma(W1, torch.zeros_like(W1), torch.zeros_like(W1)).sum()
        lqW1 = log_gaussian_logsigma(W1, self.mu_W1, self.logsigma_W1).sum()
        b1 = sample_diag_gaussian_original(self.mu_b1, self.logsigma_b1)
        # b1 = torch.Tensor(theano_dic['b1'])
        lpb1 = log_gaussian_logsigma(b1, torch.zeros_like(b1), torch.zeros_like(b1)).sum()
        lqb1 = log_gaussian_logsigma(b1, self.mu_b1, self.logsigma_b1).sum()
        W2 = sample_diag_gaussian_original(self.mu_W2, self.logsigma_W2)
        # W2 = torch.Tensor(theano_dic['w2'])
        lpW2 = log_gaussian_logsigma(W2, torch.zeros_like(W2), torch.zeros_like(W2)).sum()
        lqW2 = log_gaussian_logsigma(W2, self.mu_W2, self.logsigma_W2).sum()
        b2 = sample_diag_gaussian_original(self.mu_b2, self.logsigma_b2)
        # b2 = torch.Tensor(theano_dic['b2'])
        lpb2 = log_gaussian_logsigma(W2, torch.zeros_like(b2), torch.zeros_like(b2)).sum()
        lqb2 = log_gaussian_logsigma(W2, self.mu_b2, self.logsigma_b2).sum()
        W3 = sample_diag_gaussian_original(self.mu_W3, self.logsigma_W3)
        # W3 = torch.Tensor(theano_dic['w3'])
        lpW3 = log_gaussian_logsigma(W3, torch.zeros_like(W3), torch.zeros_like(W3)).sum()
        lqW3 = log_gaussian_logsigma(W3, self.mu_W3, self.logsigma_W3).sum()
        b3 = sample_diag_gaussian_original(self.mu_b3, self.logsigma_b3)
        # b3 = torch.Tensor(theano_dic['b3'])
        lpb3 = log_gaussian_logsigma(b3, torch.zeros_like(b3), torch.zeros_like(b3)).sum()
        lqb3 = log_gaussian_logsigma(b3, self.mu_b3, self.logsigma_b3).sum()
        S = sample_diag_gaussian_original(self.mu_S, self.logsigma_S)
        # S = torch.Tensor(theano_dic['s'])
        # print("sbefore")
        # print(S)
        lpS = log_gaussian_logsigma(S, self.mu_sparse * torch.ones_like(S),
                                    self.logsigma_sparse * torch.ones_like(S)).sum()
        lqS = log_gaussian_logsigma(S, self.mu_S, self.logsigma_S).sum()
        # print(S.shape)
        S = S.repeat(self.nb_patterns, 1)  # W-scale
        # print(S.shape)
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = W3.view(self.h2_dim * self.sequence_length, -1)
            C = sample_diag_gaussian_original(self.mu_C, self.logsigma_C)
            # C = torch.Tensor(theano_dic['c'])
            lpC = log_gaussian_logsigma(C, torch.zeros_like(C), torch.zeros_like(C)).sum()
            lqC = log_gaussian_logsigma(C, self.mu_C, self.logsigma_C).sum()
            # C = C.repeat(self.sequence_length, 1)
            W_out = torch.mm(W3, C)
            # print("inter1")
            # print(W_out)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            # print("inter2")
            # print(W_out)
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, W1.t(), b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        # print("h1")
        # print(h1)
        h2 = F.sigmoid(F.linear(h1, W2.t(), b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        # print("h2")
        # print(h2)
        # print(h2.shape)
        # h2 = h2.repeat(self.sequence_length)
        # print(h2.shape)
        # lcw = lcw.view((self.h2_dim, self.sequence_length, self.alphabet_size))
        # S = torch.unsqueeze(S, 2)
        # print("S")
        # print(S)
        # print(S.shape)
        # print(lcw.shape)
        # lcws = lcw * S
        # lcws = lcws.view(-1, self.h2_dim)
        h3 = F.linear(h2, W_out.t(), b3)
        # print("inter3")
        # print(h3)
        l = sample_diag_gaussian_original(self.mu_l, self.logsigma_l)
        # l = torch.Tensor(theano_dic['l'])
        lpl = log_gaussian_logsigma(l, torch.zeros_like(l), torch.zeros_like(l)).sum()
        lql = log_gaussian_logsigma(l, self.mu_l, self.logsigma_l).sum()
        l = torch.log(1 + l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        # print("h3")
        # print(h3)
        px_z = F.softmax(h3, 2)
        # print(F.log_softmax(h3,2).shape)
        # print(x.shape)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1)
        if True:
            return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l


class EncoderOriginal(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim):
        super(EncoderOriginal, self).__init__()
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.fc1 = nn.Linear(sequence_length*alphabet_size, h1_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3_mu = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_mu.weight)
        nn.init.constant_(self.fc3_mu.bias, 0.1)
        self.fc3_logsigma = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_logsigma.weight)
        nn.init.constant_(self.fc3_logsigma.bias, -5)
        if opt.dropout>0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x):

        if x.shape[-1]!= self.sequence_length*self.alphabet_size:
            x = x.view(-1, self.sequence_length*self.alphabet_size)

        # print("x")
        # print(x)
        # print("self.fc1.weight")
        # print(self.fc1.weight)
        # print("self.fc1.bias")
        # print(self.fc1.bias)
        h1 = F.relu(self.fc1(x))
        if opt.dropout>0:
            h1 = self.dropout(h1)
        # print("h1")
        # print(h1)
        # print("self.fc2.weight")
        # print(self.fc2.weight)
        # print("self.fc2.bias")
        # print(self.fc2.bias)
        h2 = F.relu(self.fc2(h1))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        # print("h2")
        # print(h2)
        # print("self.fc3_mu.weight")
        # print(self.fc3_mu.weight)
        # print("self.fc3_mu.bias")
        # print(self.fc3_mu.bias)
        mu = self.fc3_mu(h2)
        # print("self.fc3_logsigma.weight")
        # print(self.fc3_logsigma.weight)
        # print("self.fc3_logsigma.bias")
        # print(self.fc3_logsigma.bias)
        logsigma = self.fc3_logsigma(h2)
        # print("mu")
        # print(mu)
        # print("logsigma")
        # print(logsigma)
        return mu, logsigma


