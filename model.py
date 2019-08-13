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

from options import Options

opt = Options().parse()

def sample_diag_gaussian_original(mu, logsigma, k_iws = 1): #reparametrization trick
    std = torch.exp(logsigma)
    # print("stdshape: {}".format(std.shape))
    # if opt.IWS:
    #     print("stdshape")
    #     print(std.shape)
        # std = std.unsqueeze(1)
        # std = std.repeat(1, k_iws, 1)
        # mu = mu.unsqueeze(1)
        # mu = mu.repeat(1, k_iws, 1)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEOriginal_paper(nn.Module):
    def __init__(self, isSparse, isSemiSupervised, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(VAEOriginal_paper, self).__init__()
        
        if opt.prune_encoder:
            self.encoder = EncoderPrune(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim)
        else:
            self.encoder = EncoderOriginal(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim)

        if opt.prune_decoder:
            self.decoder = SparseDecoderOriginal_paperPrune(latent_dim, sequence_length, alphabet_size, dec_h1_dim,
                                                       dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension,
                                                       nb_patterns, hasTemperature, hasDictionary)
        else:
            self.decoder = SparseDecoderOriginal_paper(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary)


    def forward(self, x):
        if opt.IWS_debug:
            pass
            # print("---Encoder---")
            # print("x")
            # print(x)
            # print(x.mean())
        mu, logsigma = self.encoder(x)


        if opt.IWS_debug:
            pass
            # print("mu")
            # print(mu)
            # print(mu.mean())
            # print("logsigma")
            # print(logsigma)
            # print(logsigma.mean())

        if opt.IWS:
            mu = mu.unsqueeze(1)
            mu = mu.repeat(1, opt.k_IWS, 1)
            logsigma = logsigma.unsqueeze(1)
            logsigma = logsigma.repeat(1, opt.k_IWS, 1)

        z = sample_diag_gaussian_original(mu, logsigma)
        px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(z, x)
        return mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l

    def reset_mask(self):
        if opt.prune_encoder:
            self.encoder.reset_mask()
        if opt.prune_decoder:
            self.decoder.reset_mask()

    def save_trained_weight(self):
        if opt.prune_encoder:
            self.encoder.save_trained_weight()
        if opt.prune_decoder:
            self.decoder.save_trained_weight()


    def random_reinit(self):
        if opt.prune_encoder:
            self.encoder.random_reinit()
        if opt.prune_decoder:
            self.decoder.random_reinit()


    def reinitializ(self):
        if opt.prune_encoder:
            self.encoder.reinitializ()
        if opt.prune_decoder:
            self.decoder.reinitializ()


    def load_trained_weight(self):
        if opt.prune_encoder:
            self.encoder.load_trained_weight()
        if opt.prune_decoder:
            self.decoder.load_trained_weight()


    def prune(self, rate, rate_output):
        if opt.prune_encoder:
            self.encoder.prune(rate, rate_output)
        if opt.prune_decoder:
            self.decoder.prune(rate, rate_output)


class DeterministicVAE(nn.Module):
    def __init__(self, isSparse, isSemiSupervised, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(DeterministicVAE, self).__init__()

        if opt.prune_encoder:
            self.encoder = EncoderPrune(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim)
        else:
            self.encoder = EncoderOriginal(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim)

        if opt.prune_decoder:
            self.decoder = DeterministicSparseDecoderPrune(latent_dim, sequence_length, alphabet_size, dec_h1_dim,
                                                       dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension,
                                                       nb_patterns, hasTemperature, hasDictionary)
        else:
            self.decoder = DeterministicSparseDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary)


    def forward(self, x):
        if opt.IWS_debug:
            pass
            # print("---Encoder---")
            # print("x")
            # print(x)
            # print(x.mean())
        mu, logsigma = self.encoder(x)


        if opt.IWS_debug:
            pass
            # print("mu")
            # print(mu)
            # print(mu.mean())
            # print("logsigma")
            # print(logsigma)
            # print(logsigma.mean())

        if opt.IWS:
            mu = mu.unsqueeze(1)
            mu = mu.repeat(1, opt.k_IWS, 1)
            logsigma = logsigma.unsqueeze(1)
            logsigma = logsigma.repeat(1, opt.k_IWS, 1)

        z = sample_diag_gaussian_original(mu, logsigma)
        px_z, logpx_z = self.decoder(z, x)
        return mu, logsigma, px_z, logpx_z, z

    def reset_mask(self):
        if opt.prune_encoder:
            self.encoder.reset_mask()
        if opt.prune_decoder:
            self.decoder.reset_mask()

    def save_trained_weight(self):
        if opt.prune_encoder:
            self.encoder.save_trained_weight()
        if opt.prune_decoder:
            self.decoder.save_trained_weight()


    def random_reinit(self):
        if opt.prune_encoder:
            self.encoder.random_reinit()
        if opt.prune_decoder:
            self.decoder.random_reinit()


    def reinitializ(self):
        if opt.prune_encoder:
            self.encoder.reinitializ()
        if opt.prune_decoder:
            self.decoder.reinitializ()


    def load_trained_weight(self):
        if opt.prune_encoder:
            self.encoder.load_trained_weight()
        if opt.prune_decoder:
            self.decoder.load_trained_weight()


    def prune(self, rate, rate_output):
        if opt.prune_encoder:
            self.encoder.prune(rate, rate_output)
        if opt.prune_decoder:
            self.decoder.prune(rate, rate_output)

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
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)  # "W_decode_"+str(layer_num)
        b1 = sample_diag_gaussian_original(self.mu_b1, self.logsigma_b1)
        W2 = sample_diag_gaussian_original(self.mu_W2, self.logsigma_W2)
        b2 = sample_diag_gaussian_original(self.mu_b2, self.logsigma_b2)
        W3 = sample_diag_gaussian_original(self.mu_W3, self.logsigma_W3)
        b3 = sample_diag_gaussian_original(self.mu_b3, self.logsigma_b3)
        S = sample_diag_gaussian_original(self.mu_S, self.logsigma_S)
        S = S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = W3.view(self.h2_dim * self.sequence_length, -1)
            C = sample_diag_gaussian_original(self.mu_C, self.logsigma_C)
            W_out = torch.mm(W3, C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, W1.t(), b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, W2.t(), b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), b3)
        l = sample_diag_gaussian_original(self.mu_l, self.logsigma_l)
        l = torch.log(1 + l.exp())
        h3 = h3 * l
        if opt.IWS:
            h3 = h3.view((-1, opt.k_IWS, self.sequence_length, self.alphabet_size))
        else:
            h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        px_z = F.softmax(h3, -1)
        if opt.IWS:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
            x = x.unsqueeze(1)
            x = x.repeat(1, opt.k_IWS, 1, 1)
            x = x.view(-1, opt.k_IWS, self.sequence_length, self.alphabet_size)
        else:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1)
        return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l

class DeterministicSparseDecoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases = 0, nb_features = 0):
        super(DeterministicSparseDecoder, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim))
        nn.init.xavier_normal_(self.W1)
        self.b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        nn.init.constant_(self.b1, 0.1)
        self.W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.xavier_normal_(self.W2)
        self.b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        nn.init.constant_(self.b2, 0.1)
        self.W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.xavier_normal_(self.W3)
        self.b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.b3, 0.1)
        self.S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.zeros_(self.S)
        self.C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.xavier_normal_(self.C)
        self.l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.l)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):

        x = x.float()

        S = self.S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = self.W3.view(self.h2_dim * self.sequence_length, -1)
            W_out = torch.mm(W3, self.C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, self.W1.t(), self.b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, self.W2.t(), self.b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), self.b3)
        l = torch.log(1 + self.l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        # print(x.shape)
        # print(h3.shape)
        px_zy = F.softmax(h3, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        # print(x.shape)
        # print(h3.shape)
        logpx_zy = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1) #one-hot
        return px_zy, logpx_zy

class DeterministicSparseDecoderPrune(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases = 0, nb_features = 0):
        super(DeterministicSparseDecoderPrune, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim))
        nn.init.xavier_normal_(self.W1)
        self.b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        nn.init.constant_(self.b1, 0.1)
        self.W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.xavier_normal_(self.W2)
        self.b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        nn.init.constant_(self.b2, 0.1)
        self.W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.xavier_normal_(self.W3)
        self.b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.b3, 0.1)
        self.S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.zeros_(self.S)
        self.C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.xavier_normal_(self.C)
        self.l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.l)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):

        x = x.float()

        S = self.S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = self.W3.view(self.h2_dim * self.sequence_length, -1)
            W_out = torch.mm(W3, self.C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, self.W1.t(), self.b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, self.W2.t(), self.b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), self.b3)
        l = torch.log(1 + self.l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        # print(x.shape)
        # print(h3.shape)
        px_zy = F.softmax(h3, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        # print(x.shape)
        # print(h3.shape)
        logpx_zy = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1) #one-hot
        return px_zy, logpx_zy


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

        h1 = F.relu(self.fc1(x))
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        mu = self.fc3_mu(h2)
        logsigma = self.fc3_logsigma(h2)
        return mu, logsigma

class LadderEncoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim):
        super(LadderEncoder, self).__init__()
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

        h1 = F.relu(self.fc1(x))
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        mu = self.fc3_mu(h2)
        logsigma = self.fc3_logsigma(h2)
        return mu, logsigma, x

class EncoderPrune(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim):
        super(EncoderPrune, self).__init__()

        self.masks = {}

        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.fc1 = nn.Linear(sequence_length*alphabet_size, h1_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc1_w_init = self.fc1.weight.data
        self.fc1_b_init = self.fc1.bias.data
        # print("fc1init")
        # print(self.fc1_w_init)
        self.masks['1'] = nn.Linear(sequence_length*alphabet_size, h1_dim)
        nn.init.ones_(self.masks['1'].weight)
        nn.init.ones_(self.masks['1'].bias)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc2_w_init = self.fc2.weight.data
        self.fc2_b_init = self.fc2.bias.data
        self.masks['2'] = nn.Linear(h1_dim, h2_dim)
        nn.init.ones_(self.masks['2'].weight)
        nn.init.ones_(self.masks['2'].bias)
        self.fc3_mu = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_mu.weight)
        nn.init.constant_(self.fc3_mu.bias, 0.1)
        self.fc3_mu_w_init = self.fc3_mu.weight.data
        self.fc3_mu_b_init = self.fc3_mu.bias.data
        self.masks['3_mu'] = nn.Linear(h2_dim, latent_dim)
        nn.init.ones_(self.masks['3_mu'].weight)
        nn.init.ones_(self.masks['3_mu'].bias)
        self.fc3_logsigma = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_logsigma.weight)
        nn.init.constant_(self.fc3_logsigma.bias, -5)
        self.fc3_ls_w_init = self.fc3_logsigma.weight.data
        self.fc3_ls_b_init = self.fc3_logsigma.bias.data
        self.masks['3_ls'] = nn.Linear(h2_dim, latent_dim)
        nn.init.ones_(self.masks['3_ls'].weight)
        nn.init.ones_(self.masks['3_ls'].bias)
        if opt.dropout>0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x):

        if x.shape[-1]!= self.sequence_length*self.alphabet_size:
            x = x.view(-1, self.sequence_length*self.alphabet_size)


        self.fc1.weight.data = torch.mul(self.fc1.weight, self.masks['1'].weight)
        self.fc2.weight.data = torch.mul(self.fc2.weight, self.masks['2'].weight)
        self.fc3_mu.weight.data = torch.mul(self.fc3_mu.weight, self.masks['3_mu'].weight)
        self.fc3_logsigma.weight.data = torch.mul(self.fc3_logsigma.weight, self.masks['3_ls'].weight)

        self.fc1.bias.data = torch.mul(self.fc1.bias, self.masks['1'].bias)
        self.fc2.bias.data = torch.mul(self.fc2.bias, self.masks['2'].bias)
        self.fc3_mu.bias.data = torch.mul(self.fc3_mu.bias, self.masks['3_mu'].bias)
        self.fc3_logsigma.bias.data = torch.mul(self.fc3_logsigma.bias, self.masks['3_ls'].bias)

        h1 = F.relu(self.fc1(x))
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        mu = self.fc3_mu(h2)
        logsigma = self.fc3_logsigma(h2)
        return mu, logsigma

    def reset_mask(self):
        nn.init.ones_(self.masks['1'].weight)
        nn.init.ones_(self.masks['1'].bias)
        nn.init.ones_(self.masks['2'].weight)
        nn.init.ones_(self.masks['2'].bias)
        nn.init.ones_(self.masks['3_ls'].weight)
        nn.init.ones_(self.masks['3_ls'].bias)
        nn.init.ones_(self.masks['3_mu'].weight)
        nn.init.ones_(self.masks['3_mu'].bias)

    def save_trained_weight(self):
        self.fc1_w_trained = self.fc1.weight.data
        self.fc2_w_trained = self.fc2.weight.data
        self.fc3_mu_w_trained = self.fc3_mu.weight.data
        self.fc3_logsigma_w_trained = self.fc3_logsigma.weight.data
        self.fc1_b_trained = self.fc1.bias.data
        self.fc2_b_trained = self.fc2.bias.data
        self.fc3_mu_b_trained = self.fc3_mu.bias.data
        self.fc3_logsigma_b_trained = self.fc3_logsigma.bias.data

    def random_reinit(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3_mu.weight)
        nn.init.xavier_normal_(self.fc3_logsigma.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3_mu.bias, 0.1)
        nn.init.constant_(self.fc3_logsigma.bias, 0.1)



    def reinitializ(self):
        # print("fc1wei")
        # print(self.fc1.weight.data)
        # print("fc1init")
        # print(self.fc1_w_init)
        self.fc1.weight.data = self.fc1_w_init
        self.fc2.weight.data = self.fc2_w_init
        self.fc3_mu.weight.data = self.fc3_mu_w_init
        self.fc3_logsigma.weight.data = self.fc3_ls_w_init
        self.fc1.bias.data = self.fc1_b_init
        self.fc2.bias.data = self.fc2_b_init
        self.fc3_mu.bias.data = self.fc3_mu_b_init
        self.fc3_logsigma.bias.data = self.fc3_ls_b_init

    def load_trained_weight(self):
        self.fc1.weight.data = self.fc1_w_trained
        self.fc2.weight.data = self.fc2_w_trained
        self.fc3_mu.weight.data = self.fc3_mu_w_trained
        self.fc3_logsigma.weight.data = self.fc3_logsigma_w_trained
        self.fc1.bias.data = self.fc1_b_trained
        self.fc2.bias.data = self.fc2_b_trained
        self.fc3_mu.bias.data = self.fc3_mu_b_trained
        self.fc3_logsigma.bias.data = self.fc3_logsigma_b_trained

    def prune(self, rate, rate_output):
        fc1_w_treshold = np.percentile(torch.abs(self.fc1.weight).detach().cpu().numpy(), rate)
        # print(fc1_w_treshold)
        fc2_w_treshold = np.percentile(torch.abs(self.fc2.weight).detach().cpu().numpy(), rate)
        fc3_mu_w_treshold = np.percentile(torch.abs(self.fc3_mu.weight).detach().cpu().numpy(), rate_output)
        fc3_logsigma_w_treshold = np.percentile(torch.abs(self.fc3_logsigma.weight).detach().cpu().numpy(), rate_output)

        self.masks['1'].weight.data = torch.mul(torch.gt(torch.abs(self.fc1.weight), fc1_w_treshold).float(), self.masks['1'].weight)
        self.masks['2'].weight.data = torch.mul(torch.gt(torch.abs(self.fc2.weight), fc2_w_treshold).float(), self.masks['2'].weight)
        self.masks['3_mu'].weight.data = torch.mul(torch.gt(torch.abs(self.fc3_mu.weight), fc3_mu_w_treshold).float(), self.masks['3_mu'].weight)
        self.masks['3_ls'].weight.data = torch.mul(torch.gt(torch.abs(self.fc3_logsigma.weight), fc3_logsigma_w_treshold).float(), self.masks['3_ls'].weight)

        fc1_b_treshold = np.percentile(torch.abs(self.fc1.bias).detach().cpu().numpy(), rate)
        fc2_b_treshold = np.percentile(torch.abs(self.fc2.bias).detach().cpu().numpy(), rate)
        fc3_mu_b_treshold = np.percentile(torch.abs(self.fc3_mu.bias).detach().cpu().numpy(), rate_output)
        fc3_logsigma_b_treshold = np.percentile(torch.abs(self.fc3_logsigma.bias).detach().cpu().numpy(), rate_output)

        self.masks['1'].bias.data = torch.mul(torch.gt(torch.abs(self.fc1.bias), fc1_b_treshold).float(), self.masks['1'].bias)
        self.masks['2'].bias.data = torch.mul(torch.gt(torch.abs(self.fc2.bias), fc2_b_treshold).float(), self.masks['2'].bias)
        self.masks['3_mu'].bias.data = torch.mul(torch.gt(torch.abs(self.fc3_mu.bias), fc3_mu_b_treshold).float(), self.masks['3_mu'].bias)
        self.masks['3_ls'].bias.data = torch.mul(torch.gt(torch.abs(self.fc3_logsigma.bias), fc3_logsigma_b_treshold).float(), self.masks['3_ls'].bias)

def initialize_mu(param):
    nn.init.xavier_normal_(param)

def initialize_logsigma(param):
    nn.init.constant_(param, -5)

def initialize_bias(param):
    nn.init.constant_(param, 0.1)

def initialize_mask(param):
    nn.init.ones_(param)

def initialize_S(param):
    nn.init.zeros_(param)


class SparseDecoderOriginal_paperPrune(nn.Module):  # sparsity ideas of deep generative model for mutation paper
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary):
        super(SparseDecoderOriginal_paperPrune, self).__init__()

        self.init_param = {}
        self.init_param_fct = {}
        self.masks = {}

        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.mu_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        # nn.init.xavier_normal_(self.mu_W1)
        # self.mu_W1_mask = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        # nn.init.ones_(self.mu_W1_mask)
        self.logsigma_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        # nn.init.constant_(self.logsigma_W1, -5)
        # self.logsigma_W1_mask = nn.Parameter(torch.Tensor(latent_dim, h1_dim))
        # nn.init.ones_(self.logsigma_W1_mask)
        self.mu_b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        # nn.init.constant_(self.mu_b1, 0.1)
        # self.mu_b1_mask = nn.Parameter(
        #     torch.Tensor(h1_dim))
        # nn.init.ones_(self.mu_b1_mask)
        self.logsigma_b1 = nn.Parameter(torch.Tensor(h1_dim))
        # nn.init.constant_(self.logsigma_b1, -5)
        # self.logsigma_b1_mask = nn.Parameter(torch.Tensor(h1_dim))
        # nn.init.ones_(self.logsigma_b1_mask)
        self.mu_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        # nn.init.xavier_normal_(self.mu_W2)
        # self.mu_W2_mask = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        # nn.init.ones_(self.mu_W2_mask)
        self.logsigma_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        # nn.init.constant_(self.logsigma_W2, -5)
        # self.logsigma_W2_mask = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        # nn.init.ones_(self.logsigma_W2_mask)
        self.mu_b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        # nn.init.constant_(self.mu_b2, 0.1)
        # self.mu_b2_mask = nn.Parameter(
        #     torch.Tensor(h2_dim))
        # nn.init.ones_(self.mu_b2_mask)
        self.logsigma_b2 = nn.Parameter(torch.Tensor(h2_dim))
        # nn.init.constant_(self.logsigma_b2, -5)
        # self.logsigma_b2_mask = nn.Parameter(torch.Tensor(h2_dim))
        # nn.init.ones_(self.logsigma_b2_mask)
        self.mu_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        # nn.init.xavier_normal_(self.mu_W3)
        # self.mu_W3_mask = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        # nn.init.ones_(self.mu_W3_mask)
        self.logsigma_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        # nn.init.constant_(self.logsigma_W3, -5)
        # self.logsigma_W3_mask = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        # nn.init.ones_(self.logsigma_W3_mask)
        self.mu_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        # nn.init.constant_(self.mu_b3, 0.1)
        # self.mu_b3_mask = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        # nn.init.ones_(self.mu_b3_mask)
        self.logsigma_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        # nn.init.constant_(self.logsigma_b3, -5)
        # self.logsigma_b3_mask = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        # nn.init.ones_(self.logsigma_b3_mask)
        self.mu_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        # nn.init.zeros_(self.mu_S)
        # self.mu_S_mask = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        # nn.init.ones_(self.mu_S_mask)
        self.logsigma_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        # nn.init.constant_(self.logsigma_S, -5)
        # self.logsigma_S_mask = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        # nn.init.ones_(self.logsigma_S_mask)
        self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        # nn.init.xavier_normal_(self.mu_C)
        # self.mu_C_mask = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        # nn.init.ones_(self.mu_C_mask)
        self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        # nn.init.constant_(self.logsigma_C, -5)
        # self.logsigma_C_mask = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        # nn.init.ones_(self.logsigma_C_mask)
        self.mu_l = nn.Parameter(torch.Tensor(1))
        # nn.init.ones_(self.mu_l)
        self.logsigma_l = nn.Parameter(torch.Tensor(1))
        # nn.init.constant_(self.logsigma_l, -5)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

        # print("..........")
        for name, param in self.named_parameters():
            if 'mask' in name or 'mu_l' in name:
                # print(name)
                # print("initialize_mask")
                self.init_param_fct[name] = initialize_mask
            elif 'mu_S' in name:
                # print(name)
                # print("initialize_S")
                self.init_param_fct[name] = initialize_S
            elif 'mu_b' in name:
                # print(name)
                # print("initialize_bias")
                self.init_param_fct[name] = initialize_bias
            elif 'logsigma' in name:
                # print(name)
                # print("initialize_ls")
                self.init_param_fct[name] = initialize_logsigma
            else:
                # print(name)
                # print("initialize_mu")
                self.init_param_fct[name] = initialize_mu

        for name, param in self.named_parameters():
            if 'mask' not in name:
                self.init_param_fct[name](param)

        for name, param in self.named_parameters():
            self.init_param[name] = param.data

        for name, param in self.named_parameters():
            if '_l' not in name:
                self.masks[name] = nn.Parameter(torch.Tensor(param.shape))
                initialize_mask(self.masks[name])
        # print("..........")

        self.seen = False





    def forward(self, z, x):
        for name, param in self.named_parameters():
            # if not self.seen:
            #     print(name)
            if '_l' not in name and 'mask' not in name:
                # if not self.seen:
                #     print(param.data)
                param.data = torch.mul(param, self.masks[name])
                # if not self.seen:
                #     print(param.data)
        self.seen = True
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)  # "W_decode_"+str(layer_num)
        b1 = sample_diag_gaussian_original(self.mu_b1, self.logsigma_b1)
        W2 = sample_diag_gaussian_original(self.mu_W2, self.logsigma_W2)
        b2 = sample_diag_gaussian_original(self.mu_b2, self.logsigma_b2)
        W3 = sample_diag_gaussian_original(self.mu_W3, self.logsigma_W3)
        b3 = sample_diag_gaussian_original(self.mu_b3, self.logsigma_b3)
        S = sample_diag_gaussian_original(self.mu_S, self.logsigma_S)
        S = S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = W3.view(self.h2_dim * self.sequence_length, -1)
            C = sample_diag_gaussian_original(self.mu_C, self.logsigma_C)
            W_out = torch.mm(W3, C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, W1.t(), b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, W2.t(), b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), b3)
        l = sample_diag_gaussian_original(self.mu_l, self.logsigma_l)
        l = torch.log(1 + l.exp())
        h3 = h3 * l
        if opt.IWS:
            h3 = h3.view((-1, opt.k_IWS, self.sequence_length, self.alphabet_size))
        else:
            h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        px_z = F.softmax(h3, -1)
        if opt.IWS:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
            x = x.unsqueeze(1)
            x = x.repeat(1, opt.k_IWS, 1, 1)
            x = x.view(-1, opt.k_IWS, self.sequence_length, self.alphabet_size)
        else:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1)
        return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l

    def reset_mask(self):
        for name, param in self.named_parameters():
            if '_l' not in name:
                initialize_mask(self.masks[name])

    def save_trained_weight(self):
        self.trained_param = {}
        for name, param in self.named_parameters():
            if 'mask' not in name:
                self.trained_param[name] = param.data

    def random_reinit(self):
        for name, param in self.named_parameters():
            if 'mask' not in name:
                self.init_param_fct[name](param)

    def reinitializ(self):
        for name, param in self.named_parameters():
            if 'mask' not in name:
                param.data = self.init_param[name]

    def load_trained_weight(self):
        for name, param in self.named_parameters():
            if 'mask' not in name:
                param.data = self.trained_param[name]

    def prune(self, rate, rate_output):
        self.param_treshold = {}
        for name, param in self.named_parameters():
            if 'mask' not in name and '_l' not in name:
                if '3' in name or 'C' in name or 'S' in name:
                    self.param_treshold[name] = np.percentile(torch.abs(param).detach().cpu().numpy(), rate_output)
                else:
                    self.param_treshold[name] = np.percentile(torch.abs(param).detach().cpu().numpy(), rate)

        for name, param in self.named_parameters():
            if 'mask' not in name and '_l' not in name:
                self.masks[name].data = torch.mul(torch.gt(torch.abs(param), self.param_treshold[name]).float(),
                                                   self.masks[name])
