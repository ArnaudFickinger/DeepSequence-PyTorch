###
'''
April 2019
Code by: Arnaud Fickinger
'''
###
import copy

from torch.utils.data import DataLoader
from dataset import *
from loss import *
from model import *
import cv2 as cv

import pandas as pd

from scipy.stats import spearmanr

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from options import Options

opt = Options().parse()

if opt.IWS:
    str_vae = "vae_iws"
else:
    str_vae = "vae"

def main():
    if opt.custom_dataset:
        train_custom_dataset(opt.dataset_path, opt.dataset_id, opt.theta, opt.mutation_path, opt.phenotype_name, opt.mutation_iterations)
    else:
        if opt.compare_neff:
            compare_neff()
        if opt.stoch_det:
            dataset_prot = ["DLG4_RAT", "BLAT_ECOLX", "PABP_YEAST"]
            for data in dataset_prot:
                compare_deterministic_stochastic_weights(dataset = data)
        elif opt.prune:
            dataset_prot = ["DLG4_RAT", "BLAT_ECOLX"]
            pruning = [60]
            strate = ['0', '60']
            for dataset in dataset_prot:
                train_pruned(pruning, strate, dataset=dataset)
        elif opt.is_train:
            print("train")
            dataset_prot = ["DLG4_RAT"]
            dataset_virus = ["BG505"]
            theta_virus = [0.01, 0.2, 0]
            theta_prot = [0.2]
            for prot in dataset_prot:
                print(prot)
                for theta in theta_prot:
                    print(theta)
                    train_main(prot, theta)
            # for vir in dataset_virus:
            #     print(vir)
            #     for theta in theta_virus:
            #         print(theta)
            #         train_main(vir, theta)
        else:
            print("test")
            dataset_prot = ["DLG4_RAT"]
            dataset_virus = ["BG505"]
            theta_virus = [0.01, 0.2, 0]
            theta_prot = [0.2]
            for prot in dataset_prot:
                print(prot)
                for theta in theta_prot:
                    print(theta)
                    test(prot, theta)
            # for vir in dataset_virus:
            #     print(vir)
            #     for theta in theta_virus:
            #         print(theta)
            #         test(vir, theta)


if opt.test_algo:
    test_prn = 2
    epoch_prn = 3
    iter_prn = 2
    spr_prn = 1
    
else:
    test_prn = 5
    epoch_prn = 100
    iter_prn = 100
    spr_prn = 10


def train_custom_dataset(dataset, dataset_str, theta, mutation, phenotype_name, iterations):
    dataset_helper = DataHelper(dataset, theta, custom_dataset=True)
    seqlen = dataset_helper.seqlen
    datasize = dataset_helper.datasize

    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"
    else:
        theta_str = "cst"

    weights = dataset_helper.weights
    # print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datasize)  # change opt batch size
    train_dataset = Dataset(dataset_helper)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

    # print("datasetloader shape")
    # print(train_dataset.data.shape)

    model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                              opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    update_num = 0

    LB_list = []
    loss_params_list = []
    KLD_latent_list = []
    reconstruct_list = []

    # spearmans = []
    # pvals = []
    # epochs_spr = []

    titles = ["loss", "KLD_weights", "KLD_latent", "logpx_z"]

    start = 0

    # spr_virus = [1000, 5000, 10000, 15000]

    # if opt.continue_training > 0:
    #     model.load_state_dict(torch.load(opt.saving_path + "epoch_" + str(opt.continue_training)))
    #     start = opt.continue_training + 1

    if opt.neff == 0:
        Neff = dataset_helper.Neff
    else:
        Neff = opt.Neff

    model.train()

    if opt.test_algo:
        epochs = 2
    else:
        epochs = opt.epochs

    if mutation!="" and opt.plot_spearman:
        print(mutation)
        print(dataset_helper)
        dataset_helper.mutation_file_to_onehot(mutation)
        mutant_list, expr_values_ref_list = preprocess_mutation_file(mutation, phenotype_name)
        sprs = []
        spr_epochs_plot = []

    for e in range(start, epochs+1):
        # print("------------------------------")
        print(e)
        # print("len train loader")
        # print(len(train_dataset_loader))

        for i, batch in enumerate(train_dataset_loader):
            update_num += 1
            if opt.IWS_debug and update_num > 30:
                return
            # print("batch:")
            # print(update_num)
            # print(batch.shape)
            warm_up_scale = 1.0
            if update_num < opt.warm_up:
                warm_up_scale = update_num / opt.warm_up
            batch = batch.float().to(device)
            # if opt.IWS:
            #     batch = batch.repeat(opt.k_IWS, 1, 1)
            optimizer.zero_grad()
            mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                batch)
            loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale, mu_W1, logsigma_W1, mu_b1,
                                logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3,
                                logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)
            # loss = total_loss_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)

            # print("loss")
            # print(loss)

            LB_list.append(loss.item())
            # print(loss.item())
            loss_params_list.append(
                (warm_up_scale * sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2,
                                               logsigma_b2,
                                               mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C,
                                               logsigma_C,
                                               mu_l, logsigma_l) / Neff).item())
            KLD_latent_list.append((warm_up_scale * kld_latent_theano(mu, logsigma).mean()).item())
            reconstruct_list.append(logpx_z.mean().item())
            loss.backward()
            if opt.IWS_debug:
                print("..............")
                for name, param in model.encoder.named_parameters():
                    if 'fc1' in name:
                        print(name)
                        print(param.grad.data.mean())
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-100, 100)
                for name, param in model.encoder.named_parameters():
                    if 'fc1' in name:
                        print(name)
                        print(param.grad.data.mean())
                print("...............")
                # param.grad.data.clamp_(-1, 1)
            optimizer.step()

        if mutation!="" and opt.plot_spearman:
            if e % opt.spearman_every == 0:
                model.eval()
                with torch.no_grad():
                    mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model,
                                                                                               N_pred_iterations=iterations)
                    spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_list,
                                                   expr_values_ref_list)
                    sprs.append(spearman_r)
                    spr_epochs_plot.append(e + 1)
                model.train()

        # if (e + 1) in spr_virus:
        #     print(e)
        #     model.eval()
        #     with torch.no_grad():
        #         custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
        #                 mutation_file, model, N_pred_iterations=500)
        #
        #         spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
        #                                            mutation_file, phenotype_name)
        #
        #         spearmans.append(spr)
        #         pvals.append(pval)
        #         epochs_spr.append(e)

    # model.eval()
    # with torch.no_grad():
    #     custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
    #     mutation_file, model, N_pred_iterations=500,
    #     filename_prefix="pred_{}_{}_{}_{}_{}_epoch_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size,
    #                                               -int(math.log10(opt.lr)), opt.epochs))
    #
    #     spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
    #                                    mutation_file, phenotype_name)

    # spearmans.append(spr)
    # pvals.append(pval)
    # epochs_spr.append(opt.epochs)

    if mutation!="" and opt.plot_spearman:
        plt.clf()
        plt.plot(spr_epochs_plot, sprs)
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
        plt.savefig(opt.plots_path +
                    "spr_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset_str, theta_str, opt.latent_dim, opt.batch_size,
                                                      -int(math.log10(opt.lr)), opt.epochs))
        plt.close('all')


    torch.save(model.state_dict(),
                   opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(str_vae, dataset_str, theta_str, opt.latent_dim,
                                                                                opt.batch_size, -int(math.log10(opt.lr)),
                                                                                int(opt.neff), opt.epochs))

    # plt.clf()
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_spr, spearmans)
    # plt.title("Spearman")
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_spr, pvals)
    # plt.title("p-val")
    # plt.suptitle(
    #     "ds: {}, t: {}, ld: {}, bs: {}, lr: {}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, opt.lr))
    # plt.savefig(
    #     "spr_{}_{}_{}_{}_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr))))
    # plt.close('all')
    plots = [LB_list, loss_params_list, KLD_latent_list, reconstruct_list]
    plt.clf()
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(np.arange(len(plots[i])), plots[i])
        plt.title(titles[i])
    plt.suptitle(
        "ds: {}, t: {}, ld: {}, bs: {}, lr: {}, e: {}".format(dataset_str, theta_str, opt.latent_dim, opt.batch_size,
                                                              opt.lr, opt.epochs))
    plt.savefig(opt.plots_path+
        "plt_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset_str, theta_str, opt.latent_dim, opt.batch_size,
                                          -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')


def compare_neff():
    pass

def train_pruned(rates, rates_str, nb_test = test_prn, dataset = "DLG4_RAT", theta = 0.2, epochs_ = epoch_prn, iterations = iter_prn, spr_all = spr_prn):

    best_strs = []
    best_models = []
    best_initialization = None
    best_init_spr = 0
    best_init_test = 0

    masks = {rate:[] for rate in rates}

    spearman_dic = {rate:[] for rate in rates_str}


    # overall_acc_0_init = []
    # overall_acc_4_init = []
    # overall_acc_20_init = []
    # overall_acc_60_init = []

    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen
    datasize = dataset_helper.datasize

    str_prn = ""

    if opt.prune_encoder:
        str_prn += "enc"
    if opt.prune_decoder:
        str_prn += "dec"

    if opt.test_algo:
        mutation_file = "./mutations/DLG4.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505small.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520small.csv"
        phenotype_name = "fitness"

    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    weights = dataset_helper.weights
    # print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datasize)  # change opt batch size
    train_dataset = Dataset(dataset_helper)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

    # print("datasetloader shape")
    # print(train_dataset.data.shape)

    dataset_helper.mutation_file_to_onehot(mutation_file)
    mutant_measures_name_list, measurement_list = preprocess_mutation_file(mutation_file, phenotype_name)




    for test in range(nb_test):
        print("test {}".format(test))

        best_spr = 0
        best_model = None
        best_str = "no.pth"



        model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                                  opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                                  opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                                  opt.has_temperature, opt.has_dictionary).to(device)
        optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)


        initialization_state = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict())}

        # checkpoint = {'model': Classifier(),
        #               'state_dict': model.state_dict(),
        #               'optimizer': optimizer.state_dict()}


        update_num = 0

        # LB_list = []
        # loss_params_list = []
        # KLD_latent_list = []
        # reconstruct_list = []

        # spearmans = []
        # pvals = []
        # epochs_spr = []

        # titles = ["loss", "KLD_weights", "KLD_latent", "logpx_z"]

        start = 0

        # spr_virus = [1000, 5000, 10000, 15000]

        # if opt.continue_training > 0:
        #     model.load_state_dict(torch.load(opt.saving_path + "epoch_" + str(opt.continue_training)))
        #     start = opt.continue_training + 1

        if opt.neff == 0:
            Neff = dataset_helper.Neff
        else:
            Neff = opt.Neff

        model.train()

        # if opt.test_algo:
        #     epochs = 2
        # else:
        #     epochs = opt.epochs


        sprs_0 = []
        # all_sprs = []
        spr_epochs = []

        # if opt.plot_spearman:
        #     mutant_sequences_one_hot, mutant_sequences_descriptor = dataset_helper.mutation_file_to_onehot(
        #         mutation_file)
        #     mutant_list, expr_values_ref_list = preprocess_mutation_file(mutation_file, phenotype_name)
        #     sprs = []
        #     spr_epochs_plot = []


        # print(model.decoder.mu_W1.data)
        # print(model.decoder.masks['mu_W1'].data)
        seen = False
        for e in range(start, epochs_+1):
            # print("------------------------------")
            # print(e)
            # print("len train loader")
            # print(len(train_dataset_loader))

            for i, batch in enumerate(train_dataset_loader):
                update_num += 1
                warm_up_scale = 1.0
                if update_num < opt.warm_up:
                    warm_up_scale = update_num / opt.warm_up
                batch = batch.float().to(device)
                # if opt.IWS:
                #     batch = batch.repeat(opt.k_IWS, 1, 1)
                optimizer.zero_grad()
                mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                    batch)
                loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale,  mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,
                                    logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
                                    logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)

                loss.backward()
                optimizer.step()
                # print(model.decoder.mu_W1.data)


            if e % spr_all == 0:
                model.eval()
                with torch.no_grad():
                    mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model, N_pred_iterations=iterations)
                    spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_measures_name_list, measurement_list)
                    sprs_0.append(spearman_r)
                    spr_epochs.append(e + 1)
                    if spearman_r > best_spr:
                        best_spr = spearman_r
                        best_model = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()), 'spearman': spearman_r}
                        best_str = "ensemble_{}_{}_{}_{}_{}_{}_{}_epoch_{}_test_{}.pth".format(str_vae,
                                                                                                               dataset,
                                                                                                               theta_str,
                                                                                                               opt.latent_dim,
                                                                                                               opt.batch_size,
                                                                                                               -int(
                                                                                                                   math.log10(
                                                                                                                       opt.lr)),
                                                                                                               int(
                                                                                                                   opt.neff),
                                                                                                               e,
                                                                                                               test)
                    if spearman_r > best_init_spr:
                        best_init_spr = spearman_r
                        best_initialization = copy.deepcopy(initialization_state)
                        best_initialization['spearman'] = spearman_r
                        best_init_test = test

                model.train()


        spearman_dic[rates_str[0]].append(sprs_0)

        model.save_trained_weight()
        # print(model.decoder.mu_W1.data)
        # print(model.decoder.masks['mu_W1'].data)

        print("smart")

        for rate, rate_str in zip(rates,rates_str[1:]):

            print("rate: {}".format(rate))
            tmpr_str = []
            output_rate = int(rate / 2)

            optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name],
                                         lr=opt.lr)  # try with that

            model.load_trained_weight()
            # print(model.decoder.masks['mu_W1'].data)

            model.reset_mask()
            # print(model.decoder.masks['mu_W1'].data)

            # print(model.encoder.mask1.weight.data.sum().item())
            model.prune(rate, output_rate)
            # print(model.encoder.mask1.weight.data.sum().item())

            masks[rate].append(np.copy(model.encoder.masks['1'].weight.data.detach().cpu().numpy()))
            # print(np.sum(masks[rate][0]))
            # if not seen:
            #     print(model.decoder.mu_W1.data)
            #     print(model.decoder.masks['mu_W1'].data)
            model.reinitializ()
            # if not seen:
            #     print(model.decoder.mu_W1.data)
            #     print(model.decoder.masks['mu_W1'].data)
            #     seen = True


            for e in range(start, epochs_+1):
                # print("------------------------------")
                # print(e)
                # print("len train loader")
                # print(len(train_dataset_loader))

                for i, batch in enumerate(train_dataset_loader):
                    update_num += 1
                    warm_up_scale = 1.0
                    if update_num < opt.warm_up:
                        warm_up_scale = update_num / opt.warm_up
                    batch = batch.float().to(device)
                    # if opt.IWS:
                    #     batch = batch.repeat(opt.k_IWS, 1, 1)
                    optimizer.zero_grad()
                    mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                        batch)
                    loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,
                                        logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
                                        logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)

                    loss.backward()
                    optimizer.step()

                if e % spr_all == 0:
                    model.eval()
                    with torch.no_grad():
                        mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model,
                                                                                                   N_pred_iterations=iterations)
                        spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos,
                                                       mutant_measures_name_list, measurement_list)
                        tmpr_str.append(spearman_r)
                        if spearman_r > best_spr:
                            best_spr = spearman_r
                            best_model = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()), 'spearman': spearman_r}
                            best_str = "ensemble_{}_{}_{}_{}_{}_{}_{}_epoch_{}_pruned_{}_{}_smart_test_{}.pth".format(str_vae, dataset, theta_str,
                                                                                        opt.latent_dim, opt.batch_size,
                                                                                        -int(math.log10(opt.lr)),
                                                                                        int(opt.neff),
                                                                                        e, str_prn, rate, test)


                    model.train()

            spearman_dic[rate_str].append(tmpr_str)

            if opt.save_every_rates:
                torch.save(model.state_dict(),
                           opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}_pruned_{}_{}_test_{}.pth".format(str_vae, dataset, theta_str,
                                                                                        opt.latent_dim, opt.batch_size,
                                                                                        -int(math.log10(opt.lr)),
                                                                                        int(opt.neff),
                                                                                        opt.epochs, str_prn, rate, test))

        # print(all_sprs)
        # print(spr_epochs)
        # plt.clf()
        # for acc, lbl in zip(all_sprs, rates_str):
        #     plt.plot(spr_epochs, acc, label=lbl)
        # plt.legend(title="Pruning (%):")
        # plt.xlabel("Epoch")
        # plt.ylabel("Spearman")
        # plt.savefig("lotteryticket_{}_smart_init_{}".format(str_prn, test))
        # plt.close()

        plt.clf()
        for lbl in rates_str:
            plt.plot(spr_epochs, spearman_dic[lbl][-1], label=lbl)
        plt.legend(title="Pruning (%):")
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
        plt.savefig(opt.plots_path + "lotteryticket_{}_{}_smart_init_{}".format(str_prn, dataset, test))
        plt.close()

        # overall_acc_0_init.append(all_sprs[0])
        # overall_acc_4_init.append(all_sprs[1])
        # overall_acc_20_init.append(all_sprs[2])
        # overall_acc_60_init.append(all_sprs[3])

        # all_sprs = []
        # all_sprs.append(sprs_0)
        #
        # print("rand")
        #
        # for rate in rates:
        #
        #     print("rate: {}".format(rate))
        #     tmpr_str = []
        #     output_rate = int(rate / 2)
        #
        #     optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name],
        #                                  lr=opt.lr)  # try with that
        #
        #
        #
        #     model.load_trained_weight()
        #     model.reset_mask()
        #     model.prune(rate, output_rate)
        #     model.random_reinit()
        #
        #
        #     for e in range(start, epochs_+1):
        #         # print("------------------------------")
        #         # print(e)
        #         # print("len train loader")
        #         # print(len(train_dataset_loader))
        #
        #         for i, batch in enumerate(train_dataset_loader):
        #             update_num += 1
        #             warm_up_scale = 1.0
        #             if update_num < opt.warm_up:
        #                 warm_up_scale = update_num / opt.warm_up
        #             batch = batch.float().to(device)
        #             # if opt.IWS:
        #             #     batch = batch.repeat(opt.k_IWS, 1, 1)
        #             optimizer.zero_grad()
        #             mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
        #                 batch)
        #             loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,
        #                                 logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
        #                                 logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)
        #
        #             loss.backward()
        #             optimizer.step()
        #
        #         if e % opt.spearman_every == 0:
        #             model.eval()
        #             with torch.no_grad():
        #                 mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model,
        #                                                                                            N_pred_iterations=iterations)
        #                 spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos,
        #                                                mutant_measures_name_list, measurement_list)
        #                 tmpr_str.append(spearman_r)
        #                 if spearman_r > best_spr:
        #                     best_spr = spearman_r
        #                     best_model = model.state_dict()
        #                     best_str = "ensemble_{}_{}_{}_{}_{}_{}_{}_epoch_{}_pruned_{}_{}_rand_test_{}.pth".format(str_vae, dataset, theta_str,
        #                                                                                 opt.latent_dim, opt.batch_size,
        #                                                                                 -int(math.log10(opt.lr)),
        #                                                                                 int(opt.neff),
        #                                                                                 opt.epochs, str_prn, rate, test)
        #             model.train()
        #
        #     all_sprs.append(tmpr_str)
        #
        # plt.clf()
        # for acc, lbl in zip(all_sprs, rates_str):
        #     plt.plot(spr_epochs, acc, label=lbl)
        # plt.legend(title="Pruning (%):")
        # plt.xlabel("Epoch")
        # plt.ylabel("Spearman")
        # plt.savefig("lotteryticket_{}_rand_init_{}".format(str_prn, test))
        # plt.close()
        #
        # overall_acc_0_rand.append(all_sprs[0])
        # overall_acc_4_rand.append(all_sprs[1])
        # overall_acc_20_rand.append(all_sprs[2])
        # overall_acc_60_rand.append(all_sprs[3])


        # print("b:{}".format(best_str))
        torch.save(best_model,
                   opt.saving_path + best_str)
        best_strs.append(best_str)
        best_models.append(best_model)
        print(best_str)
        print(best_spr)

    for rate in rates:
        # print(rate)
        # print(masks[rate][0])
        # print(np.sum(masks[rate][0]))
        cv.imwrite("mask_{}_{}_test_1.png".format(rate, dataset), masks[rate][0]*255)
        cv.imwrite("mask_{}_{}_mean.png".format(rate, dataset), np.mean(masks[rate], 0)*255)

    cv.imwrite("mask_{}_mean.png".format(dataset), np.mean([np.mean(masks[rate], 0) for rate in rates], 0)*255)

    torch.save(best_initialization, opt.saving_path + "initialization.pth")
    for rate in rates_str:
        np.save(opt.arrays_path + "spearman_pruning_{}_{}".format(rate, dataset), spearman_dic[rate][best_init_test])

    np.save(opt.arrays_path + "epochs_plot_{}".format(dataset), spr_epochs)



    # acc_0_init_np = np.array(overall_acc_0_init)
    # acc_4_init_np = np.array(overall_acc_4_init)
    # acc_20_init_np = np.array(overall_acc_20_init)
    # acc_60_init_np = np.array(overall_acc_60_init)

    # acc_0_rand_np = np.array(overall_acc_0_rand)
    # acc_4_rand_np = np.array(overall_acc_4_rand)
    # acc_20_rand_np = np.array(overall_acc_20_rand)
    # acc_60_rand_np = np.array(overall_acc_60_rand)

    spearman_mean = {rate:np.mean(spearman_dic[rate], 0) for rate in rates_str}

    # acc_0_init_mean = np.mean(acc_0_init_np, axis=0)
    # acc_4_init_mean = np.mean(acc_4_init_np, axis=0)
    # acc_20_init_mean = np.mean(acc_20_init_np, axis=0)
    # acc_60_init_mean = np.mean(acc_60_init_np, axis=0)

    # acc_0_rand_mean = np.mean(acc_0_rand_np, axis=0)
    # acc_4_rand_mean = np.mean(acc_4_rand_np, axis=0)
    # acc_20_rand_mean = np.mean(acc_20_rand_np, axis=0)
    # acc_60_rand_mean = np.mean(acc_60_rand_np, axis=0)

    # all_acc_mean = [acc_0_init_mean, acc_4_init_mean, acc_20_init_mean, acc_60_init_mean]

    # plt.clf()
    # for acc, lbl in zip(all_acc_mean, rates_str):
    #     plt.plot(spr_epochs, acc, label=lbl)
    # plt.legend(title="Pruning (%):")
    # plt.xlabel("Epoch")
    # plt.ylabel("Spearman")
    # plt.savefig("lotteryticket_{}_smart_init_mean".format(str_prn))
    # plt.close()

    plt.clf()
    for lbl in rates_str:
        plt.plot(spr_epochs, spearman_mean[lbl], label=lbl)
    plt.legend(title="Pruning (%):")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman")
    plt.savefig(opt.plots_path + "lotteryticket_{}_{}_smart_init_mean".format(str_prn, dataset))
    plt.close()

    # all_acc_mean = [acc_0_rand_mean, acc_4_rand_mean, acc_20_rand_mean, acc_60_rand_mean]
    #
    # plt.clf()
    # for acc, lbl in zip(all_acc_mean, rates_str):
    #     plt.plot(spr_epochs, acc, label=lbl)
    # plt.legend(title="Pruning (%):")
    # plt.xlabel("Epoch")
    # plt.ylabel("Spearman")
    # plt.savefig("lotteryticket_{}_rand_init_mean".format(str_prn))
    # plt.close()

    model.eval()
    with torch.no_grad():
        mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot_ensemble(model, best_strs, N_pred_iterations=iterations)
        spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_measures_name_list,
                                                measurement_list)
        print("Ensemble spr:")
        print(spearman_r)

def compare_deterministic_stochastic_weights(nb_test = test_prn, dataset = "DLG4_RAT", theta = 0.2, epochs_ = epoch_prn, iterations = iter_prn, spr_all = spr_prn):

    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen
    datasize = dataset_helper.datasize

    best_initialization = None
    best_init_spr = 0
    best_init_test = 0

    if opt.test_algo:
        mutation_file = "./mutations/DLG4.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505small.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520small.csv"
        phenotype_name = "fitness"

    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    weights = dataset_helper.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datasize)  # change opt batch size
    train_dataset = Dataset(dataset_helper)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    dataset_helper.mutation_file_to_onehot(mutation_file)
    mutant_measures_name_list, measurement_list = preprocess_mutation_file(mutation_file, phenotype_name)

    spearmans = {}
    spearmans['det'] = []
    spearmans['stoch'] = []

    spr_epochs = []

    for test in range(nb_test):
        print("test {}".format(test))
        print("stochastic")

        model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                                  opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                                  opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                                  opt.has_temperature, opt.has_dictionary).to(device)
        optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)

        initialization_state = {'model': copy.deepcopy(model.state_dict()),
                                'optimizer': copy.deepcopy(optimizer.state_dict())}

        # checkpoint = {'model': Classifier(),
        #               'state_dict': model.state_dict(),
        #               'optimizer': optimizer.state_dict()}

        update_num = 0

        start = 0

        if opt.neff == 0:
            Neff = dataset_helper.Neff
        else:
            Neff = opt.Neff

        model.train()

        tmpr_spr = []


        for e in range(start, epochs_ + 1):

            for i, batch in enumerate(train_dataset_loader):
                update_num += 1
                warm_up_scale = 1.0
                if update_num < opt.warm_up:
                    warm_up_scale = update_num / opt.warm_up
                batch = batch.float().to(device)
                optimizer.zero_grad()
                mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                    batch)
                loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale, mu_W1, logsigma_W1, mu_b1,
                                    logsigma_b1, mu_W2,
                                    logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
                                    logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)

                loss.backward()
                optimizer.step()

            if e % spr_all == 0:
                model.eval()
                with torch.no_grad():
                    mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model,
                                                                                               N_pred_iterations=iterations)
                    spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_measures_name_list,
                                                   measurement_list)
                    tmpr_spr.append(spearman_r)
                    if test==0:
                        spr_epochs.append(e + 1)
                    if spearman_r > best_init_spr:
                        best_init_spr = spearman_r
                        best_initialization = copy.deepcopy(initialization_state)
                        best_initialization['spearman'] = spearman_r
                        best_init_test = test

                model.train()

        spearmans['stoch'].append(tmpr_spr)


        print("deterministic")

        model = DeterministicVAE(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                                  opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                                  opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                                  opt.has_temperature, opt.has_dictionary).to(device)
        optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)


        update_num = 0

        start = 0

        if opt.neff == 0:
            Neff = dataset_helper.Neff
        else:
            Neff = opt.Neff

        model.train()

        tmpr_spr = []

        for e in range(start, epochs_ + 1):

            for i, batch in enumerate(train_dataset_loader):
                update_num += 1
                warm_up_scale = 1.0
                if update_num < opt.warm_up:
                    warm_up_scale = update_num / opt.warm_up
                batch = batch.float().to(device)
                optimizer.zero_grad()
                mu, logsigma, px_z, logpx_z, z = model(
                    batch)
                loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale)

                loss.backward()
                optimizer.step()

            if e % spr_all == 0:
                model.eval()
                with torch.no_grad():
                    mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(model,
                                                                                               N_pred_iterations=iterations)
                    spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_measures_name_list,
                                                   measurement_list)
                    tmpr_spr.append(spearman_r)

                model.train()

        spearmans['det'].append(tmpr_spr)

    torch.save(best_initialization, opt.saving_path + "initialization2.pth")

    spearman_mean = {method: np.mean(spearmans[method], 0) for method in ['stoch', 'det']}

    plt.clf()
    plt.figure()
    # print(nb_test)
    # print(spearmans)
    for i in range(nb_test):
        plt.subplot(1, nb_test, i + 1)
        for method in ['stoch', 'det']:
            plt.plot(spr_epochs, spearmans[method][i], label = method)
        plt.legend(title = "Method:")
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
    plt.suptitle(
        "Comparing Stochastic and Deterministic Decoders")
    plt.savefig(
        "plt_stoch_det_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                          -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')

    plt.clf()
    plt.figure()
    for method in ['stoch', 'det']:
        plt.plot(spr_epochs, spearman_mean[method], label=method)
    plt.legend(title="Method:")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman")
    plt.title(
        "Comparing Stochastic and Deterministic Decoders (Mean of {} tests)".format(nb_test))
    plt.savefig(
        "plt_stoch_det_mean_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                                    -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')


def preprocess_mutation_file(mutation_filename, phenotype_name):
    measurement_df = pd.read_csv(mutation_filename, sep=',')

    mutant_measures_name_list = measurement_df.mutant.tolist()
    measurement_list = measurement_df[phenotype_name].tolist()

    return mutant_measures_name_list, measurement_list

def generate_spearman(mutant_name_list, delta_elbo_list, mutant_measures_name_list, measurement_list):

    # print(len(mutant_name_list))
    # print(mutant_name_list)
    # print(len(delta_elbo_list))
    # print(delta_elbo_list)

    mutant_name_to_pred = {mutant_name_list[i]: delta_elbo_list[i] for i in range(len(delta_elbo_list))}

    # If there are measurements
    wt_list = []
    preds_for_spearmanr = []
    measurements_for_spearmanr = []

    for i, mutant_name in enumerate(mutant_measures_name_list):
        expr_val = measurement_list[i]

        # Make sure we have made a prediction for that mutant
        if mutant_name in mutant_name_to_pred:
            multi_mut_name_list = mutant_name.split(':')

            # If there is no measurement for that mutant, pass over it
            if np.isnan(expr_val):
                pass

            # If it was a codon change, add it to the wt vals to average
            elif mutant_name[0] == mutant_name[-1] and len(multi_mut_name_list) == 1:
                wt_list.append(measurement_list[i])

            # If it is labeled as the wt sequence, add it to the average list
            elif mutant_name == 'wt' or mutant_name == 'WT':
                wt_list.append(measurement_list[i])

            else:
                measurements_for_spearmanr.append(expr_val)
                preds_for_spearmanr.append(mutant_name_to_pred[mutant_name])

    if wt_list != []:
        measurements_for_spearmanr.append(np.mean(wt_list))
        preds_for_spearmanr.append(0.0)

    num_data = len(measurements_for_spearmanr)
    spearman_r, spearman_pval = spearmanr(measurements_for_spearmanr, preds_for_spearmanr)
    # print ("N: " + str(num_data) + ", Spearmanr: " + str(spearman_r) + ", p-val: " + str(spearman_pval))
    return spearman_r

def generate_spearmanr(mutant_name_list, delta_elbo_list, mutation_filename, phenotype_name):
    measurement_df = pd.read_csv(mutation_filename, sep=',')

    mutant_list = measurement_df.mutant.tolist()
    expr_values_ref_list = measurement_df[phenotype_name].tolist()

    mutant_name_to_pred = {mutant_name_list[i]: delta_elbo_list[i] for i in range(len(delta_elbo_list))}
    # print("len(mutant_name_to_pred)")
    # print(len(mutant_name_to_pred))
    # If there are measurements
    wt_list = []
    preds_for_spearmanr = []
    measurements_for_spearmanr = []

    for i, mutant_name in enumerate(mutant_list):
        expr_val = expr_values_ref_list[i]

        # Make sure we have made a prediction for that mutant
        if mutant_name in mutant_name_to_pred:
            multi_mut_name_list = mutant_name.split(':')

            # If there is no measurement for that mutant, pass over it
            if np.isnan(expr_val):
                pass

            # If it was a codon change, add it to the wt vals to average
            elif mutant_name[0] == mutant_name[-1] and len(multi_mut_name_list) == 1:
                wt_list.append(expr_values_ref_list[i])

            # If it is labeled as the wt sequence, add it to the average list
            elif mutant_name == 'wt' or mutant_name == 'WT':
                wt_list.append(expr_values_ref_list[i])

            else:
                measurements_for_spearmanr.append(expr_val)
                preds_for_spearmanr.append(mutant_name_to_pred[mutant_name])

    if wt_list != []:
        measurements_for_spearmanr.append(np.mean(wt_list))
        preds_for_spearmanr.append(0.0)

    num_data = len(measurements_for_spearmanr)
    # print(measurements_for_spearmanr)
    # print(preds_for_spearmanr)
    spearman_r, spearman_pval = spearmanr(measurements_for_spearmanr, preds_for_spearmanr)
    print ("N: " + str(num_data) + ", Spearmanr: " + str(spearman_r) + ", p-val: " + str(spearman_pval))
    return spearman_r, spearman_pval

def test(dataset, theta):
    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen

    if opt.test_algo:
        mutation_file = "./mutations/DLG4.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520.csv"
        phenotype_name = "fitness"

    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                              opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)

    if opt.theano_test:
        print("theano-test")
        with open('DLG4_RAT_params_pytorch.pkl', 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
        model.load_state_dict(state_dict, strict=True)
    else:

        model.load_state_dict(torch.load(opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                                           -int(math.log10(opt.lr)), int(opt.neff), opt.epochs)))

    model.eval()
    with torch.no_grad():
        custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
            mutation_file, model, N_pred_iterations=1,
            filename_prefix="pred_{}_{}_{}_{}_{}_{}_epoch_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                                                  -int(math.log10(opt.lr)), opt.epochs))

        spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
                                       mutation_file, phenotype_name)


def train_main(dataset, theta):
    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen
    datasize = dataset_helper.datasize

    if opt.test_algo:
        mutation_file = "./mutations/DLG4.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505small.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520small.csv"
        phenotype_name = "fitness"
        
    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    weights = dataset_helper.weights
    # print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datasize) #change opt batch size
    train_dataset = Dataset(dataset_helper)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

    # print("datasetloader shape")
    # print(train_dataset.data.shape)



    model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                              opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    update_num = 0

    LB_list = []
    loss_params_list = []
    KLD_latent_list = []
    reconstruct_list = []

    # spearmans = []
    # pvals = []
    # epochs_spr = []

    titles = ["loss", "KLD_weights", "KLD_latent", "logpx_z"]

    start = 0
    
    # spr_virus = [1000, 5000, 10000, 15000]

    # if opt.continue_training > 0:
    #     model.load_state_dict(torch.load(opt.saving_path + "epoch_" + str(opt.continue_training)))
    #     start = opt.continue_training + 1

    if opt.neff == 0:
        Neff = dataset_helper.Neff
    else:
        Neff = opt.Neff

    model.train()

    if opt.test_algo:
        epochs = 2
    else:
        epochs = opt.epochs

    if opt.plot_spearman:
        mutant_sequences_one_hot, mutant_sequences_descriptor = dataset_helper.mutation_file_to_onehot(mutation_file)
        mutant_list, expr_values_ref_list = preprocess_mutation_file(mutation_file, phenotype_name)
        sprs = []
        spr_epochs_plot = []


    for e in range(start, epochs):
        # print("------------------------------")
        print(e)
        # print("len train loader")
        # print(len(train_dataset_loader))

        for i, batch in enumerate(train_dataset_loader):
            update_num += 1
            if opt.IWS_debug and update_num>30:
                return
            # print("batch:")
            # print(update_num)
            # print(batch.shape)
            warm_up_scale = 1.0
            if update_num < opt.warm_up:
                warm_up_scale = update_num / opt.warm_up
            batch = batch.float().to(device)
            # if opt.IWS:
            #     batch = batch.repeat(opt.k_IWS, 1, 1)
            optimizer.zero_grad()
            mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                batch)
            loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)
            # loss = total_loss_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)

            print("loss")
            print(loss)

            LB_list.append(loss.item())
            # print(loss.item())
            loss_params_list.append(
                (warm_up_scale * sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2,
                                               logsigma_b2,
                                               mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C,
                                               logsigma_C,
                                               mu_l, logsigma_l) / Neff).item())
            KLD_latent_list.append((warm_up_scale * kld_latent_theano(mu, logsigma).mean()).item())
            reconstruct_list.append(logpx_z.mean().item())
            loss.backward()
            if opt.IWS_debug:
                print("..............")
                for name, param in model.encoder.named_parameters():
                    if 'fc1' in name:
                        print(name)
                        print(param.grad.data.mean())
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-100, 100)
                for name, param in model.encoder.named_parameters():
                    if 'fc1' in name:
                        print(name)
                        print(param.grad.data.mean())
                print("...............")
                    # param.grad.data.clamp_(-1, 1)
            optimizer.step()

        if opt.plot_spearman:
            if e%opt.spearman_every == 0:
                model.eval()
                with torch.no_grad():
                    mutant_sequences_descriptor, delta_elbos = dataset_helper.pred_from_onehot(mutant_sequences_one_hot, mutant_sequences_descriptor, model, N_pred_iterations=500)
                    spearman_r = generate_spearman(mutant_sequences_descriptor, delta_elbos, mutant_list, expr_values_ref_list)
                    sprs.append(spearman_r)
                    spr_epochs_plot.append(e+1)
                model.train()


        # if (e + 1) in spr_virus:
        #     print(e)
        #     model.eval()
        #     with torch.no_grad():
        #         custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
        #                 mutation_file, model, N_pred_iterations=500)
        #
        #         spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
        #                                            mutation_file, phenotype_name)
        #
        #         spearmans.append(spr)
        #         pvals.append(pval)
        #         epochs_spr.append(e)

    # model.eval()
    # with torch.no_grad():
    #     custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
    #     mutation_file, model, N_pred_iterations=500,
    #     filename_prefix="pred_{}_{}_{}_{}_{}_epoch_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size,
    #                                               -int(math.log10(opt.lr)), opt.epochs))
    #
    #     spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
    #                                    mutation_file, phenotype_name)

        # spearmans.append(spr)
        # pvals.append(pval)
        # epochs_spr.append(opt.epochs)

    if opt.plot_spearman:
        plt.clf()
        plt.plot(spr_epochs_plot, sprs)
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
        plt.savefig("spr")
        plt.close('all')


    torch.save(model.state_dict(), opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr)), int(opt.neff), opt.epochs))


    # plt.clf()
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_spr, spearmans)
    # plt.title("Spearman")
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_spr, pvals)
    # plt.title("p-val")
    # plt.suptitle(
    #     "ds: {}, t: {}, ld: {}, bs: {}, lr: {}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, opt.lr))
    # plt.savefig(
    #     "spr_{}_{}_{}_{}_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr))))
    # plt.close('all')
    plots = [LB_list, loss_params_list, KLD_latent_list, reconstruct_list]
    plt.clf()
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(np.arange(len(plots[i])), plots[i])
        plt.title(titles[i])
    plt.suptitle("ds: {}, t: {}, ld: {}, bs: {}, lr: {}, e: {}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, opt.lr, opt.epochs))
    plt.savefig(
        "plt_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')





if __name__ == "__main__":
    main()
