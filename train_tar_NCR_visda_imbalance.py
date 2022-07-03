import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx, ImageList_idx_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import datetime
from randaugment import RandAugmentMC
from copy import deepcopy
import datetime


class StrongTransform(object):
    def __init__(self, mean, std):
        self.weak = None
        self.strong = None
        self.normalize = None

    def __call__(self, x):
        strong = self.strong(x)
        return self.weak(x), self.normalize(strong)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)
    log_str = str({len(dsets["target"])}) \
              + ' samples are loaded for train and ' + str({len(dsets["test"])}) + ' for test' \
              + 'at ' + str(datetime.datetime.now().strftime('%m-%d-%H-%M'))
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return dset_loaders


def data_load2(args, confident_l, unconfident_l):
    ## prepare data
    new_dsets = {}
    new_dset_loaders = {}
    train_bs = args.batch_size

    new_dsets["confident"] = ImageList_idx_idx(confident_l,
                                           transform=StrongTransform(mean=(0.485, 0.456, 0.406),
                                                                       std=(0.229, 0.224, 0.225)),
                                           make_dataset_=False)
    new_dset_loaders["confident"] = DataLoader(new_dsets["confident"], batch_size=train_bs // args.bs_factor,
                                               shuffle=True,
                                               num_workers=args.worker,
                                               drop_last=False)
    new_dsets["unconfident"] = ImageList_idx_idx(unconfident_l, transform=image_train(), make_dataset_=False)
    new_dset_loaders["unconfident"] = DataLoader(new_dsets["unconfident"],
                                                 batch_size=(train_bs // args.bs_factor) * (args.bs_factor - 1),
                                                 shuffle=True, num_workers=args.worker, drop_last=False)
    log_str = f'{len(new_dsets["confident"])} confident samples, {len(new_dsets["unconfident"])} unconfident ones,' \
              f'total: {len(new_dsets["confident"]) + len(new_dsets["unconfident"])}'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return new_dset_loaders


def cal_acc(loader, netF, netB, netC, args, flag=False, save_feature=False):
    if args.use_prototype:
        with torch.no_grad():
            w = netC.fc.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            netC.fc.weight.copy_(w)
            if save_feature:
                np.save(osp.join(args.output_dir, 'prototype.npy'), w.cpu().numpy())

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if not args.use_prototype:
                outputs = netC(netB(netF(inputs)))
            else:
                outputs = netC(F.normalize(netB(netF(inputs)), dim=1)) / args.temperature
            if save_feature:
                feature = netB(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
                if save_feature:
                    all_features = feature.float().cpu()
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                if save_feature:
                    all_features = torch.cat((all_features, feature.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).data.item()
    if save_feature:
        np.save(osp.join(args.output_dir, 'features.npy'), all_features.numpy())
        np.save(osp.join(args.output_dir, 'labels.npy'), all_label.numpy())

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, matrix
    else:
        return accuracy * 100, mean_ent


def test(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    if not args.use_prototype:
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    else:
        netC = network.feat_classifier_proto(type=args.layer, class_num=args.class_num,
                                             bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_target + '/target_F_' + args.savename + '.pt'
    # modelpath = args.output_dir_target + '/source_F' + '.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_target + '/target_B_' + args.savename + '.pt'
    # modelpath = args.output_dir_target + '/source_B' + '.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_target + '/target_C_' + args.savename + '.pt'
    # modelpath = args.output_dir_target + '/source_C' + '.pt'
    netC.load_state_dict(torch.load(modelpath))

    np.set_printoptions(threshold=np.inf)
    netF.eval()
    netB.eval()
    netC.eval()
    if args.dset == 'VISDA-C':
        acc_s_te, acc_list, con_mat = cal_acc(dset_loaders['test'], netF, netB, netC, args, True, save_feature=True)
        log_str = 'Task: {}; Accuracy = {:.2f}%'.format(args.name, acc_s_te) + '\n' + acc_list
        log_str += '\n' + 'The confusion matrix is:\n{}'.format(con_mat)

    else:
        acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False or args.source_balance, save_feature=True)
        log_str = 'Task: {}; Accuracy = {:.2f}%'.format(args.name, acc_s_te)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    np.set_printoptions(threshold=None)


def train_target_wu(args):  # train with warmup
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    if not args.use_prototype:
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    else:
        netC = network.feat_classifier_proto(type=args.layer, class_num=args.class_num,
                                             bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.bottleneck).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        if args.use_prototype:
            w = netC.fc.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            netC.fc.weight.copy_(w)
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output, dim=1)
            if args.use_prototype:
                outputs = netC(output_norm) / args.temperature
            else:
                outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone()
            score_bank[indx] = outputs.detach().clone()  # .cpu()
            bs = outputs.shape[0]

    max_iter = args.warmup_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        tar_idx = tar_idx.cuda()

        iter_num += 1
        if iter_num <= 15 * len(dset_loaders["target"]):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=15 * len(dset_loaders["target"]))
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=15 * len(dset_loaders["target"]))
        else:
            lr_scheduler(optimizer, iter_num=15 * len(dset_loaders["target"]), max_iter=15 * len(dset_loaders["target"]))
            lr_scheduler(optimizer_c, iter_num=15 * len(dset_loaders["target"]), max_iter=15 * len(dset_loaders["target"]))

        features_test = netB(netF(inputs_test))
        if args.use_prototype:
            with torch.no_grad():
                w = netC.fc.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                netC.fc.weight.copy_(w)
            outputs_test = netC(F.normalize(features_test, dim=1)) / args.temperature
        else:
            outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        bs = softmax_out.shape[0]

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

            fea_near = fea_bank[idx_near]  # batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
            _, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(match > 0., match*0.1, torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            weight_kk[idx_near_near == tar_idx_] = 0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            # print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, 12)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x C x 1
        const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))
        # kl_div here equals to dot product since we do not use log for score_near_kk (following NRC (NeuraIPS'21))
        NN_NN_loss = const

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C
        NN_loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        entropy_loss = - torch.mean((softmax_out * score_self).sum(-1))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))

        total_loss = (NN_NN_loss + NN_loss) * args.use_lc + entropy_loss * args.use_self + gentropy_loss * args.use_div

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            np.set_printoptions(threshold=np.inf)
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list, con_mat = cal_acc(dset_loaders['test'], netF, netB, netC, args, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
                log_str += '\n' + 'The confusion matrix is:\n{}'.format(con_mat)
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            if acc_s_te > acc_log:
                acc_log = acc_s_te
                if args.issave:
                    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()
            np.set_printoptions(threshold=None)

    return netF, netB, netC, dset_loaders, fea_bank, score_bank, score_bank_small


def train_target_sa_v2(netF, netB, netC, dset_loaders, fea_bank, score_bank, args):  # train with strong augment v2
    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    acc_log = 0
    for ep in range(args.max_epoch - args.warmup_epoch):
        netF, netB, netC = netF.eval(), netB.eval(), netC.eval()
        confident_l, unconfident_l = generate_split(dset_loaders["test"], netF, netB, netC, args)
        if ep == 0:
            previous_confident_l = set([e[-1] for e in confident_l])
        else:
            current_confident_l = set([e[-1] for e in confident_l])
            log_str = "{}({:.2f}%) samples are still confident as the previous epoch.".\
                format(len(current_confident_l & previous_confident_l),
                       len(current_confident_l & previous_confident_l)/len(current_confident_l)*100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            previous_confident_l = current_confident_l
        netF, netB, netC = netF.train(), netB.train(), netC.train()
        train_dset_loaders = data_load2(args, confident_l, unconfident_l)

        max_iter = max(len(train_dset_loaders["confident"]), len(train_dset_loaders["unconfident"]))
        interval_iter = max_iter
        iter_num = 0

        while iter_num < max_iter:
            try:
                (inputs_con_w, inputs_con_s), con_p_l, con_idx, _ = iter_con.next()  # con_p_l is the pseudo label of the confident sample
            except:
                iter_con = iter(train_dset_loaders["confident"])
                (inputs_con_w, inputs_con_s), con_p_l, con_idx, _ = iter_con.next()

            try:
                inputs_unc, unc_p_l, unc_idx, _ = iter_unc.next()
                # unc_p_l is the pseudo label computed as SHOT for unconfident sample
            except:
                iter_unc = iter(train_dset_loaders["unconfident"])
                inputs_unc, unc_p_l, unc_idx, _ = iter_unc.next()

            inputs_con_w, inputs_con_s = inputs_con_w.cuda(), inputs_con_s.cuda()
            inputs_unc = inputs_unc.cuda()
            tar_idx = torch.cat((con_idx, unc_idx), dim=0).cuda()

            iter_num += 1
            if ep < 15:
                lr_scheduler(optimizer, iter_num=iter_num + ep * max_iter, max_iter=15 * max_iter)
                lr_scheduler(optimizer_c, iter_num=iter_num + ep * max_iter,
                             max_iter=15 * max_iter)
            else:
                lr_scheduler(optimizer, iter_num=15 * max_iter, max_iter=15 * max_iter)
                lr_scheduler(optimizer_c, iter_num=15 * max_iter, max_iter=15 * max_iter)

            features_test = netB(netF(torch.cat((inputs_con_s, inputs_con_w, inputs_unc), dim=0)))
            if args.use_prototype:
                with torch.no_grad():
                    w = netC.fc.weight.data.clone()
                    w = F.normalize(w, dim=1, p=2)
                    netC.fc.weight.copy_(w)
                outputs_test = netC(F.normalize(features_test, dim=1)) / args.temperature
            else:
                outputs_test = netC(features_test)
            bs = con_idx.shape[0] + unc_idx.shape[0]
            bs_s = con_idx.shape[0]
            softmax_out = nn.Softmax(dim=1)(outputs_test)

            features_test_s = features_test[:bs_s]
            features_test_w = features_test[bs_s:]
            softmax_out_s = softmax_out[:bs_s]
            softmax_out_w = softmax_out[bs_s:]

            with torch.no_grad():
                output_f_norm = F.normalize(features_test_w)
                output_f_ = output_f_norm.detach().clone()

                fea_bank[tar_idx] = output_f_.detach().clone()
                score_bank[tar_idx] = softmax_out_w.detach().clone()

                score_bank_small[bs:] = score_bank_small[:-bs].clone()  # remove last samples
                score_bank_small[:bs] = softmax_out_w.detach().clone()  # update in the head

                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

                fea_near = fea_bank[idx_near]  # batch x K x num_dim
                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                _, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=args.KK + 1)  # M near neighbors for each of above K ones
                idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
                match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
                weight = torch.where(match > 0., match*0.1, torch.ones_like(match).fill_(0.1))  # batch x K

                weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK)  # batch x K x M
                weight_kk = weight_kk.fill_(0.1)

                # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                weight_kk[idx_near_near == tar_idx_] = 0

                score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                # print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # batch x KM

                score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, 12)  # batch x KM x C

                score_self = score_bank[tar_idx]

            # nn of nn
            output_re = softmax_out_w.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x KM x C
            const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))
            # kl_div here equals to negative element-wise product since we do not use log for score_near_kk (following NRC (NeuraIPS'21))
            # if reduction='none', then kl_div(x,y) = y*(logy-x)=ylogy-yx for every element x and y
            # since y is from a tensor after torch.detach(), ylogy just a negligible constant
            NN_NN_loss = const

            # nn
            softmax_out_un = softmax_out_w.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C
            NN_loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

            # self, if not explicitly removing the self feature in expanded neighbor then no need for this
            entropy_loss = - torch.mean((softmax_out_w * score_self).sum(-1))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))

            NN_NN_s_loss = torch.mean((F.kl_div(softmax_out_s.unsqueeze(1).expand(-1, args.K*args.KK, -1),
                                                score_near_kk[:bs_s], reduction='none').sum(-1) * weight_kk[:bs_s].cuda()).sum(1))
            NN_s_loss = torch.mean((F.kl_div(softmax_out_s.unsqueeze(1).expand(-1, args.K, -1),
                                             score_near[:bs_s], reduction='none').sum(-1) * weight[:bs_s].cuda()).sum(1))
            match_loss = - torch.mean((softmax_out_s * score_self[:bs_s]).sum(dim=1))

            total_loss = (NN_NN_loss + NN_loss) * args.use_lc + entropy_loss * args.use_self + gentropy_loss * args.use_div\
                         + match_loss * args.use_cv + (NN_NN_s_loss + NN_s_loss) * args.use_lcr

            optimizer.zero_grad()
            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                np.set_printoptions(threshold=np.inf)
                netF.eval()
                netB.eval()
                netC.eval()
                if args.dset == 'VISDA-C':
                    acc_s_te, acc_list, con_mat = cal_acc(dset_loaders['test'], netF, netB, netC, args, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                                acc_s_te) + '\n' + acc_list
                    log_str += '\n' + 'The confusion matrix is:\n{}'.format(con_mat)
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

                if acc_s_te > acc_log:
                    acc_log = acc_s_te
                    if args.issave:
                        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
                    print('best acc: ', acc_log, ' at epoch: ', ep)
                    args.out_file.write('best acc: ' + str(acc_log) + ' at epoch: ' + str(ep) + '\n')
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str + '\n')
                netF.train()
                netB.train()
                netC.train()
                np.set_printoptions(threshold=None)

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


@torch.no_grad()
def generate_split(loader, netF, netB, netC, args):
    # should set model.eval() before calling this function
    pass
    return confident_l, unconfident_l


if __name__ == "__main__":
    nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--KK', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--issave', type=bool_flag, default=True)

    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output_dir_src', type=str, default='')
    parser.add_argument('--use_prototype', type=bool_flag, default=False)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--just_eval', type=bool_flag, default=False)
    parser.add_argument('--output_dir_target', type=str, default='')
    parser.add_argument('--update_bn_ep', type=int, default=0)
    parser.add_argument('--threshold', type=int, default=-1)

    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--bs_factor', type=int, default=8,
                        help='bs_factor = num. unconfident sample / num. confident sample in batch')
    parser.add_argument('--portion', type=float, default=0.125,
                        help='select how many portion of samples as confident samples')
    parser.add_argument('--uniform_sample', type=bool_flag, default=False)
    parser.add_argument('--fixmatch', type=bool_flag, default=False)
    parser.add_argument('--source_balance', type=bool_flag, default=False)
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--RS_UT_factor', type=int, default=100, choices=[10, 50],
                        help='refer to the experimental setting in ISFDA')

    parser.add_argument('--use_lc', type=bool_flag, default=True)
    parser.add_argument('--use_self', type=bool_flag, default=True)
    parser.add_argument('--use_div', type=bool_flag, default=True)
    parser.add_argument('--use_lcr', type=bool_flag, default=True)
    parser.add_argument('--use_cv', type=bool_flag, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12

    args.interval = args.max_epoch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        if args.folder == '':
            folder = ''
        else:
            folder = args.folder

        if not args.source_balance:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        else:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            if args.RS_UT_factor != 100:
                args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS_' + str(args.RS_UT_factor) + '.txt'
                args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT_' + str(args.RS_UT_factor) + '.txt'
                args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT_' + str(args.RS_UT_factor) + '.txt'

        if args.output_dir_src == '':
            args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper(), nowTime)
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)

        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        print(print_args(args))
        if args.just_eval and args.output_dir_target != '':
            test(args)
        else:
            if args.fixmatch:
                netF, netB, netC, dset_loaders, fea_bank, score_bank, score_bank_small = train_target_wu(args)
                train_target_sa_v2(netF, netB, netC, dset_loaders, fea_bank, score_bank, score_bank_small, args)
