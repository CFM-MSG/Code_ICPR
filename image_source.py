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
from data_list import ImageList, ImageListBalanced
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.nn.functional as F
import datetime


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
        transforms.ToTensor(),
        normalize
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
        transforms.ToTensor(),
        normalize
    ])


def getClassDict(total_list):
    cls_dict = dict()
    for l in total_list:
        if not int(l.split(' ')[1]) in cls_dict:
            cls_dict[int(l.split(' ')[1])] = []
        cls_dict[int(l.split(' ')[1])].append(l)

    return cls_dict


def getSampleDict(total_cls_dict, per_cls_percentage, return_both=False):
    sample_cls_dict = dict()
    others = dict()
    for k in total_cls_dict.keys():
        if not k in sample_cls_dict:
            sample_cls_dict[k] = []
            others[k] = []
        this_cls_num = len(total_cls_dict[k])
        val_num = max(int(this_cls_num * per_cls_percentage), 1)
        tr, te = torch.utils.data.random_split(total_cls_dict[k], [this_cls_num - val_num, val_num])
        sample_cls_dict[k] = te
        others[k] = tr
        # random.sample(total_cls_dict[k], val_num)
    if return_both:
        return sample_cls_dict, others
    else:
        return sample_cls_dict


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    if args.val_dset_path != '':
        txt_val = open(args.val_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        if args.val_dset_path == '':
            cls_dict = getClassDict(txt_src)
            val_sample_cls_dict, tr_sample_cls_dict = getSampleDict(cls_dict, 0.1, return_both=True)
            te_txt = []
            tr_txt = []
            for k in val_sample_cls_dict.keys():
                te_txt.extend(val_sample_cls_dict[k])
                tr_txt.extend(tr_sample_cls_dict[k])
            # tr_txt = list(set(txt_src) - set(te_txt))
            # dsize = len(txt_src)
            # tr_size = int(0.9 * dsize)
            # # print(dsize, tr_size, dsize - tr_size)
            # tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        else:
            tr_txt = txt_src
            te_txt = txt_val
    else:  # recommend this branch for RSUT datasets
        if args.val_dset_path == '':
            cls_dict = getClassDict(txt_src)
            val_sample_cls_dict = getSampleDict(cls_dict, 0.1)
            te_txt = []
            for k in val_sample_cls_dict.keys():
                te_txt.extend(val_sample_cls_dict[k])
            # dsize = len(txt_src)
            # tr_size = int(0.9 * dsize)
            # _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
            tr_txt = txt_src
        else:
            tr_txt = txt_src
            te_txt = txt_val
    if not args.source_balance:
        dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    else:
        dsets["source_tr"] = ImageListBalanced(tr_txt, transform=image_train(), class_num=args.class_num)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)
    log_str = f'{len(dsets["source_tr"])} samples are loaded for train, {len(dsets["source_te"])} for validation' \
              f' and {len(dsets["test"])} for test'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return dset_loaders


def cal_acc(loader, netF, netB, netC, args, flag=False):
    if args.use_prototype:
        with torch.no_grad():
            w = netC.fc.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            netC.fc.weight.copy_(w)

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
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_acc_oda(loader, netF, netB, netC, args):
    if args.use_prototype:
        with torch.no_grad():
            w = netC.fc.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            netC.fc.weight.copy_(w)

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
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1, 1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent > threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int), :]

    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    if not args.use_prototype:
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    else:
        netC = network.feat_classifier_proto(type=args.layer, class_num=args.class_num,
                                             bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = 50
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        if not args.use_prototype:
            outputs_source = netC(netB(netF(inputs_source)))
            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                       labels_source)
        else:
            # normalize the prototypes of the classifier
            with torch.no_grad():
                w = netC.fc.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                netC.fc.weight.copy_(w)
            outputs_source = netC(F.normalize(netB(netF(inputs_source)), dim=1)) / args.temperature
            # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                       labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, args, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, args, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

            netF.train()
            netB.train()
            netC.train()

    # torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    # torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    # torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    if not args.use_prototype:
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    else:
        netC = network.feat_classifier_proto(type=args.layer, class_num=args.class_num,
                                             bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC, args)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name,
                                                                                            acc_os2, acc_os1,
                                                                                            acc_unknown)
    else:
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, args, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False or args.source_balance)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


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


if __name__ == "__main__":
    nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')

    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domainnet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    parser.add_argument('--use_prototype', type=bool_flag, default=False)
    parser.add_argument('--output_dir_src', type=str, default='')
    parser.add_argument('--source_balance', type=bool_flag, default=False)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--RS_UT_factor', type=int, default=100, choices=[10, 50],
                        help='refer to the experimental setting in ISFDA')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 40
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = None
    if not args.source_balance:
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.val_dset_path = ''
    else:
        if args.dset == 'domainnet':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
        else:
            # Art in office-home is not used for class-imbalanced setting, which means that s != 0
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'

        if args.dset == 'office-home':
            args.val_dset_path = folder + args.dset + '/' + names[args.s] + '_BS.txt'
        elif args.dset == 'domainnet':
            args.val_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini_val.txt'
        else:
            args.val_dset_path = ''

        if args.RS_UT_factor != 100 and args.dset == 'VISDA-C':  # for ablation
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS_' + str(args.RS_UT_factor) + '.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT_' + str(args.RS_UT_factor) + '.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    if args.output_dir_src == '':
        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper(), nowTime)
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        if args.source_balance and args.dset == 'office-home' and names[i] == 'Art':
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = None
        if not args.source_balance:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        else:
            if args.dset == 'domainnet':
                args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
                args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
            else:
                # Art in office-home is not used for class-imbalanced setting, which means that s != 0
                args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
                args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            if args.RS_UT_factor != 100 and args.dset == 'VISDA-C':  # for ablation
                args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS_' + str(args.RS_UT_factor) + '.txt'
                args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT_' + str(
                    args.RS_UT_factor) + '.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        test_target(args)
