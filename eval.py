from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import logging

from vehiclereid.utils.distance import distance
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision, QuantLoss
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import setup_logger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
from vehiclereid.utils import results_to_excel, make_dirs, visualize_ranked_results
from vehiclereid.datasets import init_imgreid_dataset




def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False, dataset_name = None, dataset =None):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            print('features have shape',features.shape)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0).sign()
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0).sign()
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = distance(qf,gf, dist_type= 'inner_product', pair= True)

    visualize_ranked_results(distmat,(dataset.gallery,dataset.query),args=args)
    print(distmat)
    print("distance matrix has shape,",distmat.shape)
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids,dataset_name= dataset_name)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc, mAP

parser = argument_parser()
args = parser.parse_args()
#global args
set_random_seed(args.seed)
if not args.use_avai_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()
if args.use_cpu:
    use_gpu = False

print('==========\nArgs:{}\n=========='.format(args))
output_path = './out/base'
make_dirs(output_path)
logger = setup_logger("{}_{}_{}_train".format(args.target_names, args.arch, args.hash_bit_number),
                      output_path, if_train=True)

if use_gpu:
    print('Currently using GPU {}'.format(args.gpu_devices))
    cudnn.benchmark = True
else:
    warnings.warn('Currently using CPU, however, GPU is highly recommended')

print('Initializing image data manager')
dm = ImageDataManager(use_gpu, **dataset_kwargs(args))

dataset = init_imgreid_dataset(root=args.root,name =args.source_names[0])
trainloader, testloader_dict = dm.return_dataloaders()
model_save_dir = args.save_dir
model_save_dir = model_save_dir + '{}_{}_{}.pth'.format(args.source_names[0],args.hash_bit_number,'59')
print('Initializing model: {}'.format(args.arch))
model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu, hash_bit = args.hash_bit_number)
print('Model size: {:.3f} M'.format(count_num_param(model)))
fpath = osp.join(model_save_dir, 'model.pth.tar-' + str('60'))
model = model.cuda()
load_pretrained_weights(model, fpath)

for name in args.target_names:
    queryloader = testloader_dict[name]['query']
    galleryloader = testloader_dict[name]['gallery']
    cmc, mAP = test(model, queryloader, galleryloader, use_gpu, dataset_name= args.target_names[0], dataset= dataset)
    print(
        'The cmc: Rank1:{},Rank2:{},Rank3:{},Rank4:{} Rank5:{},Rank6:{},Rank7:{},Rank8:{},Rank9:{}, Rank10:{}'
        'Rank11:{},Rank12:{},Rank13:{},Rank14{},Rank15:{},Rank16:{},Rank17:{},Rank18:{},Rank19:{},Rank20:{},mAP is {}'.format(
            cmc[0], cmc[1], cmc[2], cmc[3], cmc[4], cmc[5], cmc[6], cmc[7], cmc[8], cmc[9], cmc[10],
            cmc[11], cmc[12],
            cmc[13], cmc[14],
            cmc[15], cmc[16], cmc[17], cmc[18], cmc[19], mAP))
    results = [item for item in cmc[:20]] + [mAP]
    model_name = 'VehicleNet-{}'.format(args.hash_bit_number)
    results_to_excel(results, model_name, args.target_names[0])


























