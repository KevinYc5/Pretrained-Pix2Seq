# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
encode functions used in main.py
"""
import math
import os
import sys

import argparse
import random
from pathlib import Path
from tqdm import tqdm     

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
# from models import build_model
from playground import build_all_model

@torch.no_grad()
def encode(model, data_loader, device, output_dir):
    model.eval()

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    i = 0
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        src, mask, target, pos = model.module.encoder([samples, targets])

        # print("src[0]: ", src[0].shape)   # feature: torch.Size([256, 32, 42])
        # print("mask: ", mask.shape) # F.interpolate of m
        # print("target: ", target)   # object detection gth
        # print("pos: ", pos.shape)

        
        feature_np = src[0].cpu().numpy()
        _output = os.path.join(output_dir,"{idx}.npy".format(idx=i))
        np.save(_output, feature_np)
        # sys.exit()
        i = i+1


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_all_model[args.model](args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    n_parameters = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print('number of encoder params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'ap' in checkpoint:
            cur_ap = checkpoint['ap']
            max_ap = checkpoint['max_ap']


    encode(model, data_loader_val, device, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Seq training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    
